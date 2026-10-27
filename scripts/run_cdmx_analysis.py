"""Fit CDMX-only models and forecast a configurable future rolling horizon."""

import argparse
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf

from siniestros_riesgo.config import FIGURES, MODELS, PROCESSED, PROJECT_CONFIG, RAW, TABLES, load_yaml, setup_logging
from siniestros_riesgo.evaluation import count_metrics, block_bootstrap_difference

SEED=42
logger=setup_logging("run_cdmx_analysis")
TABLES.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
cfg=load_yaml(PROJECT_CONFIG)
default_start=(pd.Timestamp.now(tz="America/Mexico_City").tz_localize(None).to_period("M")+1)
parser=argparse.ArgumentParser(description="Pronóstico móvil de siniestros con víctimas por alcaldía")
parser.add_argument("--forecast-start",default=str(default_start),help="Primer mes futuro, formato AAAA-MM")
parser.add_argument("--horizon-months",type=int,default=int(cfg.get("forecast_horizon_months",12)))
args=parser.parse_args()
forecast_start=pd.Period(args.forecast_start,freq="M")
if args.horizon_months<1: raise SystemExit("--horizon-months debe ser positivo")
forecast_end=forecast_start+args.horizon_months-1
months_es=("enero","febrero","marzo","abril","mayo","junio","julio","agosto","septiembre","octubre","noviembre","diciembre")
def period_label(period): return f"{months_es[period.month-1]} de {period.year}"
horizon_label=f"{period_label(forecast_start)} a {period_label(forecast_end)}"

# Panel exclusivo de las 16 demarcaciones territoriales.
panel=pd.read_parquet(PROCESSED/"municipality_month_panel.parquet").query("state_code == '09'").copy()
catalog=pd.read_csv("data/external/inegi_municipal_catalog.csv",dtype={"cvegeo":"string"}); catalog["cvegeo"]=catalog.cvegeo.str.zfill(5); catalog["population"]=pd.to_numeric(catalog.pob_total,errors="coerce")
panel=panel.merge(catalog[["cvegeo","nomgeo","population"]],left_on="geo_id",right_on="cvegeo",how="left",validate="many_to_one").drop(columns="cvegeo")
if panel.geo_id.nunique()!=16: raise SystemExit(f"Se esperaban 16 alcaldías, se encontraron {panel.geo_id.nunique()}")
origin_year=min(cfg["years"])
panel["month_sin"]=np.sin(2*np.pi*panel.month/12); panel["month_cos"]=np.cos(2*np.pi*panel.month/12); panel["time_index"]=(panel.year-origin_year)*12+panel.month-1

# Vecindad Queen oficial de las alcaldías; el rezago usa únicamente el mes anterior.
geo=gpd.read_file("data/external/cdmx_alcaldias_inegi.geojson").sort_values("cvegeo").reset_index(drop=True)
w=Queen.from_dataframe(geo,use_index=False); neighbors={geo.loc[i,"cvegeo"]:[geo.loc[j,"cvegeo"] for j in js] for i,js in w.neighbors.items()}
lookup=panel.set_index(["geo_id","date"])["lag_1"]
def spatial_value(row):
    values=[lookup.get((code,row.date),np.nan) for code in neighbors[row.geo_id]]
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
panel["spatial_lag_1"]=panel.apply(spatial_value,axis=1)

# Congelar todos los rezagos del test al cierre del periodo de entrenamiento.
train_years=cfg["train_years"]; test_years=cfg["test_years"]; train_end=max(train_years)
frozen=panel.copy(); test_mask=frozen.year.isin(test_years)
last_train=frozen[(frozen.year==train_end)&(frozen.month==12)].set_index("geo_id").victim_crashes
mean_train=frozen[frozen.year==train_end].groupby("geo_id").victim_crashes.mean()
spatial_train={code:float(np.mean([last_train[n] for n in neighbors[code]])) for code in last_train.index}
frozen.loc[test_mask,"lag_1"]=frozen.loc[test_mask,"geo_id"].map(last_train); frozen.loc[test_mask,"mean_lag_12"]=frozen.loc[test_mask,"geo_id"].map(mean_train); frozen.loc[test_mask,"spatial_lag_1"]=frozen.loc[test_mask,"geo_id"].map(spatial_train)
panel.to_parquet(PROCESSED/"cdmx_alcaldia_month_panel.parquet",index=False); frozen.to_parquet(PROCESSED/"cdmx_alcaldia_month_panel_frozen.parquet",index=False)

features=["lag_1","mean_lag_12","spatial_lag_1","month_sin","month_cos","time_index","geo_id"]
numeric=features[:-1]
prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("alcaldia",OneHotEncoder(handle_unknown="ignore"),["geo_id"])])
candidates={
 "random_forest":[RandomForestRegressor(n_estimators=300,min_samples_leaf=m,criterion="poisson",n_jobs=-1,random_state=SEED) for m in (2,5,10)],
 "xgboost_poisson":[XGBRegressor(objective="count:poisson",n_estimators=300,max_depth=d,learning_rate=.04,subsample=.85,colsample_bytree=.85,n_jobs=-1,random_state=SEED) for d in (2,3,4)]}

def nb_frame(data):
    transformed=data.copy()
    for col in ("lag_1","mean_lag_12","spatial_lag_1"):
        transformed["log_"+col]=np.log1p(transformed[col].fillna(0).clip(lower=0))
    return transformed

def fit_nb(data):
    return smf.glm(
        "victim_crashes ~ log_lag_1 + log_mean_lag_12 + log_spatial_lag_1 + month_sin + month_cos + time_index + C(geo_id)",
        data=nb_frame(data), family=sm.families.NegativeBinomial(alpha=1),
    ).fit(maxiter=100)

def predict_nb(model,data):
    return np.clip(np.asarray(model.predict(nb_frame(data))),0,None)

validation=[]; selected={}; selected_options={}
for name,options in candidates.items():
    means=[]
    for option,estimator in enumerate(options):
        scores=[]
        for valid_year in cfg["validation_years"]:
            fit=panel[panel.year<valid_year]; valid=panel[panel.year==valid_year]
            model=Pipeline([("prep",clone(prep)),("model",clone(estimator))]); model.fit(fit[features],fit.victim_crashes); score=mean_absolute_error(valid.victim_crashes,np.clip(model.predict(valid[features]),0,None)); scores.append(score); validation.append({"model":name,"option":option,"validation_year":valid_year,"mae":score})
        means.append(np.mean(scores))
    chosen=int(np.argmin(means)); selected[name]=options[chosen]; selected_options[name]=chosen
for valid_year in cfg["validation_years"]:
    fit=panel[panel.year<valid_year]; valid=panel[panel.year==valid_year]
    model=fit_nb(fit); score=mean_absolute_error(valid.victim_crashes,predict_nb(model,valid))
    validation.append({"model":"negative_binomial","option":0,"validation_year":valid_year,"mae":score})
validation_frame=pd.DataFrame(validation)
selected_options["negative_binomial"]=0
validation_frame["selected_option"]=[row.option==selected_options[row.model] for row in validation_frame.itertuples()]
validation_frame.to_csv(TABLES/"cdmx_temporal_validation.csv",index=False)
selected_forecast_model=(validation_frame[validation_frame.selected_option].groupby("model").mae.mean().sort_values().index[0])
if selected_forecast_model=="random_forest":
    selected_parameters={key:selected[selected_forecast_model].get_params()[key] for key in ("n_estimators","min_samples_leaf","criterion")}
elif selected_forecast_model=="xgboost_poisson":
    selected_parameters={key:selected[selected_forecast_model].get_params()[key] for key in ("n_estimators","max_depth","learning_rate","subsample","colsample_bytree","objective")}
else:
    selected_parameters={"family":"NegativeBinomial","alpha":1}

train=panel[panel.year.isin(train_years)].copy(); test=frozen[frozen.year.isin(test_years)].copy(); predictions=test[["geo_id","nomgeo","year","month","victim_crashes"]].copy(); predictions["baseline"]=test.mean_lag_12.to_numpy()
for name,estimator in selected.items():
    model=Pipeline([("prep",clone(prep)),("model",estimator)]); model.fit(train[features],train.victim_crashes); predictions[name]=np.clip(model.predict(test[features]),0,None); joblib.dump(model,MODELS/f"cdmx_{name}.joblib")
# NB2 con rezagos logarítmicos para evitar extrapolación explosiva.
nb=fit_nb(train)
predictions["negative_binomial"]=predict_nb(nb,test); joblib.dump(nb,MODELS/"cdmx_negative_binomial.joblib")
metrics=[]
for name in ("baseline","negative_binomial","random_forest","xgboost_poisson"): metrics.append({"model":name,**count_metrics(predictions.victim_crashes.to_numpy(),predictions[name].to_numpy())})
pd.DataFrame(metrics).to_csv(TABLES/"cdmx_test_metrics.csv",index=False); predictions.to_csv(TABLES/"cdmx_test_predictions.csv",index=False)
boot=[]
for name in ("negative_binomial","random_forest","xgboost_poisson"):
    boot.append({"model":name,"metric":"mae_model_minus_baseline",**block_bootstrap_difference(predictions,lambda y,p:mean_absolute_error(y,p),name,"baseline",1000,SEED)})
pd.DataFrame(boot).to_csv(TABLES/"cdmx_bootstrap_differences.csv",index=False)

# Reajuste con toda la historia observada y proyección recursiva hasta el fin
# del horizonte. Los meses entre el último dato y forecast_start son puente.
if selected_forecast_model=="negative_binomial":
    final_model=fit_nb(panel)
    def final_predict(frame): return predict_nb(final_model,frame)
else:
    final_model=Pipeline([("prep",clone(prep)),("model",clone(selected[selected_forecast_model]))]); final_model.fit(panel[features],panel.victim_crashes)
    def final_predict(frame): return np.clip(final_model.predict(frame[features]),0,None)
latest_observed=panel.date.max().to_period("M")
if forecast_start<=latest_observed: raise SystemExit(f"El horizonte debe iniciar después de {latest_observed}")
units=panel[["geo_id","nomgeo","population"]].drop_duplicates("geo_id").sort_values("geo_id"); histories={code:list(group.sort_values("date").victim_crashes.tail(12).astype(float)) for code,group in panel.groupby("geo_id")}; monthly=[]
simulation_periods=pd.period_range(latest_observed+1,forecast_end,freq="M")
for lead,period in enumerate(simulation_periods,start=1):
    month=period.month; frame=units.copy(); frame["month"]=month; frame["year"]=period.year; frame["forecast_month"]=str(period); frame["lead_month_from_observed"]=lead; frame["forecast_role"]="bridge" if period<forecast_start else "target"; frame["lag_1"]=[histories[c][-1] for c in frame.geo_id]; frame["mean_lag_12"]=[np.mean(histories[c][-12:]) for c in frame.geo_id]; frame["spatial_lag_1"]=[np.mean([histories[n][-1] for n in neighbors[c]]) for c in frame.geo_id]; frame["month_sin"]=np.sin(2*np.pi*month/12); frame["month_cos"]=np.cos(2*np.pi*month/12); frame["time_index"]=(period.year-origin_year)*12+month-1; frame["predicted_victim_crashes"]=final_predict(frame); monthly.append(frame[["geo_id","nomgeo","population","year","month","forecast_month","lead_month_from_observed","forecast_role","predicted_victim_crashes"]]);
    for code,value in zip(frame.geo_id,frame.predicted_victim_crashes): histories[code].append(float(value))
simulation=pd.concat(monthly,ignore_index=True); forecast=simulation[simulation.forecast_role.eq("target")].copy(); forecast["horizon_month"]=forecast.groupby("geo_id").cumcount()+1; forecast.to_csv(TABLES/"cdmx_monthly_forecast_future.csv",index=False)
annual=forecast.groupby(["geo_id","nomgeo","population"],as_index=False).agg(predicted_horizon_victim_crashes=("predicted_victim_crashes","sum"),predicted_monthly_average=("predicted_victim_crashes","mean"))

# Capacidad agregada SEDESA 2020 y hospitales con urgencias verificados en la
# cartera oficial SEDESA 2025, georreferenciados con CLUES agosto de 2026.
capacity=pd.read_csv(RAW/"infraestructura_salud_alcaldias_cdmx_2020.csv")
annual=annual.merge(capacity,left_on="nomgeo",right_on="alcaldia",how="left",validate="one_to_one")
if annual.total_unidades.isna().any(): raise SystemExit("No coincidieron las 16 alcaldías con infraestructura")
emergency_path=PROCESSED/"cdmx_emergency_hospitals_sedesa_2025.parquet"
if not emergency_path.exists(): raise SystemExit("Falta ejecutar scripts/prepare_cdmx_geospatial.py")
emergency_counts=pd.read_parquet(emergency_path).groupby("geo_id").size().rename("verified_emergency_hospitals")
annual=annual.merge(emergency_counts,on="geo_id",how="left")
annual["verified_emergency_hospitals"]=annual.verified_emergency_hospitals.fillna(0).astype(int)
annual["predicted_horizon_per_100k"]=annual.predicted_horizon_victim_crashes/annual.population*100000
annual["demand_share"]=annual.predicted_horizon_victim_crashes/annual.predicted_horizon_victim_crashes.sum(); annual["total_unit_share"]=annual.total_unidades/annual.total_unidades.sum(); annual["second_level_share"]=annual.hospitales_segundo_nivel/annual.hospitales_segundo_nivel.sum()
annual["verified_emergency_share"]=annual.verified_emergency_hospitals/annual.verified_emergency_hospitals.sum()
annual["gap_total_units"]=annual.demand_share-annual.total_unit_share; annual["gap_second_level"]=annual.demand_share-annual.second_level_share
annual["gap_verified_emergency"]=annual.demand_share-annual.verified_emergency_share
annual["predicted_crashes_per_total_unit"]=annual.predicted_horizon_victim_crashes/annual.total_unidades; annual["predicted_crashes_per_second_level_hospital"]=annual.predicted_horizon_victim_crashes/annual.hospitales_segundo_nivel.replace(0,np.nan)
annual["predicted_crashes_per_verified_emergency_hospital"]=annual.predicted_horizon_victim_crashes/annual.verified_emergency_hospitals.replace(0,np.nan)
annual["capacity_priority_rank"]=annual.gap_verified_emergency.rank(method="min",ascending=False).astype(int); annual=annual.sort_values("capacity_priority_rank"); annual.to_csv(TABLES/"cdmx_risk_capacity_priority_future.csv",index=False)

# Figuras exclusivas de CDMX.
fig,ax=plt.subplots(figsize=(8,4.8)); m=pd.DataFrame(metrics).sort_values("mae"); ax.barh(m.model,m.mae,color="#247B7B"); ax.set(title=f"Modelos exclusivos de CDMX: MAE {min(test_years)}-{max(test_years)}",xlabel="MAE"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"cdmx_model_mae.png",dpi=220); plt.close(fig)
mapdata=geo.merge(annual,left_on="cvegeo",right_on="geo_id",how="left",validate="one_to_one"); fig,ax=plt.subplots(figsize=(8,8)); mapdata.plot(column="predicted_horizon_victim_crashes",cmap="YlOrRd",edgecolor="white",linewidth=.7,legend=True,ax=ax,legend_kwds={"label":f"Siniestros predichos en {args.horizon_months} meses"});
for _,r in mapdata.iterrows(): p=r.geometry.representative_point(); ax.annotate(r.cve_mun,(p.x,p.y),ha="center",fontsize=7)
ax.set_title(f"Pronóstico por alcaldía\n{horizon_label}"); ax.axis("off"); fig.tight_layout(); fig.savefig(FIGURES/"cdmx_risk_map_future.png",dpi=220,bbox_inches="tight"); plt.close(fig)
rankplot=annual.sort_values("predicted_horizon_victim_crashes"); fig,ax=plt.subplots(figsize=(9,6)); ax.barh(rankplot.nomgeo,rankplot.predicted_horizon_victim_crashes,color="#B33A3A"); ax.set(title=f"Siniestros con víctimas predichos por alcaldía\n{horizon_label}",xlabel=f"Conteo predicho en {args.horizon_months} meses"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"cdmx_forecast_ranking_future.png",dpi=220,bbox_inches="tight"); plt.close(fig)
plot=annual.sort_values("gap_verified_emergency"); fig,ax=plt.subplots(figsize=(9,6)); colors=np.where(plot.gap_verified_emergency>0,"#B33A3A","#3A7D44"); ax.barh(plot.nomgeo,plot.gap_verified_emergency*100,color=colors); ax.axvline(0,color="black",linewidth=.8); ax.set(title="Brecha relativa: riesgo predicho vs hospitales con urgencias",xlabel="Puntos porcentuales (participación en riesgo - participación en hospitales)"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"cdmx_capacity_gap.png",dpi=220,bbox_inches="tight"); plt.close(fig)
joblib.dump(final_model,MODELS/"cdmx_forecast_refit_latest.joblib")
(MODELS/"cdmx_manifest.json").write_text(json.dumps({"scope":"Ciudad de México, 16 alcaldías","features":features,"observed_through":str(latest_observed),"train_years":train_years,"validation_years":cfg["validation_years"],"test_years":test_years,"selected_forecast_model":selected_forecast_model,"selected_hyperparameters":selected_parameters,"model_selection_rule":"menor MAE medio en validación temporal progresiva con el mejor conjunto de hiperparámetros de cada familia; la prueba final 2024-2025 no selecciona el modelo","forecast_start":str(forecast_start),"forecast_end":str(forecast_end),"forecast_horizon_months":args.horizon_months,"recursive_bridge_months":max(0,len(simulation_periods)-args.horizon_months),"capacity_scenario":"34 hospitales con urgencias SEDESA 2025; recursos SINERHIAS al 15 de mayo de 2026 para IMSS-Bienestar","seed":SEED},ensure_ascii=False,indent=2),encoding="utf-8")
logger.info("Análisis CDMX terminado: %d observaciones train, %d test",len(train),len(test))
