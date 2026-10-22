"""Fit CDMX-only alcaldía models and compare predicted risk with SEDESA capacity."""

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

from siniestros_riesgo.config import FIGURES, MODELS, PROCESSED, RAW, TABLES, setup_logging
from siniestros_riesgo.evaluation import count_metrics, block_bootstrap_difference

SEED=42
logger=setup_logging("run_cdmx_analysis")
TABLES.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)

# Panel exclusivo de las 16 demarcaciones territoriales.
panel=pd.read_parquet(PROCESSED/"municipality_month_panel.parquet").query("state_code == '09'").copy()
catalog=pd.read_csv("data/external/inegi_municipal_catalog.csv",dtype={"cvegeo":"string"}); catalog["cvegeo"]=catalog.cvegeo.str.zfill(5); catalog["population"]=pd.to_numeric(catalog.pob_total,errors="coerce")
panel=panel.merge(catalog[["cvegeo","nomgeo","population"]],left_on="geo_id",right_on="cvegeo",how="left",validate="many_to_one").drop(columns="cvegeo")
if panel.geo_id.nunique()!=16: raise SystemExit(f"Se esperaban 16 alcaldías, se encontraron {panel.geo_id.nunique()}")
panel["month_sin"]=np.sin(2*np.pi*panel.month/12); panel["month_cos"]=np.cos(2*np.pi*panel.month/12); panel["time_index"]=(panel.year-2018)*12+panel.month-1

# Vecindad Queen oficial de las alcaldías; el rezago usa únicamente el mes anterior.
geo=gpd.read_file("data/external/cdmx_alcaldias_inegi.geojson").sort_values("cvegeo").reset_index(drop=True)
w=Queen.from_dataframe(geo,use_index=False); neighbors={geo.loc[i,"cvegeo"]:[geo.loc[j,"cvegeo"] for j in js] for i,js in w.neighbors.items()}
lookup=panel.set_index(["geo_id","date"])["lag_1"]
def spatial_value(row):
    values=[lookup.get((code,row.date),np.nan) for code in neighbors[row.geo_id]]
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
panel["spatial_lag_1"]=panel.apply(spatial_value,axis=1)

# Congelar todos los rezagos del test en diciembre/promedio de 2022.
frozen=panel.copy(); train_mask=frozen.year.le(2022); test_mask=frozen.year.ge(2023)
dec22=frozen[(frozen.year==2022)&(frozen.month==12)].set_index("geo_id").victim_crashes
mean22=frozen[frozen.year==2022].groupby("geo_id").victim_crashes.mean()
spatial22={code:float(np.mean([dec22[n] for n in neighbors[code]])) for code in dec22.index}
frozen.loc[test_mask,"lag_1"]=frozen.loc[test_mask,"geo_id"].map(dec22); frozen.loc[test_mask,"mean_lag_12"]=frozen.loc[test_mask,"geo_id"].map(mean22); frozen.loc[test_mask,"spatial_lag_1"]=frozen.loc[test_mask,"geo_id"].map(spatial22)
panel.to_parquet(PROCESSED/"cdmx_alcaldia_month_panel.parquet",index=False); frozen.to_parquet(PROCESSED/"cdmx_alcaldia_month_panel_frozen.parquet",index=False)

features=["lag_1","mean_lag_12","spatial_lag_1","month_sin","month_cos","time_index","geo_id"]
numeric=features[:-1]
prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("alcaldia",OneHotEncoder(handle_unknown="ignore"),["geo_id"])])
candidates={
 "random_forest":[RandomForestRegressor(n_estimators=300,min_samples_leaf=m,criterion="poisson",n_jobs=-1,random_state=SEED) for m in (2,5,10)],
 "xgboost_poisson":[XGBRegressor(objective="count:poisson",n_estimators=300,max_depth=d,learning_rate=.04,subsample=.85,colsample_bytree=.85,n_jobs=-1,random_state=SEED) for d in (2,3,4)]}
validation=[]; selected={}
for name,options in candidates.items():
    means=[]
    for option,estimator in enumerate(options):
        scores=[]
        for valid_year in (2021,2022):
            fit=panel[panel.year<valid_year]; valid=panel[panel.year==valid_year]
            model=Pipeline([("prep",clone(prep)),("model",clone(estimator))]); model.fit(fit[features],fit.victim_crashes); score=mean_absolute_error(valid.victim_crashes,np.clip(model.predict(valid[features]),0,None)); scores.append(score); validation.append({"model":name,"option":option,"validation_year":valid_year,"mae":score})
        means.append(np.mean(scores))
    selected[name]=options[int(np.argmin(means))]
pd.DataFrame(validation).to_csv(TABLES/"cdmx_temporal_validation.csv",index=False)

train=panel[panel.year<=2022].copy(); test=frozen[frozen.year>=2023].copy(); predictions=test[["geo_id","nomgeo","year","month","victim_crashes"]].copy(); predictions["baseline"]=test.mean_lag_12.to_numpy()
for name,estimator in selected.items():
    model=Pipeline([("prep",clone(prep)),("model",estimator)]); model.fit(train[features],train.victim_crashes); predictions[name]=np.clip(model.predict(test[features]),0,None); joblib.dump(model,MODELS/f"cdmx_{name}.joblib")
# NB2 con rezagos logarítmicos para evitar extrapolación explosiva.
nb_train=train.copy(); nb_test=test.copy()
for col in ("lag_1","mean_lag_12","spatial_lag_1"):
    nb_train["log_"+col]=np.log1p(nb_train[col].fillna(0).clip(lower=0)); nb_test["log_"+col]=np.log1p(nb_test[col].fillna(0).clip(lower=0))
nb=smf.glm("victim_crashes ~ log_lag_1 + log_mean_lag_12 + log_spatial_lag_1 + month_sin + month_cos + time_index + C(geo_id)",data=nb_train,family=sm.families.NegativeBinomial(alpha=1)).fit(maxiter=100)
predictions["negative_binomial"]=np.clip(np.asarray(nb.predict(nb_test)),0,None); joblib.dump(nb,MODELS/"cdmx_negative_binomial.joblib")
metrics=[]
for name in ("baseline","negative_binomial","random_forest","xgboost_poisson"): metrics.append({"model":name,**count_metrics(predictions.victim_crashes.to_numpy(),predictions[name].to_numpy())})
pd.DataFrame(metrics).to_csv(TABLES/"cdmx_test_metrics.csv",index=False); predictions.to_csv(TABLES/"cdmx_test_predictions.csv",index=False)
boot=[]
for name in ("negative_binomial","random_forest","xgboost_poisson"):
    boot.append({"model":name,"metric":"mae_model_minus_baseline",**block_bootstrap_difference(predictions,lambda y,p:mean_absolute_error(y,p),name,"baseline",1000,SEED)})
pd.DataFrame(boot).to_csv(TABLES/"cdmx_bootstrap_differences.csv",index=False)

# Reajuste del XGBoost seleccionado en 2018-2024 y proyección recursiva 2025.
final_model=Pipeline([("prep",clone(prep)),("model",clone(selected["xgboost_poisson"]))]); final_model.fit(panel[features],panel.victim_crashes)
units=panel[["geo_id","nomgeo","population"]].drop_duplicates("geo_id").sort_values("geo_id"); histories={code:list(group.sort_values("date").victim_crashes.tail(12).astype(float)) for code,group in panel.groupby("geo_id")}; monthly=[]
for month in range(1,13):
    frame=units.copy(); frame["month"]=month; frame["year"]=2025; frame["lag_1"]=[histories[c][-1] for c in frame.geo_id]; frame["mean_lag_12"]=[np.mean(histories[c][-12:]) for c in frame.geo_id]; frame["spatial_lag_1"]=[np.mean([histories[n][-1] for n in neighbors[c]]) for c in frame.geo_id]; frame["month_sin"]=np.sin(2*np.pi*month/12); frame["month_cos"]=np.cos(2*np.pi*month/12); frame["time_index"]=84+month-1; frame["predicted_victim_crashes"]=np.clip(final_model.predict(frame[features]),0,None); monthly.append(frame[["geo_id","nomgeo","population","year","month","predicted_victim_crashes"]]);
    for code,value in zip(frame.geo_id,frame.predicted_victim_crashes): histories[code].append(float(value))
forecast=pd.concat(monthly,ignore_index=True); forecast.to_csv(TABLES/"cdmx_monthly_forecast_2025.csv",index=False)
annual=forecast.groupby(["geo_id","nomgeo","population"],as_index=False).predicted_victim_crashes.sum().rename(columns={"predicted_victim_crashes":"predicted_annual_victim_crashes"})

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
annual["predicted_rate_per_100k"]=annual.predicted_annual_victim_crashes/annual.population*100000
annual["demand_share"]=annual.predicted_annual_victim_crashes/annual.predicted_annual_victim_crashes.sum(); annual["total_unit_share"]=annual.total_unidades/annual.total_unidades.sum(); annual["second_level_share"]=annual.hospitales_segundo_nivel/annual.hospitales_segundo_nivel.sum()
annual["verified_emergency_share"]=annual.verified_emergency_hospitals/annual.verified_emergency_hospitals.sum()
annual["gap_total_units"]=annual.demand_share-annual.total_unit_share; annual["gap_second_level"]=annual.demand_share-annual.second_level_share
annual["gap_verified_emergency"]=annual.demand_share-annual.verified_emergency_share
annual["predicted_crashes_per_total_unit"]=annual.predicted_annual_victim_crashes/annual.total_unidades; annual["predicted_crashes_per_second_level_hospital"]=annual.predicted_annual_victim_crashes/annual.hospitales_segundo_nivel.replace(0,np.nan)
annual["predicted_crashes_per_verified_emergency_hospital"]=annual.predicted_annual_victim_crashes/annual.verified_emergency_hospitals.replace(0,np.nan)
annual["capacity_priority_rank"]=annual.gap_verified_emergency.rank(method="min",ascending=False).astype(int); annual=annual.sort_values("capacity_priority_rank"); annual.to_csv(TABLES/"cdmx_risk_capacity_priority_2025.csv",index=False)

# Figuras exclusivas de CDMX.
fig,ax=plt.subplots(figsize=(8,4.8)); m=pd.DataFrame(metrics).sort_values("mae"); ax.barh(m.model,m.mae,color="#247B7B"); ax.set(title="Modelos exclusivos de CDMX: MAE 2023-2024",xlabel="MAE"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"cdmx_model_mae.png",dpi=220); plt.close(fig)
mapdata=geo.merge(annual,left_on="cvegeo",right_on="geo_id",how="left",validate="one_to_one"); fig,ax=plt.subplots(figsize=(8,8)); mapdata.plot(column="predicted_annual_victim_crashes",cmap="YlOrRd",edgecolor="white",linewidth=.7,legend=True,ax=ax,legend_kwds={"label":"Siniestros con víctimas predichos"});
for _,r in mapdata.iterrows(): p=r.geometry.representative_point(); ax.annotate(r.cve_mun,(p.x,p.y),ha="center",fontsize=7)
ax.set_title("Predicción 2025 por alcaldía"); ax.axis("off"); fig.tight_layout(); fig.savefig(FIGURES/"cdmx_risk_map_2025.png",dpi=220,bbox_inches="tight"); plt.close(fig)
rankplot=annual.sort_values("predicted_annual_victim_crashes"); fig,ax=plt.subplots(figsize=(9,6)); ax.barh(rankplot.nomgeo,rankplot.predicted_annual_victim_crashes,color="#B33A3A"); ax.set(title="Siniestros con víctimas predichos por alcaldía, 2025",xlabel="Conteo anual predicho"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"cdmx_forecast_ranking.png",dpi=220,bbox_inches="tight"); plt.close(fig)
plot=annual.sort_values("gap_verified_emergency"); fig,ax=plt.subplots(figsize=(9,6)); colors=np.where(plot.gap_verified_emergency>0,"#B33A3A","#3A7D44"); ax.barh(plot.nomgeo,plot.gap_verified_emergency*100,color=colors); ax.axvline(0,color="black",linewidth=.8); ax.set(title="Brecha relativa: riesgo predicho vs hospitales con urgencias",xlabel="Puntos porcentuales (participación en riesgo - participación en hospitales)"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"cdmx_capacity_gap.png",dpi=220,bbox_inches="tight"); plt.close(fig)
joblib.dump(final_model,MODELS/"cdmx_xgboost_refit_2018_2024.joblib")
(MODELS/"cdmx_manifest.json").write_text(json.dumps({"scope":"Ciudad de México, 16 alcaldías","features":features,"train_years":[2018,2019,2020,2021,2022],"test_years":[2023,2024],"forecast_year":2025,"capacity_scenario":"34 hospitales con al menos un servicio de urgencias marcado en cartera SEDESA 2025; coordenadas CLUES agosto 2026","seed":SEED},ensure_ascii=False,indent=2),encoding="utf-8")
logger.info("Análisis CDMX terminado: %d observaciones train, %d test",len(train),len(test))
