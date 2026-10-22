"""Create prospective 2025 municipal and CDMX alcaldía risk rankings."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from siniestros_riesgo.config import FIGURES, MODELS, PROCESSED, TABLES, setup_logging

logger=setup_logging("rank_places")
panel=pd.read_parquet(PROCESSED/"modeling_panel.parquet")
features=["lag_1","mean_lag_12","month_sin","month_cos","time_index","state_code"]
locked=joblib.load(MODELS/"xgboost_poisson.joblib")
model=clone(locked).fit(panel[features],panel.victim_crashes)
units=panel[["geo_id","state_code"]].drop_duplicates("geo_id").sort_values("geo_id")
histories={code:list(group.sort_values("date").victim_crashes.tail(12).astype(float)) for code,group in panel.groupby("geo_id")}
monthly=[]
for month in range(1,13):
    frame=units.copy(); frame["year"]=2025; frame["month"]=month
    frame["lag_1"]=[histories[code][-1] for code in frame.geo_id]
    frame["mean_lag_12"]=[float(np.mean(histories[code][-12:])) for code in frame.geo_id]
    frame["month_sin"]=np.sin(2*np.pi*month/12); frame["month_cos"]=np.cos(2*np.pi*month/12); frame["time_index"]=84+month-1
    frame["predicted_victim_crashes"]=np.clip(model.predict(frame[features]),0,None)
    monthly.append(frame[["geo_id","state_code","year","month","predicted_victim_crashes"]])
    for code,value in zip(frame.geo_id,frame.predicted_victim_crashes): histories[code].append(float(value))
forecast=pd.concat(monthly,ignore_index=True); forecast.to_csv(TABLES/"municipal_month_forecast_2025.csv",index=False)
catalog=pd.read_csv("data/external/inegi_municipal_catalog.csv",dtype={"cvegeo":"string","cve_ent":"string","cve_mun":"string"}); catalog["cvegeo"]=catalog.cvegeo.str.zfill(5); catalog["population"]=pd.to_numeric(catalog.pob_total,errors="coerce")
ranking=forecast.groupby(["geo_id","state_code"],as_index=False).agg(predicted_annual_victim_crashes=("predicted_victim_crashes","sum"),predicted_monthly_average=("predicted_victim_crashes","mean"))
ranking=ranking.merge(catalog[["cvegeo","nomgeo","population"]],left_on="geo_id",right_on="cvegeo",how="left",validate="one_to_one").drop(columns="cvegeo")
ranking["predicted_rate_per_100k_population"]=ranking.predicted_annual_victim_crashes/ranking.population*100000
ranking["national_absolute_rank"]=ranking.predicted_annual_victim_crashes.rank(method="min",ascending=False).astype("Int64")
ranking["national_rate_rank"]=ranking.predicted_rate_per_100k_population.rank(method="min",ascending=False).astype("Int64")
ranking.sort_values("predicted_annual_victim_crashes",ascending=False).to_csv(TABLES/"municipal_risk_ranking_2025.csv",index=False)
cdmx=ranking[ranking.state_code.eq("09")].sort_values("predicted_annual_victim_crashes",ascending=False).copy(); cdmx["cdmx_rank"]=range(1,len(cdmx)+1); cdmx.to_csv(TABLES/"cdmx_alcaldia_risk_ranking_2025.csv",index=False)
for data,name,title in ((ranking.nlargest(20,"predicted_annual_victim_crashes"),"national_top20_predicted_risk.png","Municipios con mayor número predicho de siniestros con víctimas, 2025"),(cdmx,"cdmx_alcaldia_predicted_risk.png","Alcaldías de CDMX: siniestros con víctimas predichos, 2025")):
    plot=data.sort_values("predicted_annual_victim_crashes").copy(); plot["label"]=plot.nomgeo+" ("+plot.geo_id+")"; fig,ax=plt.subplots(figsize=(9,6)); ax.barh(plot.label,plot.predicted_annual_victim_crashes,color="#B33A3A"); ax.set(xlabel="Siniestros con víctimas predichos (suma anual)",title=title); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/name,dpi=220,bbox_inches="tight"); plt.close(fig)
geojson="data/external/cdmx_alcaldias_inegi.geojson"
if __import__("pathlib").Path(geojson).exists():
    import geopandas as gpd
    geo=gpd.read_file(geojson).merge(cdmx[["geo_id","predicted_annual_victim_crashes"]],left_on="cvegeo",right_on="geo_id",how="left",validate="one_to_one")
    fig,ax=plt.subplots(figsize=(8,8)); geo.plot(column="predicted_annual_victim_crashes",cmap="YlOrRd",linewidth=.7,edgecolor="white",legend=True,ax=ax,legend_kwds={"label":"Siniestros con víctimas predichos"});
    for _,row in geo.iterrows():
        point=row.geometry.representative_point(); ax.annotate(row["cve_mun"],(point.x,point.y),ha="center",fontsize=7)
    ax.set_title("Riesgo predicho por alcaldía, CDMX, 2025"); ax.axis("off"); fig.tight_layout(); fig.savefig(FIGURES/"cdmx_alcaldia_risk_map.png",dpi=220,bbox_inches="tight"); plt.close(fig)
joblib.dump(model,MODELS/"xgboost_poisson_refit_2018_2024.joblib")
logger.info("Ranking generado para %d municipios y %d alcaldías CDMX",len(ranking),len(cdmx))
