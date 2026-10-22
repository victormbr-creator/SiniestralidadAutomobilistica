"""Create model comparisons, heterogeneity tables, and descriptive explanations."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from siniestros_riesgo.config import FIGURES, MODELS, PROCESSED, TABLES, PROJECT_CONFIG, load_yaml, setup_logging

logger=setup_logging("analyze_results"); cfg=load_yaml(PROJECT_CONFIG)
FIGURES.mkdir(parents=True,exist_ok=True); TABLES.mkdir(parents=True,exist_ok=True)
metrics=pd.read_csv(TABLES/"test_count_metrics.csv")
fig,ax=plt.subplots(figsize=(8,4.8)); ordered=metrics.sort_values("mae"); ax.barh(ordered.model,ordered.mae,color="#247B7B"); ax.set(xlabel="MAE en test congelado",title="Error absoluto medio, 2023-2024"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"model_mae_comparison.png",dpi=220); plt.close(fig)
capture=metrics.set_index("model")[["capture_top_5","capture_top_10"]]; ax=capture.plot(kind="bar",figsize=(9,5),color=["#1B4965","#62B6CB"]); ax.set(ylabel="Proporción de eventos capturados",title="Captura de siniestros en unidades priorizadas",xlabel="Modelo"); ax.legend(["Top 5%","Top 10%"]); ax.grid(axis="y",alpha=.25); plt.tight_layout(); plt.savefig(FIGURES/"model_capture_comparison.png",dpi=220); plt.close()
binary=pd.read_csv(TABLES/"test_binary_metrics.csv"); fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(binary.model,binary.pr_auc,color="#7A5195"); ax.set(ylabel="PR-AUC",title="Clasificación binaria, 2023-2024"); ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"binary_pr_auc.png",dpi=220); plt.close(fig)
pred=pd.read_parquet(PROCESSED/"test_predictions.parquet"); panel=pd.read_parquet(PROCESSED/"modeling_panel_frozen.parquet")[["geo_id","state_code"]].drop_duplicates("geo_id"); pred=pred.merge(panel,on="geo_id",how="left",validate="many_to_one")
rows=[]
for state,group in pred.groupby("state_code"):
    for model in ("baseline","negative_binomial","random_forest","xgboost_poisson"):
        rows.append({"state_code":state,"model":model,"n_unit_months":len(group),"mae":mean_absolute_error(group.victim_crashes,group[model])})
pd.DataFrame(rows).to_csv(TABLES/"performance_by_state.csv",index=False)
# Importancia predictiva de árboles; no implica causalidad.
for name in ("random_forest","xgboost_poisson"):
    model=joblib.load(MODELS/f"{name}.joblib"); names=model.named_steps["prep"].get_feature_names_out(); values=model.named_steps["model"].feature_importances_; table=pd.DataFrame({"feature":names,"importance":values}).sort_values("importance",ascending=False); table.to_csv(TABLES/f"{name}_feature_importance.csv",index=False)
    top=table.head(15).sort_values("importance"); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(top.feature,top.importance,color="#D17A22"); ax.set(title=f"Importancia predictiva descriptiva: {name}",xlabel="Importancia"); fig.tight_layout(); fig.savefig(FIGURES/f"{name}_feature_importance.png",dpi=220); plt.close(fig)
# SHAP sobre una muestra reproducible del XGBoost.
import shap
features=["lag_1","mean_lag_12","month_sin","month_cos","time_index","state_code"]
xgb=joblib.load(MODELS/"xgboost_poisson.joblib"); sample=pd.read_parquet(PROCESSED/"modeling_panel_frozen.parquet").query("year >= 2023").sample(n=1000,random_state=cfg["seed"])
matrix=xgb.named_steps["prep"].transform(sample[features]); names=xgb.named_steps["prep"].get_feature_names_out(); explanation=shap.TreeExplainer(xgb.named_steps["model"])(matrix)
pd.DataFrame({"feature":names,"mean_abs_shap":np.abs(explanation.values).mean(axis=0)}).sort_values("mean_abs_shap",ascending=False).to_csv(TABLES/"xgboost_shap_summary.csv",index=False)
shap.summary_plot(explanation.values,matrix,feature_names=names,show=False,max_display=15); plt.title("SHAP descriptivo: XGBoost Poisson"); plt.tight_layout(); plt.savefig(FIGURES/"xgboost_shap_summary.png",dpi=220,bbox_inches="tight"); plt.close()
# Heterogeneidad descriptiva de eventos; el modelo municipal-mes no predice hora.
events=pd.concat([pd.read_parquet(PROCESSED/f"atus_events_{year}.parquet") for year in cfg["years"]],ignore_index=True)
time_table=events.groupby(["night","weekend"],as_index=False).agg(crashes=("event_id","size"),victim_crashes=("victimas","sum"),deaths=("deaths_total","sum"),injuries=("injuries_total","sum")); time_table["victim_share"]=time_table.victim_crashes/time_table.crashes; time_table.to_csv(TABLES/"event_severity_by_time.csv",index=False)
logger.info("Figuras e interpretación descriptiva guardadas")
