"""Select models within training years, fit final models, and evaluate frozen holdout."""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from siniestros_riesgo.config import MODELS, PROCESSED, TABLES, PROJECT_CONFIG, load_yaml, setup_logging
from siniestros_riesgo.evaluation import binary_metrics, count_metrics, block_bootstrap_difference

logger = setup_logging("run_modeling")
cfg = load_yaml(PROJECT_CONFIG); seed = cfg["seed"]
data = pd.read_parquet(PROCESSED / "modeling_panel.parquet")
frozen = pd.read_parquet(PROCESSED / "modeling_panel_frozen.parquet")
features = json.loads((PROCESSED / "approved_features.json").read_text())["features"]
numeric = [x for x in features if x != "state_code"]
prep = ColumnTransformer([("num", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]), numeric), ("cat",OneHotEncoder(handle_unknown="ignore"),["state_code"])])
candidates = {
 "random_forest": [RandomForestRegressor(n_estimators=180,min_samples_leaf=m,criterion="poisson",n_jobs=-1,random_state=seed) for m in (3,8)],
 "xgboost_poisson": [XGBRegressor(objective="count:poisson",n_estimators=220,max_depth=d,learning_rate=.05,subsample=.85,colsample_bytree=.85,n_jobs=-1,random_state=seed) for d in (3,5)]}
validation=[]; selected={}
for name, options in candidates.items():
    option_scores=[]
    for i, estimator in enumerate(options):
        scores=[]
        for validation_year in (2021, 2022):
            train=data[data.year < validation_year]; valid=data[data.year == validation_year]
            model=Pipeline([("prep",clone(prep)),("model",clone(estimator))])
            model.fit(train[features],train.victim_crashes)
            score=mean_absolute_error(valid.victim_crashes,np.clip(model.predict(valid[features]),0,None)); scores.append(score)
            validation.append({"model":name,"option":i,"validation_year":validation_year,"mae":score})
        option_scores.append(np.mean(scores))
    selected[name]=options[int(np.argmin(option_scores))]
TABLES.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
pd.DataFrame(validation).to_csv(TABLES / "temporal_validation.csv",index=False)
train=data[data.year.isin(cfg["train_years"])].copy(); test=frozen[frozen.year.isin(cfg["test_years"])].copy()
predictions=pd.DataFrame({"geo_id":test.geo_id,"year":test.year,"month":test.month,"victim_crashes":test.victim_crashes,"actual_binary":test.any_victim_crash})
predictions["baseline"] = test.mean_lag_12.fillna(0).clip(lower=0).to_numpy()
models={}
# Binomial negativa NB2 con alpha=1; mes y entidad como efectos fijos categóricos.
nb_train=train.copy(); nb_test=test.copy()
for column in ("lag_1","mean_lag_12"):
    nb_train[f"log_{column}"]=np.log1p(nb_train[column].fillna(0).clip(lower=0)); nb_test[f"log_{column}"]=np.log1p(nb_test[column].fillna(0).clip(lower=0))
for column in ("log_lag_1","log_mean_lag_12","month_sin","month_cos","time_index"):
    median=nb_train[column].median(); nb_train[column]=nb_train[column].fillna(median); nb_test[column]=nb_test[column].fillna(median)
    mean=nb_train[column].mean(); scale=nb_train[column].std() or 1
    nb_train[f"z_{column}"]=(nb_train[column]-mean)/scale; nb_test[f"z_{column}"]=(nb_test[column]-mean)/scale
nb = smf.glm("victim_crashes ~ z_log_lag_1 + z_log_mean_lag_12 + z_month_sin + z_month_cos + z_time_index + C(state_code)", data=nb_train, family=sm.families.NegativeBinomial(alpha=1.0)).fit(maxiter=100)
predictions["negative_binomial"] = np.clip(nb.predict(nb_test),0,None)
joblib.dump(nb,MODELS/"negative_binomial.joblib")
for name, estimator in selected.items():
    model=Pipeline([("prep",clone(prep)),("model",estimator)]); model.fit(train[features],train.victim_crashes); models[name]=model
    predictions[name]=np.clip(model.predict(test[features]),0,None); joblib.dump(model,MODELS/f"{name}.joblib")
binary_specs={"logistic":LogisticRegression(max_iter=1500,class_weight="balanced",random_state=seed),"rf_binary":RandomForestClassifier(n_estimators=180,min_samples_leaf=5,class_weight="balanced",n_jobs=-1,random_state=seed),"xgb_binary":XGBClassifier(n_estimators=220,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,n_jobs=-1,random_state=seed,eval_metric="logloss")}
binary_rows=[]
for name, estimator in binary_specs.items():
    model=Pipeline([("prep",clone(prep)),("model",estimator)]); model.fit(train[features],train.any_victim_crash)
    probability=model.predict_proba(test[features])[:,1]; predictions[name]=probability
    binary_rows.append({"model":name,**binary_metrics(test.any_victim_crash.to_numpy(),probability)}); joblib.dump(model,MODELS/f"{name}.joblib")
count_rows=[]
for name in ("baseline","negative_binomial","random_forest","xgboost_poisson"):
    count_rows.append({"model":name,**count_metrics(predictions.victim_crashes.to_numpy(),predictions[name].to_numpy())})
pd.DataFrame(count_rows).to_csv(TABLES/"test_count_metrics.csv",index=False)
pd.DataFrame(binary_rows).to_csv(TABLES/"test_binary_metrics.csv",index=False)
predictions.to_parquet(PROCESSED/"test_predictions.parquet",index=False)
boot=[]
for name in ("negative_binomial","random_forest","xgboost_poisson"):
    result=block_bootstrap_difference(predictions,lambda y,p: mean_absolute_error(y,p),name,"baseline",cfg["bootstrap_repetitions"],seed)
    boot.append({"model":name,"metric":"mae_model_minus_baseline",**result})
pd.DataFrame(boot).to_csv(TABLES/"bootstrap_differences.csv",index=False)
(MODELS/"manifest.json").write_text(json.dumps({"features":features,"train_years":cfg["train_years"],"test_years":cfg["test_years"],"seed":seed,"holdout":"frozen_at_2022","internal_validation_years":[2021,2022],"limitations":["No se ejecutó leave-one-state-out exhaustivo por costo de cómputo; se reportará heterogeneidad estatal.","Sin población, red vial ni rezago espacial por falta de fuentes externas."]},ensure_ascii=False,indent=2),encoding="utf-8")
logger.info("Modelado terminado: %d filas train, %d filas test",len(train),len(test))
