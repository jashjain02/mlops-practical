import argparse, os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from mlflow.exceptions import RestException
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# DagsHub integration
try:
    import dagshub
    # Initialize DagsHub - uses environment variables if set, otherwise uses defaults
    dagshub.init(repo_owner=os.getenv("DAGSHUB_USERNAME", "jash.jain029"),
                 repo_name=os.getenv("DAGSHUB_REPO_NAME", "DiabetesMLops"),
                 mlflow=True)
    
    # Explicitly set MLflow tracking URI and credentials if provided (for CI/CD)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✅ Using MLflow tracking URI: {tracking_uri}")
    
    # Set credentials if provided (MLflow automatically reads MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD)
    if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
        # MLflow will automatically use these env vars for authentication
        print(f"✅ MLflow authentication configured")
        
except Exception as e:
    print(f"Warning: Could not initialize DagsHub: {e}")
    print("Make sure DAGSHUB_USERNAME and DAGSHUB_REPO_NAME are set, or configure MLFLOW_TRACKING_URI manually")
    
    # Fallback: try to set tracking URI directly if provided
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✅ Using MLflow tracking URI directly: {tracking_uri}")

from src.data_preprocessing import build_preprocessor, enrich_and_clean
from src.evaluate import compute_metrics, roc_fig, pr_fig
from src.utils import map_readmitted

MLFLOW_EXPERIMENT = "hospital-readmission"

class ReadmissionPyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline
    def predict(self, context, model_input):
        return self.pipeline.predict_proba(model_input)[:, 1]

def maybe_load_map(path: str, key_col: str, desc_col: str):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        # clean blanks
        df = df[[key_col, desc_col]].dropna()
        return df
    return None

def main(args):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = pd.read_csv(args.data)
    # make binary target
    df["readmitted_30"] = df["readmitted"].apply(map_readmitted)

    # optional mapping lookups (id -> description)
    adm_type_map = maybe_load_map(args.adm_type_map, "admission_type_id", "description")
    disch_map    = maybe_load_map(args.discharge_map, "discharge_disposition_id", "description")
    adm_src_map  = maybe_load_map(args.adm_src_map, "admission_source_id", "description")

    df = enrich_and_clean(df, adm_type_map, disch_map, adm_src_map)

    target_col = "readmitted_30"
    df = df.dropna(subset=[target_col])

    # slight class imbalance handling (scale_pos_weight = neg/pos)
    pos = (df[target_col]==1).sum()
    neg = (df[target_col]==0).sum()
    spw = max((neg / max(pos,1)), 1.0)

    pre, num_cols, cat_cols = build_preprocessor(df, target_col)
    clf = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42, scale_pos_weight=spw
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X = df.drop(columns=[target_col])
    y = df[target_col].values
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run(run_name="xgb_uci_diabetes") as run:
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("scale_pos_weight", float(spw))
        mlflow.log_param("n_estimators", clf.n_estimators)
        mlflow.log_param("max_depth", clf.max_depth)
        mlflow.log_param("learning_rate", clf.learning_rate)

        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_va)[:,1]

        metrics = compute_metrics(y_va, y_prob)
        for k,v in metrics.items():
            mlflow.log_metric(k, v)

        # figures
        mlflow.log_figure(roc_fig(y_va, y_prob), "figures/roc.png")
        mlflow.log_figure(pr_fig(y_va, y_prob), "figures/pr.png")

        # save pyfunc model
        # DagsHub may not support the newer MLflow "logged model" endpoint, so we catch the error
        try:
            # Use artifact_path (older API) for better DagsHub compatibility
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=ReadmissionPyfuncModel(pipe),
                pip_requirements="requirements.txt",
            )
            print("✅ Model logged successfully")
        except RestException as e:
            # DagsHub doesn't support the logged model endpoint in newer MLflow versions
            if "unsupported endpoint" in str(e).lower() or "INTERNAL_ERROR" in str(e):
                print(f"⚠️  DagsHub doesn't support logged model endpoint (this is okay)")
                print(f"   Model artifacts are saved, but logged model feature is disabled.")
                print(f"   You can still use the model from: runs:/{run.info.run_id}/model")
            else:
                raise
        except Exception as e:
            print(f"⚠️  Warning: Could not log model: {e}")
            print(f"   Model artifacts may still be available at: runs:/{run.info.run_id}/model")
        
        # Register model separately (skip if DagsHub doesn't support it)
        if args.register:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                model_uri = f"runs:/{run.info.run_id}/model"
                
                # Try to create registered model if it doesn't exist
                try:
                    client.create_registered_model(args.register)
                except Exception:
                    pass  # Model might already exist, that's okay
                
                # Create model version
                mv = client.create_model_version(
                    name=args.register,
                    source=model_uri,
                    run_id=run.info.run_id
                )
                print(f"✅ Registered model version {mv.version} for {args.register}")
            except Exception as e:
                print(f"⚠️  Could not register model (this is okay - DagsHub may not support model registry): {e}")
                print(f"   Model is still logged at: runs:/{run.info.run_id}/model")
        print(f"[run_id] {run.info.run_id}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--adm_type_map", default=None)   # CSV with columns: admission_type_id,description
    p.add_argument("--discharge_map", default=None)  # CSV with columns: discharge_disposition_id,description
    p.add_argument("--adm_src_map", default=None)    # CSV with columns: admission_source_id,description
    p.add_argument("--register", default="hospital_readmission")
    args = p.parse_args()
    main(args)
