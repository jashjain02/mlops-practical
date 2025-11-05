import os
import mlflow
import pandas as pd

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
except Exception as e:
    print(f"Warning: Could not initialize DagsHub: {e}")
    print("Make sure DAGSHUB_USERNAME and DAGSHUB_REPO_NAME are set in .env file, or configure MLFLOW_TRACKING_URI manually")

def load_model(model_uri: str):
    """
    model_uri examples:
      - "runs:/<run_id>/model"
      - "models:/hospital_readmission/Production"
      - local path like "artifacts/model"
    """
    return mlflow.pyfunc.load_model(model_uri)

def predict_df(model, df: pd.DataFrame):
    preds = model.predict(df)  # this will return probabilities if we save it that way
    return preds
