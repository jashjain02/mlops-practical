# Why DagsHub Artifact Downloads Fail

## The Problem

When you try to download model artifacts from DagsHub, you get errors like:
```
MlflowException: Failed to download artifacts from path 'model'
```

## Root Cause

**DagsHub's MLflow tracking and artifact storage work differently:**

1. ✅ **MLflow Tracking Works**: Metrics, parameters, and metadata are successfully logged
2. ❌ **Artifact Storage Doesn't Work**: Model files (pickle files, model artifacts) are NOT actually saved

### Why This Happens

When `mlflow.pyfunc.log_model()` is called, it tries to save the model artifacts. However:

1. **DagsHub's MLflow tracking** only stores metadata (metrics, params, tags)
2. **Artifact storage** requires a separate backend:
   - DVC (Data Version Control) - DagsHub's preferred method
   - S3/GCS - Cloud storage backends
   - Local filesystem - Not available in CI/CD

3. **The RestException you see** means the model logging API call fails, but we catch it and continue, so the model never actually gets saved.

## What's Actually Saved

✅ **These are saved to DagsHub:**
- Metrics (ROC-AUC, PR-AUC)
- Parameters (model hyperparameters)
- Tags and metadata
- Figures (ROC curve, PR curve)
- Run information

❌ **These are NOT saved:**
- Model pickle files
- Model artifacts
- The actual trained model

## Solutions

### Option 1: Use DVC for Artifact Storage (Recommended)

DagsHub is built on DVC (Data Version Control). You need to set up DVC to store artifacts:

1. **Install DVC with DagsHub support:**
   ```bash
   pip install dvc dvc-s3  # or dvc-gs for GCS
   ```

2. **Initialize DVC in your repo:**
   ```bash
   dvc init
   ```

3. **Configure DVC to use DagsHub storage:**
   ```bash
   dvc remote add -d origin https://dagshub.com/jash.jain029/DiabetesMLops.dvc
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user jash.jain029
   dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
   ```

4. **Configure MLflow to use DVC:**
   ```python
   import mlflow
   mlflow.set_tracking_uri("https://dagshub.com/jash.jain029/DiabetesMLops.mlflow")
   # Set artifact URI to DVC
   mlflow.set_experiment("hospital-readmission")
   ```

### Option 2: Use S3/GCS Backend

Configure MLflow to use cloud storage for artifacts:

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("https://dagshub.com/jash.jain029/DiabetesMLops.mlflow")

# Set artifact URI to S3
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://s3.amazonaws.com'
mlflow.set_experiment("hospital-readmission")

# When logging model, artifacts will go to S3
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=model,
    pip_requirements="requirements.txt"
)
```

### Option 3: Save Model Locally and Upload Separately

Save models as GitHub Actions artifacts:

```python
# In training script
import joblib
joblib.dump(model, "model.pkl")
mlflow.log_artifact("model.pkl", "models")
```

Then in GitHub Actions:
```yaml
- name: Upload model artifact
  uses: actions/upload-artifact@v4
  with:
    name: trained-model
    path: model.pkl
```

### Option 4: Use DagsHub's Native Storage (Future)

DagsHub is working on native artifact storage support. For now, use DVC or cloud storage.

## Current Workaround

The workflow now:
1. ✅ Logs metrics and parameters (works)
2. ✅ Completes training successfully
3. ⚠️ Skips evaluation if artifacts can't be downloaded (graceful degradation)

**You can still:**
- View all experiments on DagsHub
- Compare model metrics
- See parameters and figures
- Track experiment history

**You cannot:**
- Download/load the trained model from DagsHub
- Use the model registry for deployments

## Verification

To check if your model was actually saved, look at the training logs:

- ✅ **"Model logged successfully"** + **"Verified: Model artifacts saved"** = Model is saved
- ⚠️ **"DagsHub doesn't support logged model endpoint"** = Model was NOT saved
- ❌ **"Model artifacts were NOT saved"** = Model was NOT saved

## Next Steps

1. **For now**: Use the current setup - metrics tracking works great
2. **For production**: Set up DVC or S3 backend for artifact storage
3. **For evaluation**: Models can be saved locally in CI/CD and uploaded as GitHub Actions artifacts

## References

- [DagsHub DVC Setup](https://dagshub.com/docs/integration_guide/dvc/)
- [MLflow Artifact Storage](https://mlflow.org/docs/latest/tracking.html#artifact-storage)
- [DagsHub MLflow Integration](https://dagshub.com/docs/integration_guide/mlflow/)

