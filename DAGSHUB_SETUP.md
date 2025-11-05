# DagsHub Integration Setup Guide

This guide will help you set up DagsHub for MLflow tracking and model registry in your DiabetesCareMLOps project.

## Prerequisites

1. A DagsHub account (sign up at [dagshub.com](https://dagshub.com))
2. A DagsHub repository created for this project

## Step 1: Create a DagsHub Repository

1. Log in to [dagshub.com](https://dagshub.com)
2. Click "New Repository" 
3. Create a repository (e.g., `DiabetesMLops`)
4. Note your **username** and **repository name** - you'll need these later

**Your repository details:**
- Username: `jash.jain029`
- Repository: `DiabetesMLops`

## Step 2: Generate DagsHub Token

1. Go to your DagsHub repository: https://dagshub.com/jash.jain029/DiabetesMLops/settings/access-tokens
2. Navigate to **Settings** → **Access Tokens**
3. Click **Generate New Token**
4. Give it a name (e.g., "MLOps Pipeline")
5. Select scopes: `repo` and `mlflow`
6. **Copy the token immediately** - you won't be able to see it again!

## Step 3: Configure Local Environment with .env File

**The easiest way is to use the `.env` file:**

1. Copy the example file:
   ```bash
   cp env.example .env
   ```
   
   Or copy `local.env`:
   ```bash
   cp local.env .env
   ```

2. Edit `.env` and add your DagsHub token:
   ```bash
   # Open .env in your editor
   nano .env  # or use your preferred editor
   ```
   
   Update the `DAGSHUB_TOKEN` line:
   ```bash
   DAGSHUB_TOKEN=your-actual-token-here
   ```

3. The `.env` file already has your repository details pre-filled:
   - `DAGSHUB_USERNAME=jash.jain029`
   - `DAGSHUB_REPO_NAME=DiabetesMLops`

**The code will automatically load the `.env` file** - no need to set environment variables manually!

## Step 4: Configure GitHub Actions Secrets

For your CI/CD pipeline to work, you need to add secrets to your GitHub repository:

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add:

   - **Name**: `DAGSHUB_USERNAME`
     **Value**: `jash.jain029`
   
   - **Name**: `DAGSHUB_REPO_NAME`
     **Value**: `DiabetesMLops`
   
   - **Name**: `DAGSHUB_TOKEN`
     **Value**: Your DagsHub access token (from Step 2)

## Step 5: Install Dependencies

Make sure you have the updated requirements installed:

```bash
pip install -r requirements.txt
```

The `dagshub` package should now be installed.

## Step 6: Test the Integration

### Test Locally:

1. **Train a model** (this will log to DagsHub):
   ```bash
   python -m src.train --data data/diabetes.csv --register hospital_readmission
   ```

2. **Check DagsHub**:
   - Go to your DagsHub repository
   - Navigate to the **Experiments** tab
   - You should see your MLflow runs logged there
   - Check the **Models** tab to see registered models

3. **Test Streamlit App**:
   ```bash
   streamlit run app_streamlit.py
   ```
   - The app should be able to load models from DagsHub
   - Use model URI: `models:/hospital_readmission/Production` (or latest version)

### Test GitHub Actions:

1. Push changes to trigger the workflow:
   ```bash
   git add .
   git commit -m "Add DagsHub integration"
   git push
   ```

2. Check GitHub Actions:
   - Go to your GitHub repository
   - Click on the **Actions** tab
   - Watch the workflow run - it should train and evaluate using DagsHub

3. Verify on DagsHub:
   - After the workflow completes, check your DagsHub repository
   - You should see new experiment runs logged from CI/CD

## How It Works

### MLflow Tracking

- **Training** (`src/train.py`): Automatically logs experiments, metrics, parameters, and models to DagsHub
- **Evaluation** (GitHub Actions): Loads models from DagsHub and evaluates them
- **Inference** (`src/inference.py`): Can load models from DagsHub for predictions
- **Streamlit App** (`app_streamlit.py`): Loads models from DagsHub for interactive predictions

### Model Registry

- Models are registered with name: `hospital_readmission`
- You can promote models through stages: `None` → `Staging` → `Production`
- Access models via: `models:/hospital_readmission/Production`

### Code Integration

The code automatically detects DagsHub through:
1. Environment variables (`DAGSHUB_USERNAME`, `DAGSHUB_REPO_NAME`)
2. Or `MLFLOW_TRACKING_URI` if set directly
3. The `dagshub.init()` function configures MLflow to use DagsHub

## Troubleshooting

### Issue: "Could not initialize DagsHub"

**Solution**: Make sure environment variables are set correctly:
```bash
echo $DAGSHUB_USERNAME
echo $DAGSHUB_REPO_NAME
echo $MLFLOW_TRACKING_URI
```

### Issue: "Authentication failed"

**Solution**: 
- Verify your DagsHub token is correct
- Check that the token has `mlflow` scope
- Regenerate the token if needed

### Issue: "Model not found"

**Solution**:
- Check if the model is registered in DagsHub
- Use the correct model URI format: `models:/hospital_readmission/Production`
- Check model stages in DagsHub UI

### Issue: GitHub Actions fails with authentication error

**Solution**:
- Verify all three secrets are set in GitHub: `DAGSHUB_USERNAME`, `DAGSHUB_REPO_NAME`, `DAGSHUB_TOKEN`
- Check that the token has the correct scopes
- Ensure repository name matches exactly (case-sensitive)

## Viewing Results

Once set up, you can:

1. **View Experiments**: Go to DagsHub → Your Repo → **Experiments** tab
   - See all training runs
   - Compare metrics
   - View parameters and artifacts

2. **View Models**: Go to **Models** tab
   - See registered models
   - View model versions
   - Promote models through stages

3. **Download Models**: Use MLflow CLI or Python:
   ```python
   import mlflow
   mlflow.set_tracking_uri("https://dagshub.com/USER/REPO.mlflow")
   model = mlflow.pyfunc.load_model("models:/hospital_readmission/Production")
   ```

## Next Steps

- **Data Versioning**: Consider using DVC with DagsHub for data versioning
- **Experiment Tracking**: Explore DagsHub's experiment comparison features
- **Model Deployment**: Use DagsHub model registry for production deployments
- **Collaboration**: Share your repository with team members for collaboration

## Support

- DagsHub Documentation: https://dagshub.com/docs
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- Issues: Report issues in your GitHub repository



