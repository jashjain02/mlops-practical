# CI/CD Integration with DagsHub

This document explains how the GitHub Actions workflow is integrated with DagsHub for MLflow tracking.

## Overview

The `.github/workflows/mlops-pipeline.yml` workflow is now fully integrated with DagsHub. All MLflow experiments, metrics, and models are automatically logged to your DagsHub repository.

## Required GitHub Secrets

Before the workflow can run, you need to add these secrets to your GitHub repository:

### Step 1: Go to GitHub Repository Settings

1. Navigate to your GitHub repository
2. Go to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**

### Step 2: Add the Following Secrets

Add these three secrets:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `DAGSHUB_USERNAME` | `jash.jain029` | Your DagsHub username |
| `DAGSHUB_REPO_NAME` | `DiabetesMLops` | Your DagsHub repository name |
| `DAGSHUB_TOKEN` | `your-token-here` | Your DagsHub access token (with `repo` and `mlflow` scopes) |

### How to Get Your DagsHub Token

1. Go to: https://dagshub.com/jash.jain029/DiabetesMLops/settings/access-tokens
2. Click **Generate New Token**
3. Name it: "GitHub Actions CI/CD"
4. Select scopes: `repo` and `mlflow`
5. Copy the token and paste it into the GitHub secret

## Workflow Features

### 1. Automatic DagsHub Authentication

The workflow automatically:
- Sets up DagsHub MLflow tracking URI
- Configures authentication using GitHub secrets
- Initializes MLflow to log to DagsHub

### 2. Training Job

- **Location**: `train-model` job
- **What it does**:
  - Trains the model using your training script
  - Logs all experiments, metrics, parameters to DagsHub
  - Attempts to register models (gracefully handles if registry not available)
  - Captures run ID for use in evaluation

### 3. Evaluation Job

- **Location**: `evaluate-model` job  
- **What it does**:
  - Loads the trained model from DagsHub
  - Tries to load from model registry first
  - Falls back to loading from run ID if registry unavailable
  - Evaluates model performance
  - Validates performance thresholds

### 4. Notification Job

- **Location**: `notify` job
- **What it does**:
  - Provides links to view results on DagsHub
  - Shows experiment and run URLs
  - Reports success/failure status

## Viewing Results

After the workflow runs, you'll see:

1. **Experiments Tab**: https://dagshub.com/jash.jain029/DiabetesMLops/experiments
   - View all training runs
   - Compare metrics
   - See parameters and artifacts

2. **MLflow UI**: https://dagshub.com/jash.jain029/DiabetesMLops.mlflow
   - Full MLflow tracking UI
   - Detailed run information
   - Model artifacts

3. **Workflow Logs**: Check the GitHub Actions run for direct links to your runs

## Workflow Triggers

The workflow runs automatically when:
- **Push events**: Changes to `data/`, `src/`, or `requirements.txt`
- **Manual trigger**: Use "Run workflow" button in GitHub Actions

## Troubleshooting

### Error: "DAGSHUB_USERNAME and DAGSHUB_TOKEN secrets must be set"

**Solution**: Make sure all three secrets are added in GitHub repository settings.

### Error: "Authentication failed"

**Solution**: 
- Verify your DagsHub token is correct
- Ensure token has `mlflow` scope
- Check that token hasn't expired

### Error: "Could not load from registry"

**Solution**: This is normal! The workflow will automatically fall back to loading from run ID. Model registry may not be fully supported by DagsHub yet.

### Model not appearing on DagsHub

**Solution**:
- Check that the workflow completed successfully
- Verify DagsHub credentials are correct
- Check workflow logs for any MLflow errors
- Ensure the experiment name matches: `hospital-readmission`

## Testing the Integration

1. **Push changes** to trigger the workflow:
   ```bash
   git add .
   git commit -m "Test DagsHub CI/CD integration"
   git push
   ```

2. **Monitor the workflow**:
   - Go to GitHub → Actions tab
   - Watch the workflow run
   - Check logs for DagsHub connection status

3. **Verify on DagsHub**:
   - After completion, check your DagsHub repository
   - You should see new experiment runs
   - Metrics and parameters should be logged

## Workflow Outputs

After successful completion, the workflow provides:

- **Run ID**: Unique identifier for the MLflow run
- **Model Version**: Version number (if registry available)
- **Direct Links**: URLs to view results on DagsHub

## Next Steps

- **Monitor Experiments**: Regularly check DagsHub for new training runs
- **Compare Models**: Use DagsHub's experiment comparison features
- **Model Registry**: Once models are registered, promote them through stages
- **Deployment**: Use registered models for production deployments

## Support

- **DagsHub Docs**: https://dagshub.com/docs
- **MLflow Docs**: https://mlflow.org/docs/latest/index.html
- **GitHub Actions**: Check workflow logs for detailed error messages

