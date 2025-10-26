# 🚀 End-to-End MLOps GitHub Workflow

This document explains how to set up and use the automated MLOps pipeline that triggers when new data is detected.

## 📋 Overview

The GitHub Actions workflow automatically:
1. **Detects changes** in the `data/` folder
2. **Validates data** quality and structure
3. **Preprocesses data** using the preprocessing pipeline
4. **Trains the model** with MLflow tracking
5. **Evaluates model** performance with quality gates
6. **Builds Docker image** with the trained model
7. **Deploys to staging** environment
8. **Sends notifications** on success/failure

## 🔧 Setup Instructions

### 1. Repository Setup

Ensure your repository has the following structure:
```
DiabetesCareMLOps/
├── .github/workflows/
│   └── mlops-pipeline.yml
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── data/
│   ├── diabetes.csv
│   └── IDS_mapping.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── app_streamlit.py
```

### 2. GitHub Secrets (Optional)

For advanced MLflow tracking, set these secrets in your repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add the following secrets:

```
MLFLOW_TRACKING_URI=https://your-mlflow-server.com
MLFLOW_REGISTRY_URI=https://your-mlflow-server.com
```

*Note: If not set, the workflow will use local file-based MLflow tracking.*

### 3. Environment Setup

The workflow uses the following environments:
- **staging**: For staging deployments
- **production**: For production deployments (optional)

To set up environments:
1. Go to **Settings** → **Environments**
2. Create `staging` environment
3. Add any required secrets or protection rules

## 🎯 Workflow Triggers

### Automatic Triggers

The workflow automatically runs when:
- **Data changes**: Any file in `data/` folder is modified
- **Code changes**: Files in `src/`, `requirements.txt`, or `Dockerfile` are modified
- **Manual trigger**: Use "Run workflow" button with optional force retrain

### Manual Triggers

You can manually trigger the workflow:
1. Go to **Actions** tab
2. Select **MLOps Pipeline - Auto Training & Deployment**
3. Click **Run workflow**
4. Choose branch and optional parameters

## 📊 Workflow Steps

### 1. Change Detection
- Detects if data or source code has changed
- Skips training if no relevant changes detected
- Supports force retrain option

### 2. Data Validation
- Validates data file exists and is readable
- Checks for required columns
- Ensures data quality standards

### 3. Data Preprocessing
- Runs the preprocessing pipeline
- Creates processed dataset
- Handles missing values and feature engineering

### 4. Model Training
- Trains XGBoost model with MLflow tracking
- Logs parameters, metrics, and artifacts
- Registers model in MLflow Model Registry

### 5. Model Evaluation
- Evaluates model performance
- Checks quality gates (ROC-AUC > 0.7)
- Fails pipeline if performance is insufficient

### 6. Docker Build
- Builds multi-stage Docker image
- Pushes to GitHub Container Registry
- Supports multi-platform builds (AMD64, ARM64)

### 7. Deployment
- Deploys to staging environment
- Can be extended for production deployment
- Includes health checks and monitoring

## 🐳 Docker Setup

### Local Development

Run the application locally with Docker:

```bash
# Build and run with docker-compose
docker-compose up --build

# Access the application
# Streamlit: http://localhost:8501
# MLflow UI: http://localhost:5000
```

### Production Deployment

The workflow creates a production-ready Docker image:

```bash
# Pull the latest image
docker pull ghcr.io/your-username/your-repo/diabetes-mlops:latest

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/mlruns:/app/mlruns \
  ghcr.io/your-username/your-repo/diabetes-mlops:latest
```

## 📈 Monitoring and Observability

### MLflow Tracking
- **Experiments**: Track all training runs
- **Models**: Version control and staging
- **Artifacts**: Model files and visualizations
- **Metrics**: ROC-AUC, PR-AUC, and custom metrics

### GitHub Actions
- **Workflow runs**: View execution history
- **Artifacts**: Download model files and logs
- **Notifications**: Email/Slack notifications on failure

### Docker Registry
- **Images**: Versioned Docker images
- **Security**: Vulnerability scanning
- **Usage**: Pull and deployment metrics

## 🔧 Customization

### Adding New Data Sources

To add new data sources:

1. **Update workflow triggers**:
```yaml
on:
  push:
    paths:
      - 'data/**'
      - 'new_data_folder/**'  # Add new path
```

2. **Modify data validation**:
```python
# Add validation for new data format
required_cols = ['readmitted', 'age', 'gender', 'race', 'new_column']
```

### Custom Quality Gates

Modify the evaluation step to add custom quality gates:

```python
# Add custom metrics
if roc_auc < 0.8:  # Stricter threshold
    print('❌ Model performance below threshold')
    exit(1)
```

### Deployment Targets

Extend the deployment step for different targets:

```yaml
# Add Kubernetes deployment
- name: Deploy to Kubernetes
  run: |
    kubectl apply -f k8s/deployment.yaml
    kubectl set image deployment/diabetes-mlops \
      diabetes-mlops=ghcr.io/${{ github.repository }}/diabetes-mlops:${{ github.sha }}
```

## 🚨 Troubleshooting

### Common Issues

1. **Workflow not triggering**:
   - Check file paths in workflow triggers
   - Ensure files are committed to the repository

2. **Model training fails**:
   - Check data format and required columns
   - Verify MLflow configuration

3. **Docker build fails**:
   - Check Dockerfile syntax
   - Verify all dependencies in requirements.txt

4. **Deployment fails**:
   - Check environment secrets
   - Verify deployment permissions

### Debug Steps

1. **Check workflow logs**:
   - Go to Actions → Select workflow run
   - Click on failed step to see logs

2. **Test locally**:
   ```bash
   # Test data preprocessing
   python -c "from src.data_preprocessing import enrich_and_clean; print('OK')"
   
   # Test model training
   python -m src.train --data data/diabetes.csv
   
   # Test Docker build
   docker build -t diabetes-mlops .
   ```

3. **Validate MLflow**:
   ```bash
   # Check MLflow tracking
   mlflow ui
   # Visit http://localhost:5000
   ```

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

To contribute to the workflow:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the workflow
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
