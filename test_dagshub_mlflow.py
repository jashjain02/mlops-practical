#!/usr/bin/env python3
"""
Quick test script to verify DagsHub MLflow integration.
This will create a test MLflow run that DagsHub can detect.
"""

import os
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# DagsHub integration
try:
    import dagshub
    dagshub.init(repo_owner=os.getenv("DAGSHUB_USERNAME", "jash.jain029"),
                 repo_name=os.getenv("DAGSHUB_REPO_NAME", "DiabetesMLops"),
                 mlflow=True)
    print("✅ DagsHub initialized successfully")
except Exception as e:
    print(f"❌ Error initializing DagsHub: {e}")
    exit(1)

import mlflow

# Set experiment
mlflow.set_experiment("hospital-readmission")

# Create a test run
print("🔄 Creating test MLflow run...")
with mlflow.start_run(run_name="test_dagshub_integration"):
    # Log a test parameter
    mlflow.log_param("test_parameter", "test_value")
    print("✅ Logged test parameter")
    
    # Log a test metric
    mlflow.log_metric("test_metric", 1.0)
    print("✅ Logged test metric")
    
    # Get run info
    run_id = mlflow.active_run().info.run_id
    print(f"✅ Test run created with ID: {run_id}")
    print(f"📊 View it on DagsHub: https://dagshub.com/jash.jain029/DiabetesMLops/experiments/")

print("\n✅ MLflow integration test completed!")
print("The red X should now turn green on DagsHub. Click 'Check' or refresh the page.")

