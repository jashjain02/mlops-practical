#!/usr/bin/env python3
"""
Test script to validate the MLOps workflow components locally.
Run this script to test the pipeline before pushing to GitHub.
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def test_data_validation():
    """Test data validation step."""
    print("\n📊 Testing data validation...")
    
    data_file = "data/diabetes.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check required columns
        required_cols = ['readmitted', 'age', 'gender', 'race']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        print("✅ Data validation passed")
        return True
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing step."""
    print("\n🔄 Testing data preprocessing...")
    
    try:
        from src.data_preprocessing import enrich_and_clean
        
        df = pd.read_csv("data/diabetes.csv")
        print(f"Original data shape: {df.shape}")
        
        df_processed = enrich_and_clean(df)
        print(f"Processed data shape: {df_processed.shape}")
        
        # Save processed data
        df_processed.to_csv("data/processed_diabetes.csv", index=False)
        print("✅ Data preprocessing completed")
        return True
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False

def test_model_training():
    """Test model training step."""
    print("\n🤖 Testing model training...")
    
    cmd = "python -m src.train --data data/diabetes.csv --register hospital_readmission"
    return run_command(cmd, "Model training")

def test_model_evaluation():
    """Test model evaluation step."""
    print("\n📈 Testing model evaluation...")
    
    try:
        import mlflow
        import mlflow.pyfunc
        from sklearn.metrics import roc_auc_score
        from src.utils import map_readmitted
        from mlflow.tracking import MlflowClient
        
        # Load and preprocess test data (same as training)
        from src.data_preprocessing import enrich_and_clean
        
        df = pd.read_csv("data/diabetes.csv")
        df['readmitted_30'] = df['readmitted'].apply(map_readmitted)
        
        # Preprocess the data to match training format
        df_processed = enrich_and_clean(df)
        
        # Try to find the latest model version
        client = MlflowClient()
        try:
            # First try to get the latest version
            latest_version = client.get_latest_versions('hospital_readmission', stages=['None'])[0]
            model_uri = f"models:/hospital_readmission/{latest_version.version}"
            print(f"Using model version: {latest_version.version}")
        except Exception:
            # If no registered model, try to load from local run
            print("No registered model found, trying to load from local runs...")
            # Get the most recent run
            experiment = mlflow.get_experiment_by_name("hospital-readmission")
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
                if not runs.empty:
                    run_id = runs.iloc[0]['run_id']
                    model_uri = f"runs:/{run_id}/model"
                    print(f"Using model from run: {run_id}")
                else:
                    print("❌ No model runs found")
                    return False
            else:
                print("❌ No experiment found")
                return False
        
        # Load model
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Make predictions (use a smaller sample for testing)
        test_sample = df_processed.sample(n=min(1000, len(df_processed)), random_state=42)
        predictions = model.predict(test_sample.drop(columns=['readmitted_30']))
        
        # Calculate metrics
        roc_auc = roc_auc_score(test_sample['readmitted_30'], predictions)
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        if roc_auc < 0.7:
            print("❌ Model performance below threshold")
            return False
        else:
            print("✅ Model performance meets requirements")
            return True
            
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return False

def test_docker_build():
    """Test Docker build."""
    print("\n🐳 Testing Docker build...")
    
    # Check if Docker is available
    try:
        result = subprocess.run("docker --version", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Docker not available, skipping Docker build test")
            return True  # Don't fail the test if Docker isn't available
    except Exception:
        print("⚠️  Docker not available, skipping Docker build test")
        return True
    
    cmd = "docker build -t diabetes-mlops-test ."
    return run_command(cmd, "Docker build")

def test_streamlit_app():
    """Test Streamlit app startup."""
    print("\n🌐 Testing Streamlit app...")
    
    # Test if the app can start (without actually running it)
    try:
        import streamlit as st
        import app_streamlit
        print("✅ Streamlit app imports successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting MLOps workflow tests...")
    print("=" * 50)
    
    tests = [
        ("Data Validation", test_data_validation),
        ("Data Preprocessing", test_preprocessing),
        ("Model Training", test_model_training),
        ("Model Evaluation", test_model_evaluation),
        ("Docker Build", test_docker_build),
        ("Streamlit App", test_streamlit_app),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The workflow should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues before pushing to GitHub.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
