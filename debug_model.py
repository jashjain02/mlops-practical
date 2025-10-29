#!/usr/bin/env python3
"""
Debug script to understand what columns the saved model expects
"""
import pandas as pd
import mlflow
import mlflow.pyfunc
from src.data_preprocessing import enrich_and_clean
from src.utils import map_readmitted

# Set MLflow tracking URI
mlflow.set_tracking_uri("file://" + "/Users/jashjain/Desktop/NMIMS/mlops/DiabetesCareMLOps/mlruns")

# Load raw data and preprocess
print("Loading and preprocessing data...")
df = pd.read_csv('data/diabetes.csv')
df['readmitted_30'] = df['readmitted'].apply(map_readmitted)
df_processed = enrich_and_clean(df)

print(f"Processed data shape: {df_processed.shape}")
print(f"Processed columns: {list(df_processed.columns)}")

# Try to load the latest model
try:
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions('hospital_readmission', stages=['None'])[0]
    model_uri = f"models:/hospital_readmission/{latest_version.version}"
    print(f"Loading model: {model_uri}")
    
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model loaded successfully: {type(model)}")
    
    # Try to inspect the model's preprocessor
    if hasattr(model, 'predict') and hasattr(model.predict, '__self__'):
        print("Model has predict method")
        
        # Try to get the underlying pipeline
        if hasattr(model, 'predict') and hasattr(model.predict, '__self__'):
            # This is a ReadmissionPyfuncModel, get the pipeline
            pipeline = model.predict.__self__.pipeline
            print(f"Pipeline type: {type(pipeline)}")
            
            # Get the preprocessor
            preprocessor = pipeline.named_steps['pre']
            print(f"Preprocessor type: {type(preprocessor)}")
            
            # Try to get the column names the preprocessor expects
            if hasattr(preprocessor, 'transformers_'):
                print("Preprocessor transformers:")
                for name, transformer, columns in preprocessor.transformers_:
                    print(f"  {name}: {columns}")
            
            # Try to get feature names
            if hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    print(f"Expected feature names: {feature_names}")
                except Exception as e:
                    print(f"Could not get feature names: {e}")
    
    # Try to make a prediction on a small sample
    print("\nTrying to make prediction on small sample...")
    sample = df_processed.head(1).drop(columns=['readmitted_30'])
    print(f"Sample columns: {list(sample.columns)}")
    
    try:
        prediction = model.predict(sample)
        print(f"Prediction successful: {prediction}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        print(f"Error type: {type(e)}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Error type: {type(e)}")
