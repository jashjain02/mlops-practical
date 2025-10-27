#!/usr/bin/env python3
"""
Test script to verify Streamlit prediction functionality
"""

import pandas as pd
import mlflow
import mlflow.pyfunc
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("file://" + os.path.join(os.getcwd(), "mlruns"))

def age_bucket_to_years(age_bucket):
    """Convert age bucket to years (midpoint)"""
    age_mapping = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95
    }
    return age_mapping.get(age_bucket, 65)

def create_test_row():
    """Create a test row exactly as in Streamlit app"""
    return {
        # Basic demographics
        "race": "Caucasian", 
        "gender": "Female", 
        "age_years": age_bucket_to_years("[60-70)"),
        "weight": 80.0,
        
        # Hospital stay info
        "time_in_hospital": 4, 
        "num_lab_procedures": 40,
        "num_procedures": 1, 
        "num_medications": 12,
        "number_outpatient": 0, 
        "number_emergency": 0, 
        "number_inpatient": 0,
        "number_diagnoses": 5,
        
        # Admission/discharge info (using IDs instead of descriptions)
        "admission_type_id": 1,  # Emergency
        "discharge_disposition_id": 1,  # Discharged to home
        "admission_source_id": 1,  # Physician Referral
        
        # Diagnosis chapters
        "diag_1_chapter": "endocrine_metabolic", 
        "diag_2_chapter": "circulatory", 
        "diag_3_chapter": "respiratory",
        
        # Lab results
        "max_glu_serum": "None", 
        "A1Cresult": "None",
        
        # Diabetes medications (all possible columns)
        "insulin": "No", 
        "metformin": "No", 
        "glipizide": "No",
        "glimepiride": "No", 
        "glyburide": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride-pioglitazone": "No",
        "glipizide-metformin": "No",
        "glyburide-metformin": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "examide": "No",
        "citoglipton": "No",
        "acetohexamide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        
        # Other features
        "change": "No", 
        "diabetesMed": "No",
        "readmitted": "NO"  # This is a feature, not the target
    }

def main():
    print("🧪 Testing Streamlit prediction functionality...")
    
    try:
        # Load the model
        print("📥 Loading model...")
        model = mlflow.pyfunc.load_model('models:/hospital_readmission/Production')
        print("✅ Model loaded successfully!")
        
        # Create test data
        print("📊 Creating test data...")
        row = create_test_row()
        df_single = pd.DataFrame([row])
        
        print(f"📋 Test data shape: {df_single.shape}")
        print(f"📋 Columns: {len(df_single.columns)}")
        
        # Test prediction
        print("🔮 Making prediction...")
        prob = model.predict(df_single)[0]
        
        print(f"✅ Prediction successful!")
        print(f"📈 30-day Readmission Probability: {prob:.4f} ({prob:.2%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
