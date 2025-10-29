import pandas as pd
import streamlit as st
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

st.set_page_config(page_title="Hospital Readmission (30d)", page_icon="🏥", layout="wide")
st.title("🏥 Hospital Readmission (30-day)")

with st.sidebar:
    st.header("Model")
    model_uri = st.text_input("MLflow model URI", "models:/hospital_readmission/Production")
    if st.button("Load model"):
        try:
            st.session_state.model = mlflow.pyfunc.load_model(model_uri)
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Load failed: {e}")

if "model" not in st.session_state:
    st.session_state.model = None

tab1, tab2 = st.tabs(["Single Entry", "Batch CSV"])

with tab1:
    st.subheader("Single patient features")
    # minimal but schema-true fields (match training columns after enrich_and_clean)
    age_bucket = st.selectbox("age (bucket)", ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"], index=6)
    gender = st.selectbox("gender", ["Female","Male","Unknown/Invalid"])
    race = st.selectbox("race", ["Caucasian","AfricanAmerican","Asian","Hispanic","Other"])
    time_in_hosp = st.number_input("time_in_hospital (days)", 1, 100, 4)
    num_lab = st.number_input("num_lab_procedures", 0, 200, 40)
    num_proc = st.number_input("num_procedures", 0, 10, 1)
    num_meds = st.number_input("num_medications", 0, 100, 12)
    n_out = st.number_input("number_outpatient", 0, 50, 0)
    n_er = st.number_input("number_emergency", 0, 50, 0)
    n_in = st.number_input("number_inpatient", 0, 50, 0)
    n_dx = st.number_input("number_diagnoses", 1, 20, 5)

    adm_type_desc = st.selectbox("admission_type_desc", ["Emergency","Urgent","Elective","Newborn","Not Available","Trauma Center","Not Mapped"])
    disch_desc = st.selectbox("discharge_disposition_desc", ["Discharged to home","Discharged/transferred to SNF","Discharged/transferred to ICF","Left AMA","Expired","Hospice / home","Unknown/Invalid","Not Mapped"])
    adm_src_desc = st.selectbox("admission_source_desc", ["Physician Referral","Clinic Referral","HMO Referral","Transfer from a hospital","Emergency Room","Court/Law Enforcement","Not Available","Transfer from another health care facility","Unknown/Invalid","Not Mapped"])

    # diagnosis chapters
    diag1 = st.selectbox("diag_1_chapter", ["endocrine_metabolic","circulatory","respiratory","digestive","genitourinary","injury","ill_defined","supplemental_v","supplemental_e","other","unknown"])
    diag2 = st.selectbox("diag_2_chapter", ["endocrine_metabolic","circulatory","respiratory","digestive","genitourinary","injury","ill_defined","supplemental_v","supplemental_e","other","unknown"])
    diag3 = st.selectbox("diag_3_chapter", ["endocrine_metabolic","circulatory","respiratory","digestive","genitourinary","injury","ill_defined","supplemental_v","supplemental_e","other","unknown"])

    max_glu = st.selectbox("max_glu_serum", ["None","Norm",">200",">300"])
    a1c = st.selectbox("A1Cresult", ["None","Norm",">7",">8"])
    # diabetes meds columns use values: "No","Steady","Up","Down"
    insulin = st.selectbox("insulin", ["No","Steady","Up","Down"])
    metformin = st.selectbox("metformin", ["No","Steady","Up","Down"])
    glipizide = st.selectbox("glipizide", ["No","Steady","Up","Down"])
    glimepiride = st.selectbox("glimepiride", ["No","Steady","Up","Down"])
    glyburide = st.selectbox("glyburide", ["No","Steady","Up","Down"])
    change = st.selectbox("change", ["No","Ch"])
    diabetesMed = st.selectbox("diabetesMed", ["No","Yes"])

    # Create a complete row with all required columns
    row = {
        # Basic demographics
        "race": race, 
        "gender": gender, 
        "age_years": age_bucket_to_years(age_bucket),  # Calculate age from bucket
        "weight": 80.0,  # Default weight
        
        # Hospital stay info
        "time_in_hospital": time_in_hosp, 
        "num_lab_procedures": num_lab,
        "num_procedures": num_proc, 
        "num_medications": num_meds,
        "number_outpatient": n_out, 
        "number_emergency": n_er, 
        "number_inpatient": n_in,
        "number_diagnoses": n_dx,
        
        # Admission/discharge info (using IDs instead of descriptions)
        "admission_type_id": 1,  # Emergency
        "discharge_disposition_id": 1,  # Discharged to home
        "admission_source_id": 1,  # Physician Referral
        
        # Diagnosis chapters
        "diag_1_chapter": diag1, 
        "diag_2_chapter": diag2, 
        "diag_3_chapter": diag3,
        
        # Lab results
        "max_glu_serum": max_glu, 
        "A1Cresult": a1c,
        
        # Diabetes medications (all possible columns)
        "insulin": insulin, 
        "metformin": metformin, 
        "glipizide": glipizide,
        "glimepiride": glimepiride, 
        "glyburide": glyburide,
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
        "change": change, 
        "diabetesMed": diabetesMed,
        "readmitted": "NO"  # This is a feature, not the target
    }
    df_single = pd.DataFrame([row])

    if st.button("Predict"):
        if st.session_state.model is None:
            st.warning("Load a model from the sidebar first.")
        else:
            # Use all columns as features (readmitted is a feature, not target)
            prob = st.session_state.model.predict(df_single)[0]
            st.metric("30-day Readmission Probability", f"{prob:.2%}")
            st.progress(min(max(float(prob), 0.0), 1.0))

with tab2:
    st.subheader("Batch CSV")
    st.caption("Upload raw UCI-style CSV; the app will preprocess it to match the training format.")
    file = st.file_uploader("CSV", type=["csv"])
    if file and st.session_state.model is not None:
        df = pd.read_csv(file)
        try:
            # Apply the same preprocessing that was used during training
            from src.data_preprocessing import enrich_and_clean
            df_processed = enrich_and_clean(df)
            preds = st.session_state.model.predict(df_processed)
            out = df.copy()  # Keep original data for display
            out["readmission_prob"] = preds
            st.dataframe(out.head(50))
            st.download_button("Download predictions", out.to_csv(index=False).encode(), "readmission_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            st.info("Make sure your CSV has the expected columns from the original UCI dataset.")
    elif file and st.session_state.model is None:
        st.warning("Load a model first.")
