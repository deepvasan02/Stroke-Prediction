import streamlit
import pickle
import numpy as np
import pandas as pd


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data["model"]
cols_when_model_builds = model.get_booster().feature_names

def show_predict_page():

    streamlit.title("Stroke Prediction")
    streamlit.write("""### We need some information to predict your stroke risk status""") 

    # Categorical inputs
    genders = (
        "Female",
        "Male"
    )

    hypertension_status = (
        "Yes",
        "No"
    )

    heart_disease_status = (
        "Yes",
        "No"
    )

    maritial_status = (
        "No",
        "Yes"
    )

    employment_status = (
        "Govt_job", 
        "Never_worked", 
        "Private", 
        "Self-employed", 
        "children"
    )

    residence_type = (
        "Rural", 
        "Urban"
    )

    smoking_status_grp = (
        "Unknown", 
        "formerly smoked", 
        "never smoked", 
        "smokes"
    )

    # Input attributes
    age = streamlit.number_input("Age")
    gender_st = streamlit.selectbox("Gender", genders)
    hypertension_st = streamlit.selectbox("Suffered from Hypertension", hypertension_status)
    heart_disease_st = streamlit.selectbox("Suffered from Heart disease", heart_disease_status)
    married_st = streamlit.selectbox("Married", maritial_status)
    work_type_st = streamlit.selectbox("Employment", employment_status) 
    residence_st = streamlit.selectbox("Residence Status", residence_type)
    avg_glucose_level  = streamlit.number_input("Average Glucose Level")
    bmi = streamlit.number_input("BMI")
    smoking_st = streamlit.selectbox("Smoking", smoking_status_grp)

    # Label Encoding
    gender = 1 if gender_st == "Male" else 0
    hypertension = 1 if hypertension_st == "Yes" else 0
    heart_disease = 1 if heart_disease_st == "Yes" else 0
    ever_married = 1 if married_st == "Yes" else 0
    work_type = 0 if work_type_st == "Govt_job" else (1 if work_type_st == "Never_worked" else (2 if work_type_st == "Private" else (3 if work_type_st == "Self-employed" else 4)))
    Residence_type =  1 if residence_st == "Urban" else 0
    smoking_status = 0 if smoking_st == "Unknown" else (1 if smoking_st == "formerly smoked" else (2 if smoking_st == "never smoked" else 3))


    ok = streamlit.button("Predict Stroke Possibility")

    if ok:        
        X = [[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]]
        pd_df = pd.DataFrame(X, columns=["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]) 
        pd_df = pd_df[cols_when_model_builds]

        prediction = model.predict(pd_df)
        if(prediction == 1):
            streamlit.subheader("You are vulnerable to stroke.You need to take care of yourself.")
        else: 
            streamlit.subheader("Yay! You are not vulnerable to stroke.")





