import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open('Student_lr_Regression_model.pkl', 'rb') as file:
        model, scaler, le = pickle.load(file)
        return model, scaler, le

def preprocessing_input_data(data, scaler, le):
    data['Extracurricular_Activities'] = le.transform([data['Extracurricular_Activities']])
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data, scaler, le)
    return model.predict(processed_data)

def main():
    st.title('Student Performance Prediction')
    st.write('Enter your data to get a prediction for your performance')
    
    hour_studied = st.number_input('Hour Studied', min_value=1, max_value=10, value=5)
    previous_score = st.number_input('Previous Score', min_value=40, max_value=100, value=70)
    extra_carr_activity = st.selectbox('Extra Carricular Activities', ['Yes', 'No'])
    sleeping_hours = st.number_input('Sleeping Hours', min_value=4, max_value=10, value=7)
    sample_paper_solved = st.number_input('Sample Question Paper Practiced', min_value=0, max_value=10, value=5)
    
    if st.button("Predict Your Scores"):
        user_data = {
            'Hours_Studied':hour_studied, 
            'Previous_Scores':previous_score, 
            'Extracurricular_Activities':extra_carr_activity, 
            'Sleep_Hours':sleeping_hours,  
            'Sample_Question_Papers_Practiced':sample_paper_solved
        }
        prediction = predict_data(user_data)
        st.success(f'Your Predicted Score is: {prediction[0][0]}')
    
if __name__ == "__main__":
    main()