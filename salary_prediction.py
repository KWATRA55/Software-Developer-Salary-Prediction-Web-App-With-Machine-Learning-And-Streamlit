# CREATING A STREAMLIT APP FOR PREDICTION

# import modules
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder


# now opening the created model (for prediction)
def load_model():
    with open("database.pkl", "rb") as file:
        data = pickle.load(file)
    return data

data = load_model()

random_forest_reg_model = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# THIS IS OUR STREAMLIT APPLICATION
def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write(""" ### We Need Some Information To Predict The Salary """)


    # creating countries and education columns
    countries = ( 'United States' ,'United Kingdom' ,'Spain', 'Netherlands', 'Germany', 'Canada' ,'Italy', 'Brazil', 'France' ,'Sweden' ,'India', 'Poland', 'Australia', 'Russian Federation' )

    education = ( 'Bachelor’s degree' ,'Master’s degree' ,'Less than a Bachelors', 'Post grad' )


    # creating a selectbox
    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 50, 3)

    # creating a buttton for prediction
    ok = st.button("Calculate Salary")

    if ok:     # if ok is pressed or true
        x = np.array([[ country, education, experience]])
        x[:, 0] = le_country.fit_transform(x[:, 0])
        x[:, 1] = le_education.fit_transform (x[:, 1])
        x = x.astype(float)


        # now predicting
        salary = random_forest_reg_model.predict(x)

        st.subheader(f"The Estimated Salary Is ${salary[0]:.2f}")








