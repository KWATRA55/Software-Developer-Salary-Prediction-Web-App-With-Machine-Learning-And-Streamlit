# developer salary prediction web app

#import libs
from logging import error
import pandas as pd
from scipy.sparse.construct import random
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# reading dataset

df = pd.read_csv("C:\data\developer_data/survey_results_public.csv")

# cleaning the data
df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
df = df.rename({ "ConvertedComp" : "Salary" }, axis= 1)
df = df[df["Salary"].notnull()]
df = df.dropna()

# custom function to clean the countries column and creating a new "Other" column
def shorten_categories(categories, cutoff ):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "Other"
    return categorical_map


country_map = shorten_categories(df.Country.value_counts(), 401)
df["Country"] = df["Country"].map(country_map)
df.Country.value_counts()

# cleaning the salary column of the data and drop the other column
df = df[df["Salary"] <= 250000]
df = df[df["Salary"] > 10000]
df = df[df["Country"] != "Other"]

# Cleaning the years code pro column
def clean_experience(x):
    if x == "More than 50 years":
         return 50
    if x == "Less than 1 year":
        return 0.5
    return float(x)

df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)


# cleaning the education column
def clean_education(x):
    if "Bachelor’s degree" in x:
        return "Bachelor’s degree"
    if "Master’s degree" in x:
        return "Master’s degree"
    if "Professional degree" in x or "Other doctoral" in x:
        return "Post grad"
    return "Less than a Bachelors"

df["EdLevel"] = df["EdLevel"].apply(clean_education)


# emcode the EdLevel so that our model understands it
le_education =  LabelEncoder()
df["EdLevel"] = le_education.fit_transform(df["EdLevel"])

# encode the country 
le_country =  LabelEncoder()
df["Country"] = le_education.fit_transform(df["Country"])

# now for training our data we have to split the data into features and labels
# X WOULD BE THE FEATURES - which is the main data
# Y WOULD BE THE LABELS - which would be predicted
x = df.drop("Salary", axis= 1)
y = df["Salary"]


# now creating our linear regression model
linear_reg = LinearRegression()
linear_reg.fit(x, y.values) 

y_pred = linear_reg.predict(x)

# now calculating the error
error = np.sqrt(mean_squared_error(y, y_pred))




# prediction through RandomForest regressor
random_forest_reg = RandomForestRegressor(random_state= 0)
random_forest_reg.fit(x, y.values)
y_pred = random_forest_reg.predict(x)
error = np.sqrt(mean_squared_error(y, y_pred))



# now we would save our random forest model
import pickle
#creating a dictionary with our model and label encoders
data = {"model" : random_forest_reg, "le_country": le_country, "le_education" : le_education}
with open("databse.pkl", "wb") as file:
    pickle.dump(data, file)




