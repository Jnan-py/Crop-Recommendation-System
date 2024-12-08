import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('crs.csv')  

data = data.dropna()  
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.title("Crop Recommendation System")

st.sidebar.header("Enter Environmental Conditions")
N = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=100, value=50)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=100, value=50)
K = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=100, value=50)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_data = scaler.transform(input_data)

    probabilities = model.predict_proba(input_data)
    crop_classes = model.classes_

    crop_recommendations = dict(zip(crop_classes, probabilities[0]))
    recommended_crops = sorted(crop_recommendations.items(), key=lambda x: x[1], reverse=True)
    recommended_df = pd.DataFrame(recommended_crops, columns=["Crop", "Recommendation Score"])
    
    return recommended_df

if st.sidebar.button("Get Recommendations"):
    recommended_df = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
    st.write("### Recommended Crops")
    st.dataframe(recommended_df)

st.write("""
This crop recommendation system suggests the best crops to plant based on soil and environmental conditions.
Enter the values in the sidebar, then click "Get Recommendations" to see the suggested crops and their scores.
""")
