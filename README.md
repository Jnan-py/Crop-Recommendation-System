# Crop Recommendation System

This Streamlit application recommends the most suitable crops to plant based on environmental and soil conditions. The recommendations are generated using a trained Random Forest Classifier model.

## Features

- **Input Environmental Conditions**: Users can input values for nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH level, and rainfall using a sidebar.
- **Crop Recommendations**: Based on the provided inputs, the app will recommend the best crops to plant, along with their recommendation scores.
- **Machine Learning Model**: The recommendations are generated using a Random Forest Classifier trained on environmental data.

## Requirements

To run the application, you will need the following Python libraries:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`

You can install them using pip:

```bash
pip install streamlit pandas numpy scikit-learn
```

## How to Run

1. Clone the repository or download the Python script.
2. Make sure you have the required libraries installed (as mentioned above).
3. Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

4. Open the app in your browser and input environmental conditions using the sidebar to get crop recommendations.

## Data

The app uses a CSV file (`crs.csv`) containing data on environmental conditions and crop labels. The CSV file should have the following columns:

- `N`: Nitrogen content
- `P`: Phosphorus content
- `K`: Potassium content
- `temperature`: Temperature in °C
- `humidity`: Humidity in percentage
- `ph`: pH level
- `rainfall`: Rainfall in mm
- `label`: Crop label (target variable)

## How It Works

1. The app reads the data from `crs.csv` and preprocesses it by dropping missing values.
2. The input features are then scaled using `StandardScaler`.
3. A Random Forest Classifier model is trained on the data and used to predict the probability of each crop based on the input values.
4. The app outputs a list of recommended crops with their respective recommendation scores.

## Example Usage

1. Enter the required values for nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall in the sidebar.
2. Click on the "Get Recommendations" button.
3. View the recommended crops and their scores in the main section of the app.
