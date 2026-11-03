import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
iris = load_iris()

# Input and output data
X = iris.data
y = iris.target

# Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Streamlit application
st.title("Iris Flower Prediction 🌸")

st.write("Enter the flower measurements:")

# User inputs
sl = st.number_input("Sepal Length", min_value=0.0)
sw = st.number_input("Sepal Width", min_value=0.0)
pl = st.number_input("Petal Length", min_value=0.0)
pw = st.number_input("Petal Width", min_value=0.0)

# Prediction
if st.button("Predict"):

    pred = model.predict([[sl, sw, pl, pw]])

    flowers = ["Setosa", "Versicolor", "Virginica"]

    st.success(
        f"Predicted Iris Flower: {flowers[pred[0]]}"
    )