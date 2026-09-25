import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Social Media Impact Analyzer", page_icon="📱")

folder = Path(__file__).parent
model = joblib.load(folder / "social_media_model.pkl")

csv = next(folder.glob("*.csv"))
df = pd.read_csv(csv)

st.title("📱 Social Media Impact Analyzer")
st.write("Enter your details to predict your mental health score.")

age = st.number_input("Age", 10, 100, 20)
usage = st.number_input("Average Daily Social Media Usage (Hours)", 0.0, 24.0, 4.0)
sleep = st.number_input("Sleep Hours Per Night", 0.0, 24.0, 7.0)

if st.button("🔮 Predict"):

    data = pd.DataFrame({
        "Avg_Daily_Usage_Hours": [usage],
        "Sleep_Hours_Per_Night": [sleep],
        "Age": [age]
    })

    prediction = float(model.predict(data)[0])

    mean = df["Mental_Health_Score"].mean()
    std = df["Mental_Health_Score"].std()

    st.subheader(f"🔮 Predicted Mental Health Score: {prediction:.2f}")

    if prediction < mean - std:
        st.info("🔵 **Lower Mental Health Score Range**")
        st.write("The predicted score is below the typical range in the dataset.")

    elif prediction > mean + std:
        st.error("🔴 **Higher Mental Health Score Range**")
        st.write("The predicted score is above the typical range in the dataset.")

    else:
        st.warning("🟡 **Moderate Mental Health Score Range**")
        st.write("The predicted score is within the typical range in the dataset.")

    st.caption(
        "Categories are based on this project dataset and are not clinical diagnoses."
    )
