
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Social Media Impact Analyzer",
    page_icon="📱",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.sub-title {
    font-size: 17px;
    color: #d0d3d8;
    margin-bottom: 25px;
}

.result-box {
    background-color: #123d29;
    padding: 20px;
    border-radius: 8px;
    margin-top: 15px;
    margin-bottom: 15px;
}

.result-text {
    color: #39ff88;
    font-size: 18px;
    font-weight: 500;
}

.interpretation-box {
    background-color: #19324a;
    padding: 20px;
    border-radius: 8px;
    margin-top: 10px;
}

.interpretation-title {
    color: white;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}

.interpretation-text {
    color: #e5e7eb;
    font-size: 16px;
    line-height: 1.5;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 30px;
    padding: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:
    model = joblib.load("social_media_model.pkl")

except FileNotFoundError:
    st.error("❌ Model file 'social_media_model.pkl' not found.")
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

try:
    df = pd.read_csv("Social Media Impact on Life.csv")

except FileNotFoundError:
    st.error(
        "❌ Dataset file 'Social Media Impact on Life.csv' not found."
    )
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Age",
    "Mental_Health_Score"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("❌ Required columns are missing from the dataset.")
    st.write(missing_columns)
    st.stop()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📱 Social Media Impact Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Enter your details to predict your mental health score.</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# INPUTS
# --------------------------------------------------

age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=20,
    step=1
)

usage = st.number_input(
    "Average Daily Social Media Usage (Hours)",
    min_value=0.0,
    max_value=24.0,
    value=4.0,
    step=0.1
)

sleep = st.number_input(
    "Sleep Hours Per Night",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.1
)

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------

if st.button("🔮 Predict"):

    try:

        # Create input for model
        input_data = pd.DataFrame({
            "Avg_Daily_Usage_Hours": [usage],
            "Sleep_Hours_Per_Night": [sleep],
            "Age": [age]
        })

        # Prediction
        prediction = float(model.predict(input_data)[0])

        # Dataset statistics
        average_score = float(
            df["Mental_Health_Score"].mean()
        )

        standard_deviation = float(
            df["Mental_Health_Score"].std()
        )

        # --------------------------------------------------
        # CLASSIFICATION
        # --------------------------------------------------

        if prediction < average_score - standard_deviation:

            classification = "Lower Mental Health Score Range"
            icon = "🔵"

            explanation = (
                "The predicted score is lower than the typical "
                "range observed in the project dataset."
            )

        elif prediction > average_score + standard_deviation:

            classification = "Higher Mental Health Score Range"
            icon = "🔴"

            explanation = (
                "The predicted score is higher than the typical "
                "range observed in the project dataset."
            )

        else:

            classification = "Moderate Mental Health Score Range"
            icon = "🟡"

            explanation = (
                "The predicted score falls within the average "
                "range observed in the project dataset."
            )

        # --------------------------------------------------
        # PREDICTION RESULT
        # --------------------------------------------------

        st.markdown(
            f'<div class="result-box">'
            f'<div class="result-text">'
            f'Predicted Mental Health Score: {prediction:.2f}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # --------------------------------------------------
        # INTERPRETATION
        # --------------------------------------------------

        interpretation_html = (
            '<div class="interpretation-box">'
            f'<div class="interpretation-title">'
            f'{icon} {classification}'
            f'</div>'
            f'<div class="interpretation-text">'
            f'{explanation}'
            f'</div>'
            '</div>'
        )

        st.markdown(
            interpretation_html,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    '<div class="footer">'
    '📱 Social Media Impact Analyzer<br>'
    'Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
