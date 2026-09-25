%%writefile "Social Media Impact on Life app.py"

import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Social Media Impact Analyzer",
    page_icon="📱",
    layout="centered"
)


# ==================================================
# FILE LOCATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent


# ==================================================
# CUSTOM CSS
# ==================================================

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

.footer {
    text-align: center;
    color: #777;
    margin-top: 30px;
    padding: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL
# ==================================================

model_path = BASE_DIR / "social_media_model.pkl"

try:
    model = joblib.load(model_path)

except FileNotFoundError:
    st.error("❌ social_media_model.pkl was not found.")
    st.info(f"Expected location: {model_path}")
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()


# ==================================================
# FIND DATASET
# ==================================================

# Look for every CSV in the same folder as the app
csv_files = list(BASE_DIR.glob("*.csv"))


if not csv_files:

    st.error("❌ No CSV dataset was found.")

    st.info(
        "Put your Social Media Impact CSV file in the same "
        "folder as 'Social Media Impact on Life app.py'."
    )

    st.info(f"App folder: {BASE_DIR}")

    st.stop()


# ==================================================
# FIND THE CORRECT CSV
# ==================================================

required_columns = {
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Age",
    "Mental_Health_Score"
}

df = None
dataset_path = None

for csv_file in csv_files:

    try:

        temp_df = pd.read_csv(csv_file)

        if required_columns.issubset(temp_df.columns):

            df = temp_df
            dataset_path = csv_file
            break

    except Exception:
        continue


# ==================================================
# DATASET NOT FOUND
# ==================================================

if df is None:

    st.error(
        "❌ The CSV file was found, but it does not contain "
        "the required columns."
    )

    st.write("Required columns:")

    st.write(list(required_columns))

    st.write("CSV files found:")

    for file in csv_files:
        st.write(file.name)

    st.stop()


# ==================================================
# TITLE
# ==================================================

st.markdown(
    '<div class="main-title">'
    '📱 Social Media Impact Analyzer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Enter your details to predict your mental health score.'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# INPUTS
# ==================================================

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


# ==================================================
# PREDICTION
# ==================================================

if st.button("🔮 Predict"):

    try:

        input_data = pd.DataFrame({
            "Avg_Daily_Usage_Hours": [usage],
            "Sleep_Hours_Per_Night": [sleep],
            "Age": [age]
        })

        prediction = float(
            model.predict(input_data)[0]
        )


        # ----------------------------------------------
        # DATASET STATISTICS
        # ----------------------------------------------

        average_score = float(
            df["Mental_Health_Score"].mean()
        )

        standard_deviation = float(
            df["Mental_Health_Score"].std()
        )


        # ----------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------

        if prediction < average_score - standard_deviation:

            classification = "Lower Mental Health Score Range"

            explanation = (
                "The predicted score is below the typical "
                "range observed in the project dataset."
            )

            result_color = "#3b82f6"
            result_background = "#172554"
            icon = "🔵"


        elif prediction > average_score + standard_deviation:

            classification = "Higher Mental Health Score Range"

            explanation = (
                "The predicted score is above the typical "
                "range observed in the project dataset."
            )

            result_color = "#ef4444"
            result_background = "#450a0a"
            icon = "🔴"


        else:

            classification = "Moderate Mental Health Score Range"

            explanation = (
                "The predicted score falls within the typical "
                "range observed in the project dataset."
            )

            result_color = "#f59e0b"
            result_background = "#451a03"
            icon = "🟡"


        # ----------------------------------------------
        # RESULT
        # ----------------------------------------------

        st.markdown(
            f"""
            <div style="
                background-color: {result_background};
                border-left: 5px solid {result_color};
                padding: 18px 20px;
                border-radius: 8px;
                margin-top: 15px;
                margin-bottom: 15px;
            ">

                <div style="
                    color: {result_color};
                    font-size: 18px;
                    font-weight: 600;
                ">
                    Predicted Mental Health Score:
                    {prediction:.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------

        st.markdown(
            f"""
            <div style="
                background-color: #19324a;
                padding: 20px;
                border-radius: 8px;
                margin-top: 10px;
            ">

                <div style="
                    color: {result_color};
                    font-size: 20px;
                    font-weight: 700;
                    margin-bottom: 8px;
                ">
                    {icon} {classification}
                </div>

                <div style="
                    color: #e5e7eb;
                    font-size: 16px;
                    line-height: 1.5;
                ">
                    {explanation}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.caption(
            "These categories are based on the project dataset "
            "and are not clinical diagnostic categories."
        )


    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    '<div class="footer">'
    '📱 Social Media Impact Analyzer<br>'
    'Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
