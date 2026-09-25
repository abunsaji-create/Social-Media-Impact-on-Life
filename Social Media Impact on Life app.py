import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Social Media Impact Analyzer",
    page_icon="📱",
    layout="centered"
)


# ==================================================
# FIND APP FOLDER
# ==================================================

BASE_DIR = Path(__file__).resolve().parent


# ==================================================
# PAGE DESIGN
# ==================================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)


# ==================================================
# LOAD MACHINE LEARNING MODEL
# ==================================================

model_path = BASE_DIR / "social_media_model.pkl"

try:

    model = joblib.load(model_path)

except FileNotFoundError:

    st.error("❌ Model file not found.")

    st.info(
        "Please make sure 'social_media_model.pkl' "
        "is in the same folder as this app.py file."
    )

    st.info(f"Expected folder: {BASE_DIR}")

    st.stop()

except Exception as e:

    st.error(f"❌ Error loading model: {e}")

    st.stop()


# ==================================================
# FIND DATASET
# ==================================================

required_columns = {
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Age",
    "Mental_Health_Score"
}

csv_files = list(BASE_DIR.glob("*.csv"))

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
# DATASET ERROR
# ==================================================

if df is None:

    st.error("❌ Dataset could not be found.")

    st.write("CSV files found in the app folder:")

    if csv_files:

        for file in csv_files:

            st.write(f"• {file.name}")

    else:

        st.write("No CSV files were found.")

    st.write("")

    st.write("Your dataset must contain these columns:")

    for column in required_columns:

        st.write(f"• {column}")

    st.info(f"App folder: {BASE_DIR}")

    st.stop()


# ==================================================
# TITLE
# ==================================================

st.markdown(
    '<div class="main-title">📱 Social Media Impact Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Enter your details to predict your mental health score.'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# USER INPUT
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
# PREDICT BUTTON
# ==================================================

if st.button("🔮 Predict"):

    try:

        # ------------------------------------------
        # CREATE INPUT DATA
        # ------------------------------------------

        input_data = pd.DataFrame({

            "Avg_Daily_Usage_Hours": [usage],

            "Sleep_Hours_Per_Night": [sleep],

            "Age": [age]

        })


        # ------------------------------------------
        # MAKE PREDICTION
        # ------------------------------------------

        prediction = float(
            model.predict(input_data)[0]
        )


        # ------------------------------------------
        # DATASET STATISTICS
        # ------------------------------------------

        average_score = float(
            df["Mental_Health_Score"].mean()
        )

        standard_deviation = float(
            df["Mental_Health_Score"].std()
        )


        lower_limit = (
            average_score - standard_deviation
        )

        upper_limit = (
            average_score + standard_deviation
        )


        # ------------------------------------------
        # CLASSIFICATION
        # ------------------------------------------

        if prediction < lower_limit:

            classification = (
                "Lower Mental Health Score Range"
            )

            explanation = (
                "The predicted score is below the typical "
                "range observed in the project dataset."
            )

            icon = "🔵"


        elif prediction > upper_limit:

            classification = (
                "Higher Mental Health Score Range"
            )

            explanation = (
                "The predicted score is above the typical "
                "range observed in the project dataset."
            )

            icon = "🔴"


        else:

            classification = (
                "Moderate Mental Health Score Range"
            )

            explanation = (
                "The predicted score falls within the typical "
                "range observed in the project dataset."
            )

            icon = "🟡"


        # ==================================================
        # DISPLAY PREDICTION
        # ==================================================

        st.subheader(
            f"🔮 Predicted Mental Health Score: {prediction:.2f}"
        )


        # ==================================================
        # DISPLAY CLASSIFICATION
        # ==================================================

        if prediction < lower_limit:

            st.info(
                f"{icon} **{classification}**\n\n"
                f"{explanation}"
            )


        elif prediction > upper_limit:

            st.error(
                f"{icon} **{classification}**\n\n"
                f"{explanation}"
            )


        else:

            st.warning(
                f"{icon} **{classification}**\n\n"
                f"{explanation}"
            )


        # ==================================================
        # DISCLAIMER
        # ==================================================

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
