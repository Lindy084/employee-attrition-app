import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Load model and features
model, features = joblib.load("attrition_model.pkl")

# Page config
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📉", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📝 Single Prediction", "📥 Bulk Prediction"])

# --- Helper functions ---
def encode_input(df):
    mappings = {
        "BusinessTravel": {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2},
        "Department": {"Sales": 2, "Research & Development": 1, "Human Resources": 0},
        "EducationField": {"Life Sciences": 1, "Medical": 2, "Marketing": 0, "Technical Degree": 4, "Other": 3},
        "Gender": {"Male": 1, "Female": 0},
        "JobRole": {
            "Sales Executive": 6,
            "Research Scientist": 4,
            "Laboratory Technician": 3,
            "Manager": 2
        },
        "MaritalStatus": {"Single": 2, "Married": 1, "Divorced": 0},
        "OverTime": {"Yes": 1, "No": 0}
    }
    encoded_df = df.copy()
    for col, mapping in mappings.items():
        if col in encoded_df.columns:
            encoded_df[col] = encoded_df[col].map(mapping)
    return encoded_df[features]

def plot_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Attrition Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if probability >= 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': 'lightgreen'},
                {'range': [50, 100], 'color': 'lightcoral'}
            ],
        }
    ))
    return fig

# --- Pages ---

if page == "🏠 Home":
    st.title("📉 Welcome to the Employee Attrition Predictor")
    st.markdown("""
    ### About this App
    This tool helps HR professionals and managers predict which employees are at risk of leaving the company.
    
    By analyzing key employee attributes such as age, department, job role, and overtime status, it predicts the likelihood of attrition using a trained machine learning model.

    ---
    ### How to Use
    - **Single Prediction:** Input details for one employee in the sidebar and get an instant prediction.
    - **Bulk Prediction:** Upload a CSV file containing multiple employee records to get predictions and visual insights.
    
    ---
    ### Why It Matters
    Employee attrition is costly for companies. Early detection helps with retention strategies, saving time and resources.
    
    ---
    Built with ❤️ by Lindy — CAPACITI AI Academy 2025
    """)

elif page == "📝 Single Prediction":
    st.title("📝 Single Employee Attrition Prediction")

    st.sidebar.header("Enter Employee Details")

    # New fields
    name = st.sidebar.text_input("First Name")
    surname = st.sidebar.text_input("Surname")
    id_number = st.sidebar.text_input("ID Number")

    age = st.sidebar.slider("Age", 18, 60, 30)
    business_travel = st.sidebar.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    education = st.sidebar.selectbox("Education Level", [1, 2, 3, 4, 5])
    education_field = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
    overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

    user_data = {
        "Name": name,
        "Surname": surname,
        "IDNumber": id_number,
        "Age": age,
        "BusinessTravel": business_travel,
        "Department": department,
        "Education": education,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "OverTime": overtime
    }
    input_df = pd.DataFrame([user_data])

    if st.button("🔍 Predict Attrition"):
        processed = encode_input(input_df)
        prediction = model.predict(processed)[0]
        proba = model.predict_proba(processed)[0][1]

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        # Show Name, Surname, ID for clarity
        st.markdown(f"**Employee:** {name} {surname} (ID: {id_number})")

        if prediction == 1:
            st.error(f"⚠️ **High Risk**: This employee may leave. (Confidence: {proba:.2f})")
        else:
            st.success(f"✅ **Low Risk**: This employee is likely to stay. (Confidence: {1 - proba:.2f})")

        gauge_fig = plot_gauge(proba)
        st.plotly_chart(gauge_fig)

elif page == "📥 Bulk Prediction":
    st.title("📥 Bulk Employee Attrition Prediction (CSV Upload)")

    uploaded_file = st.file_uploader("Upload a CSV file with employee data", type=["csv"])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.markdown("### 👀 Preview of Uploaded Data:")
            st.dataframe(df_uploaded)

            encoded_bulk = encode_input(df_uploaded)
            preds = model.predict(encoded_bulk)
            probs = model.predict_proba(encoded_bulk)[:, 1]

            df_uploaded["Attrition Risk"] = preds
            df_uploaded["Confidence"] = [round(p, 2) for p in probs]
            df_uploaded["Prediction"] = df_uploaded["Attrition Risk"].map({1: "⚠️ Likely to Leave", 0: "✅ Likely to Stay"})

            st.markdown("### 📊 Bulk Prediction Results:")
            # Show Name, Surname, IDNumber if available, else show all columns except the newly added ones
            display_cols = []
            for col in ["Name", "Surname", "IDNumber"]:
                if col in df_uploaded.columns:
                    display_cols.append(col)
            display_cols += ["Prediction", "Confidence"] + [col for col in df_uploaded.columns if col not in display_cols + ["Prediction", "Confidence", "Attrition Risk"]]

            st.dataframe(df_uploaded[display_cols])

            # Pie charts
            st.markdown("### 📊 Attrition Insights")

            attrition_counts = df_uploaded["Prediction"].value_counts().reset_index()
            attrition_counts.columns = ["Attrition Status", "Count"]
            fig1 = px.pie(attrition_counts, values="Count", names="Attrition Status",
                          title="Overall Attrition Distribution",
                          color="Attrition Status",
                          color_discrete_map={"✅ Likely to Stay": "green", "⚠️ Likely to Leave": "red"})
            st.plotly_chart(fig1)

            if "Department" in df_uploaded.columns:
                dept_attrition = df_uploaded.groupby(["Department", "Prediction"]).size().reset_index(name="Count")
                fig2 = px.pie(dept_attrition, values="Count", names="Prediction",
                              title="Attrition Distribution by Department",
                              color="Prediction",
                              color_discrete_map={"✅ Likely to Stay": "green", "⚠️ Likely to Leave": "red"},
                              facet_col="Department", facet_col_wrap=2)
                st.plotly_chart(fig2)

            csv = df_uploaded.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results as CSV", csv, file_name="attrition_predictions.csv", mime='text/csv')

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

st.markdown("---")
st.caption("👩‍💻 Built by Lindy • CAPACITI AI Academy • 2025")
