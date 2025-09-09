import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Mock patient dataset
patients = pd.DataFrame({
    "Patient ID": [f"P{i}" for i in range(1, 11)],
    "Disease": np.random.choice(["Diabetes", "Obesity", "Heart Failure"], size=10),
    "Risk Score": np.random.rand(10)
})
patients["Risk Level"] = pd.cut(
    patients["Risk Score"], bins=[0, 0.33, 0.66, 1],
    labels=["Low", "Medium", "High"]
)

st.title("AI-Driven Risk Prediction Dashboard")

# --- Cohort View ---
st.subheader("Cohort View: Risk Scores by Disease")

# Disease filter
disease_filter = st.selectbox("Filter by Disease:", ["All"] + patients["Disease"].unique().tolist())
if disease_filter != "All":
    filtered_patients = patients[patients["Disease"] == disease_filter]
else:
    filtered_patients = patients

st.dataframe(filtered_patients)

# --- Patient Detail View ---
st.subheader("Patient Detail View")

selected_patient = st.selectbox("Choose a patient:", patients["Patient ID"])
row = patients[patients["Patient ID"] == selected_patient].iloc[0]

st.metric("Risk Probability", f"{row['Risk Score']:.2f}")
st.write(f"**Risk Level:** {row['Risk Level']}")
st.write(f"**Disease Type:** {row['Disease']}")

# --- Graphs: vitals/labs trends ---
st.write("### Patient Trends (Last 90 Days)")
days = np.arange(1, 91)
fig, ax = plt.subplots(figsize=(6, 3))  

if row["Disease"] == "Diabetes":
    glucose = 100 + np.random.randn(90).cumsum()
    ax.plot(days, glucose, color="red", linewidth=2, marker="o", markersize=3, label="Glucose")
    ax.set_title("Glucose Trend (Last 90 Days)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Day", fontsize=10)
    ax.set_ylabel("Glucose (mg/dL)", fontsize=10)

elif row["Disease"] == "Obesity":
    bmi = 30 + np.random.randn(90).cumsum() * 0.01
    ax.plot(days, bmi, color="blue", linewidth=2, marker="s", markersize=3, label="BMI")
    ax.set_title("BMI Trend (Last 90 Days)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Day", fontsize=10)
    ax.set_ylabel("BMI", fontsize=10)

elif row["Disease"] == "Heart Failure":
    bp = 120 + np.random.randn(90).cumsum() * 0.2
    ax.plot(days, bp, color="green", linewidth=2, marker="^", markersize=3, label="BP")
    ax.set_title("Blood Pressure Trend (Last 90 Days)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Day", fontsize=10)
    ax.set_ylabel("BP (mmHg)", fontsize=10)

# Style improvements
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.5)  
st.pyplot(fig)

# --- Explanations ---
st.write("### Top 3 Risk Drivers")
if row["Disease"] == "Diabetes":
    st.write("- Rising HbA1c (last 30 days)")
    st.write("- Missed 20% medications")
    st.write("- Glucose variability high")

elif row["Disease"] == "Obesity":
    st.write("- BMI rising trend")
    st.write("- Low daily steps")
    st.write("- Poor sleep pattern")

elif row["Disease"] == "Heart Failure":
    st.write("- High blood pressure variability")
    st.write("- Missed medication refills")
    st.write("- Reduced physical activity")

# --- Suggested Actions ---
st.write("### Recommended Next Actions")
if row["Disease"] == "Diabetes":
    st.write("🔹 Increase frequency of glucose monitoring")
    st.write("🔹 Review medication adherence")
    st.write("🔹 Schedule dietitian consult")

elif row["Disease"] == "Obesity":
    st.write("🔹 Encourage physical activity (target +2000 steps/day)")
    st.write("🔹 Nutrition counseling")
    st.write("🔹 Track sleep consistency")

elif row["Disease"] == "Heart Failure":
    st.write("🔹 Monitor BP daily")
    st.write("🔹 Adjust diuretics if necessary")
    st.write("🔹 Cardiology follow-up within 2 weeks")
