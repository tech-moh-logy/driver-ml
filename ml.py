import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
data = {
    "Week": list(range(1, 15)),  # Weeks 1 to 14
    "Hybrid Gas ($)": [115, 130, 120, 125, 130, 130, 140, 145, 150, 155, 160, 162, 168, 170],
    "Hybrid Miles": [747, 773, 757, 797, 789, 818, 835, 850, 860, 880, 895, 910, 930, 945],
    "Non-Hybrid Gas ($)": [165, 185, 175, 210, 200, 215, 210, 225, 230, 240, 250, 260, 265, 270],
    "Non-Hybrid Miles": [753, 780, 762, 805, 790, 798, 824, 840, 855, 875, 890, 900, 920, 940],
    
    # Varying hours: Hybrid drives more on weeks 4, 7, 10, and 13
    "Hybrid Hours Driven": [70, 72, 71, 80, 74, 78, 85, 76, 79, 88, 80, 83, 90, 85],  
    "Non-Hybrid Hours Driven": [72, 74, 73, 78, 76, 80, 82, 78, 80, 85, 82, 85, 87, 88]  
}



df = pd.DataFrame(data)

# Calculate cost per mile
df["Hybrid Cost Per Mile"] = df["Hybrid Gas ($)"] / df["Hybrid Miles"]
df["Non-Hybrid Cost Per Mile"] = df["Non-Hybrid Gas ($)"] / df["Non-Hybrid Miles"]

# Streamlit Dashboard
st.title("🚖 Fuel Cost & Efficiency Dashboard")

st.markdown("""
This interactive dashboard showcases the **cost efficiency and predictive fuel savings** of Hybrid vs. Non-Hybrid Toyota Highlanders.  
Key features:
- **Fuel Cost Comparison**
- **Cost per Mile Analysis**
- **Machine Learning Prediction of Future Savings**
""")

# Sidebar Inputs
st.sidebar.header("Customize Parameters")
fuel_price = st.sidebar.number_input("Gas Price per Gallon ($)", min_value=2.0, max_value=6.0, value=3.5)
mileage = st.sidebar.number_input("Miles Driven per Week", min_value=500, max_value=2000, value=750)

# Display Dataset
st.subheader("📊 Weekly Fuel Cost Data")
st.dataframe(df)

# Cost Per Mile Comparison
st.subheader("⚡ Cost Per Mile Comparison")
fig, ax = plt.subplots()
ax.plot(df["Week"], df["Hybrid Cost Per Mile"], label="Hybrid", marker="o", linestyle="-")
ax.plot(df["Week"], df["Non-Hybrid Cost Per Mile"], label="Non-Hybrid", marker="o", linestyle="-")
ax.set_xlabel("Week")
ax.set_ylabel("Cost per Mile ($)")
ax.set_title("Hybrid vs. Non-Hybrid Cost Per Mile")
ax.legend()
st.pyplot(fig)

# ML Model - Predict Savings
st.subheader("🔮 Predictive Model: Fuel Savings Over Time")

# Train Model (Simple Linear Regression)
X = np.array(df["Week"]).reshape(-1, 1)
y_hybrid = np.array(df["Hybrid Gas ($)"])
y_non_hybrid = np.array(df["Non-Hybrid Gas ($)"])

model_hybrid = LinearRegression().fit(X, y_hybrid)
model_non_hybrid = LinearRegression().fit(X, y_non_hybrid)

# Predict Future Costs
weeks_future = np.array([8, 9, 10, 11, 12]).reshape(-1, 1)
hybrid_pred = model_hybrid.predict(weeks_future)
non_hybrid_pred = model_non_hybrid.predict(weeks_future)

# Plot Prediction
fig, ax = plt.subplots()
ax.plot(df["Week"], df["Hybrid Gas ($)"], label="Hybrid - Actual", marker="o")
ax.plot(df["Week"], df["Non-Hybrid Gas ($)"], label="Non-Hybrid - Actual", marker="o")
ax.plot([8, 9, 10, 11, 12], hybrid_pred, label="Hybrid - Predicted", linestyle="--", color="blue")
ax.plot([8, 9, 10, 11, 12], non_hybrid_pred, label="Non-Hybrid - Predicted", linestyle="--", color="red")
ax.set_xlabel("Week")
ax.set_ylabel("Fuel Cost ($)")
ax.set_title("Predicted Fuel Cost Over Time")
ax.legend()
st.pyplot(fig)

# Summary
savings = np.mean(non_hybrid_pred - hybrid_pred)
st.success(f"📉 Estimated Future Savings: **${savings:.2f} per week** by switching to Hybrid.")

# Footer
st.markdown("**Developed by [Your Name] - Machine Learning & Predictive Modeling**")
