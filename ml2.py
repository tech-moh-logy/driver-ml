import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset (Matches Your Data Exactly)
data = {
    "Week": list(range(1, 15)),  

    # Toyota Highlander Hybrid
    "Highlander Hybrid Gas ($)": [115, 130, 120, 125, 130, 130, 140, 145, 150, 155, 160, 162, 168, 170],
    "Highlander Hybrid Miles": [747, 773, 757, 797, 789, 818, 835, 850, 860, 880, 895, 910, 930, 945],

    # Toyota Highlander Non-Hybrid (Gas Only)
    "Highlander Non-Hybrid Gas ($)": [165, 185, 175, 210, 200, 215, 210, 225, 230, 240, 250, 260, 265, 270],
    "Highlander Non-Hybrid Miles": [753, 780, 762, 805, 790, 798, 824, 840, 855, 875, 890, 900, 920, 940],
    
    # Toyota Sienna Hybrid (More Efficient Hybrid)
    "Sienna Hybrid Gas ($)": [100, 115, 110, 118, 115, 120, 125, 130, 135, 138, 140, 142, 145, 148],
    "Sienna Hybrid Miles": [770, 790, 780, 810, 800, 825, 840, 860, 875, 890, 905, 920, 940, 955],

    # Hours Driven
    "Highlander Hybrid Hours Driven": [70, 72, 71, 80, 74, 78, 85, 76, 79, 88, 80, 83, 90, 85],  
    "Highlander Non-Hybrid Hours Driven": [72, 74, 73, 78, 76, 80, 82, 78, 80, 85, 82, 85, 87, 88],
    "Sienna Hybrid Hours Driven": [75, 77, 76, 82, 78, 80, 86, 79, 81, 90, 83, 85, 92, 88],  

    # Mercedes EQC (Electric)
    "Mercedes EQC Charging Cost ($)": [40, 42, 39, 41, 38, 37, 35, 39, 42, 44, 46, 45, 48, 50],  
    "Mercedes EQC Miles": [370, 360, 343, 356, 357, 352, 359, 372, 384, 391, 389, 370, 381, 371],  
    "Mercedes EQC Hours Driven": [40, 37, 40, 43, 38, 34, 36, 39, 47, 43, 44, 43, 47, 43],  

    # Tesla Model 3 (Better Efficiency than EQC)
    "Tesla Model 3 Charging Cost ($)": [35, 37, 34, 36, 33, 31, 30, 33, 36, 38, 39, 37, 40, 42],  
    "Tesla Model 3 Miles": [410, 400, 390, 405, 398, 390, 400, 415, 428, 440, 435, 420, 432, 425],  
    "Tesla Model 3 Hours Driven": [45, 42, 44, 48, 43, 40, 42, 45, 50, 48, 49, 47, 51, 50]  
}

df = pd.DataFrame(data)

# Calculate cost per mile
df["Highlander Hybrid Cost Per Mile"] = df["Highlander Hybrid Gas ($)"] / df["Highlander Hybrid Miles"]
df["Highlander Non-Hybrid Cost Per Mile"] = df["Highlander Non-Hybrid Gas ($)"] / df["Highlander Non-Hybrid Miles"]
df["Sienna Hybrid Cost Per Mile"] = df["Sienna Hybrid Gas ($)"] / df["Sienna Hybrid Miles"]

# Streamlit Dashboard
st.title("🚖 Fuel Cost & Efficiency Dashboard")

st.markdown("""
This interactive dashboard showcases the **cost efficiency and predictive fuel savings** of different vehicles.  
Key features:
- **Hybrid vs. Non-Hybrid Fuel Cost Comparison**
- **Electric vs. Gasoline Efficiency**
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
ax.plot(df["Week"], df["Highlander Hybrid Cost Per Mile"], label="Highlander Hybrid", marker="o", linestyle="-")
ax.plot(df["Week"], df["Highlander Non-Hybrid Cost Per Mile"], label="Highlander Non-Hybrid", marker="o", linestyle="-")
ax.plot(df["Week"], df["Sienna Hybrid Cost Per Mile"], label="Sienna Hybrid", marker="o", linestyle="-")
ax.set_xlabel("Week")
ax.set_ylabel("Cost per Mile ($)")
ax.set_title("Hybrid vs. Non-Hybrid Cost Per Mile")
ax.legend()
st.pyplot(fig)

# ML Model - Predict Savings
st.subheader("🔮 Predictive Model: Fuel Savings Over Time")

# Train Model (Simple Linear Regression)
X = np.array(df["Week"]).reshape(-1, 1)
y_hybrid = np.array(df["Highlander Hybrid Gas ($)"])
y_non_hybrid = np.array(df["Highlander Non-Hybrid Gas ($)"])

model_hybrid = LinearRegression().fit(X, y_hybrid)
model_non_hybrid = LinearRegression().fit(X, y_non_hybrid)

# Predict Future Costs
weeks_future = np.array([15, 16, 17, 18, 19, 20]).reshape(-1, 1)
hybrid_pred = model_hybrid.predict(weeks_future)
non_hybrid_pred = model_non_hybrid.predict(weeks_future)

# Plot Prediction
fig, ax = plt.subplots()
ax.plot(df["Week"], df["Highlander Hybrid Gas ($)"], label="Hybrid - Actual", marker="o")
ax.plot(df["Week"], df["Highlander Non-Hybrid Gas ($)"], label="Non-Hybrid - Actual", marker="o")
ax.plot(weeks_future, hybrid_pred, label="Hybrid - Predicted", linestyle="--", color="blue")
ax.plot(weeks_future, non_hybrid_pred, label="Non-Hybrid - Predicted", linestyle="--", color="red")
ax.set_xlabel("Week")
ax.set_ylabel("Fuel Cost ($)")
ax.set_title("Predicted Fuel Cost Over Time")
ax.legend()
st.pyplot(fig)

# Summary
savings = np.mean(non_hybrid_pred - hybrid_pred)
st.success(f"📉 Estimated Future Savings: **${savings:.2f} per week** by switching to a Highlander Hybrid.")

# Footer
st.markdown("**Developed by [Your Name] - Machine Learning & Predictive Modeling**")
