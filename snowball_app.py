import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hardcoded dummy data for demo purposes
def load_dummy_data():
    dates = pd.date_range(start="2023-01-01", periods=24, freq='M')
    data = {
        'date': dates,
        'total_customers': np.random.randint(500, 1500, size=len(dates)),
        'churn_percentage': np.random.uniform(0.01, 0.15, size=len(dates)),
        'average_contract_length': np.random.randint(6, 48, size=len(dates)),
        'total_arr': np.random.randint(100000, 1000000, size=len(dates)),
        'spend_forecast': np.random.randint(50000, 200000, size=len(dates)),
        'product_1_revenue': np.random.randint(50000, 200000, size=len(dates)),
        'product_2_revenue': np.random.randint(50000, 200000, size=len(dates)),
        'product_3_revenue': np.random.randint(30000, 150000, size=len(dates)),
    }
    return pd.DataFrame(data)

# Helper functions to create charts
def plot_revenue_forecast(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['total_arr'], label="Total ARR Forecast")
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.title('Monthly Revenue Forecast')
    plt.legend()
    st.pyplot(plt)

def plot_spend_forecast(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['spend_forecast'], label="Spend Forecast")
    plt.xlabel('Date')
    plt.ylabel('Spend ($)')
    plt.title('Spend Forecast by Product Type')
    plt.legend()
    st.pyplot(plt)

def plot_product_profitability(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['product_1_revenue'], label="Product 1")
    plt.plot(df['date'], df['product_2_revenue'], label="Product 2")
    plt.plot(df['date'], df['product_3_revenue'], label="Product 3")
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.title('Product Profitability Trend')
    plt.legend()
    st.pyplot(plt)

def plot_product_mix(df):
    plt.figure(figsize=(10, 6))
    plt.stackplot(df['date'], df['product_1_revenue'], df['product_2_revenue'], df['product_3_revenue'], labels=["Product 1", "Product 2", "Product 3"])
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.title('Product Mix Distribution Over Time')
    plt.legend()
    st.pyplot(plt)

def plot_contract_length_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['average_contract_length'], bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Contract Length (Months)')
    plt.ylabel('Count')
    plt.title('Contract Length Distribution')
    st.pyplot(plt)

# Define Streamlit app layout
st.title("SaaS Business Dashboard Demo")

# Load hardcoded dummy data
data = load_dummy_data()

# Sidebar filters
st.sidebar.header("Filter options")
start_month, end_month = st.sidebar.select_slider("Select date range:", options=data['date'], value=(data['date'].min(), data['date'].max()))
product_selection = st.sidebar.multiselect("Select products:", ['Product 1', 'Product 2', 'Product 3'], default=['Product 1', 'Product 2', 'Product 3'])
forecast_period = st.sidebar.slider("Forecast period (months):", 1, 12, value=6)
contract_length_range = st.sidebar.slider("Contract length (months):", 2, 48, value=(2, 48))

# Filter data based on user selections
filtered_data = data[(data['date'] >= start_month) & (data['date'] <= end_month)]

# Overview section
st.header("Overview")
st.metric("Total Customers", filtered_data['total_customers'].sum())
st.metric("Churn Percentage", f"{filtered_data['churn_percentage'].mean():.2%}")
st.metric("Average Contract Length", f"{filtered_data['average_contract_length'].mean():.1f} months")
st.metric("Total ARR", f"${filtered_data['total_arr'].sum():,}")

# Monthly Revenue Forecast Chart
st.subheader("Monthly Revenue Forecast")
plot_revenue_forecast(filtered_data)

# Spend Forecast by Product Type Chart
st.subheader("Spend Forecast by Product Type")
plot_spend_forecast(filtered_data)

# Product Profitability Trend Chart
st.subheader("Product Profitability Trend")
plot_product_profitability(filtered_data)

# Product Insights section
st.header("Product Insights")

# Product Mix Distribution Over Time Chart
st.subheader("Product Mix Distribution Over Time")
plot_product_mix(filtered_data)

# Contract Length Distribution Chart
st.subheader("Contract Length Distribution")
plot_contract_length_distribution(filtered_data)

# Advanced Analytics Section
st.header("Advanced Analytics")
st.subheader("Customer Segmentation Plot (placeholder)")
# Placeholder for customer segmentation plot - implement with clustering if needed

st.subheader("Cohort Analysis (placeholder)")
# Placeholder for cohort analysis - add cohort logic here

# AI-driven Q&A Section
st.header("AI-driven Q&A Section (placeholder)")
st.text("Cortex Analyst chat bot integration can be added here.")

# Save this as app.py and run it via 'streamlit run app.py' in your terminal
