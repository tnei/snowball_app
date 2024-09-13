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
    plt.plot(df['date'], df['total_arr'], label="Total ARR Forecast", color='green')
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.title('Monthly Revenue Forecast')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def plot_spend_forecast(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['spend_forecast'], label="Spend Forecast", color='blue')
    plt.xlabel('Date')
    plt.ylabel('Spend ($)')
    plt.title('Spend Forecast by Product Type')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def plot_product_profitability(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['product_1_revenue'], label="Product 1", color='orange')
    plt.plot(df['date'], df['product_2_revenue'], label="Product 2", color='purple')
    plt.plot(df['date'], df['product_3_revenue'], label="Product 3", color='red')
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.title('Product Profitability Trend')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def plot_product_mix(df):
    plt.figure(figsize=(10, 6))
    plt.stackplot(df['date'], df['product_1_revenue'], df['product_2_revenue'], df['product_3_revenue'], 
                  labels=["Product 1", "Product 2", "Product 3"], colors=['orange', 'purple', 'red'])
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.title('Product Mix Distribution Over Time')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def plot_contract_length_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['average_contract_length'], bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Contract Length (Months)')
    plt.ylabel('Count')
    plt.title('Contract Length Distribution')
    plt.grid(True)
    st.pyplot(plt)

# Define Streamlit app layout
st.markdown("<h1 style='text-align: center; color: black;'>SaaS Business Dashboard Demo</h1>", unsafe_allow_html=True)

# Load hardcoded dummy data
data = load_dummy_data()

# Sidebar filters
st.sidebar.header("Filter Options")
start_month, end_month = st.sidebar.select_slider("Select date range:", options=data['date'], value=(data['date'].min(), data['date'].max()))
product_selection = st.sidebar.multiselect("Select products:", ['Product 1', 'Product 2', 'Product 3'], default=['Product 1', 'Product 2', 'Product 3'])
forecast_period = st.sidebar.slider("Forecast period (months):", 1, 12, value=6)
contract_length_range = st.sidebar.slider("Contract length (months):", 2, 48, value=(2, 48))

# Filter data based on user selections
filtered_data = data[(data['date'] >= start_month) & (data['date'] <= end_month)]

# Overview section
st.markdown("<h2 style='text-align: center; color: black;'>Overview</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", filtered_data['total_customers'].sum())
col2.metric("Churn Percentage", f"{filtered_data['churn_percentage'].mean():.2%}")
col3.metric("Avg Contract Length", f"{filtered_data['average_contract_length'].mean():.1f} months")
col4.metric("Total ARR", f"${filtered_data['total_arr'].sum():,}")

# Monthly Revenue Forecast Chart
st.markdown("<h2 style='color: black;'>Monthly Revenue Forecast</h2>", unsafe_allow_html=True)
plot_revenue_forecast(filtered_data)

# Spend Forecast by Product Type Chart
st.markdown("<h2 style='color: black;'>Spend Forecast by Product Type</h2>", unsafe_allow_html=True)
plot_spend_forecast(filtered_data)

# Product Profitability Trend Chart
st.markdown("<h2 style='color: black;'>Product Profitability Trend</h2>", unsafe_allow_html=True)
plot_product_profitability(filtered_data)

# Product Insights section
st.markdown("<h2 style='text-align: center; color: black;'>Product Insights</h2>", unsafe_allow_html=True)

# Product Mix Distribution Over Time Chart
st.markdown("<h3 style='color: black;'>Product Mix Distribution Over Time</h3>", unsafe_allow_html=True)
plot_product_mix(filtered_data)

# Contract Length Distribution Chart
st.markdown("<h3 style='color: black;'>Contract Length Distribution</h3>", unsafe_allow_html=True)
plot_contract_length_distribution(filtered_data)

# Advanced Analytics Section
st.markdown("<h2 style='text-align: center; color: black;'>Advanced Analytics</h2>", unsafe_allow_html=True)
st.markdown("<h3>Customer Segmentation Plot (placeholder)</h3>", unsafe_allow_html=True)
# Placeholder for customer segmentation plot - implement with clustering if needed

st.markdown("<h3>Cohort Analysis (placeholder)</h3>", unsafe_allow_html=True)
# Placeholder for cohort analysis - add cohort logic here

# AI-driven Q&A Section
st.markdown("<h2 style='text-align: center; color: black;'>AI-driven Q&A Section (placeholder)</h2>", unsafe_allow_html=True)
st.text("Cortex Analyst chat bot integration can be added here.")

# Save this as app.py and run it via 'streamlit run app.py' in your terminal
