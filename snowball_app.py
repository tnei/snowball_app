import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

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

# Generate dummy customer data for segmentation and cohort analysis
def generate_customer_data():
    np.random.seed(42)
    customer_data = {
        'customer_id': range(1, 101),
        'spend': np.random.randint(500, 5000, 100),
        'contract_length': np.random.randint(6, 48, 100),
        'sign_up_date': pd.date_range(start="2020-01-01", periods=100, freq='7D'),
        'last_active': pd.date_range(start="2020-01-01", periods=100, freq='7D') + pd.to_timedelta(np.random.randint(30, 365, 100), unit='D')
    }
    return pd.DataFrame(customer_data)

# Helper functions to create interactive charts with Plotly
def plot_revenue_forecast(df):
    fig = px.line(df, x='date', y='total_arr', title='Monthly Revenue Forecast', labels={'total_arr': 'Revenue ($)'})
    fig.update_traces(line_color='green')
    st.plotly_chart(fig)

def plot_spend_forecast(df):
    fig = px.line(df, x='date', y='spend_forecast', title='Spend Forecast by Product Type', labels={'spend_forecast': 'Spend ($)'})
    fig.update_traces(line_color='blue')
    st.plotly_chart(fig)

def plot_product_profitability(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['product_1_revenue'], mode='lines', name='Product 1', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['product_2_revenue'], mode='lines', name='Product 2', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['product_3_revenue'], mode='lines', name='Product 3', line=dict(color='red')))
    fig.update_layout(title='Product Profitability Trend', xaxis_title='Date', yaxis_title='Revenue ($)')
    st.plotly_chart(fig)

def plot_product_mix(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['product_1_revenue'], stackgroup='one', name='Product 1', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['product_2_revenue'], stackgroup='one', name='Product 2', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['product_3_revenue'], stackgroup='one', name='Product 3', line=dict(color='red')))
    fig.update_layout(title='Product Mix Distribution Over Time', xaxis_title='Date', yaxis_title='Revenue ($)')
    st.plotly_chart(fig)

def plot_contract_length_distribution(df):
    fig = px.histogram(df, x='average_contract_length', nbins=10, title='Contract Length Distribution', labels={'average_contract_length': 'Contract Length (Months)'})
    st.plotly_chart(fig)

# Customer Segmentation using KMeans
def plot_customer_segmentation(customer_data):
    X = customer_data[['spend', 'contract_length']]
    kmeans = KMeans(n_clusters=3)
    customer_data['segment'] = kmeans.fit_predict(X)

    fig = px.scatter(customer_data, x='spend', y='contract_length', color='segment', title='Customer Segmentation',
                     labels={'spend': 'Customer Spend ($)', 'contract_length': 'Contract Length (Months)'}, hover_data=['customer_id'])
    st.plotly_chart(fig)

# Cohort Analysis: Group customers by sign-up month and calculate retention
def plot_cohort_analysis(customer_data):
    customer_data['cohort'] = customer_data['sign_up_date'].dt.to_period('M')
    # Convert the cohort (Period type) to string for Plotly compatibility
    customer_data['cohort'] = customer_data['cohort'].astype(str)
    
    cohort_data = customer_data.groupby('cohort')['customer_id'].count().reset_index()

    fig = px.bar(cohort_data, x='cohort', y='customer_id', title='Cohort Analysis (Sign-ups by Month)',
                 labels={'customer_id': 'Number of Customers', 'cohort': 'Cohort (Month of Sign-up)'})
    st.plotly_chart(fig)

# Define Streamlit app layout
st.markdown("<h1 style='text-align: center; color: black;'>SaaS Business Dashboard Demo</h1>", unsafe_allow_html=True)

# Load hardcoded dummy data
data = load_dummy_data()

# Sidebar filters
st.sidebar.header("Filter Options")
start_month = st.sidebar.date_input("Start Date", value=data['date'].min())
end_month = st.sidebar.date_input("End Date", value=data['date'].max())
product_selection = st.sidebar.multiselect("Select products:", ['Design Live', 'Winner Flex', 'Insight'], default=['Design Live', 'Winner Flex'])
forecast_period = st.sidebar.slider("Forecast period (months):", 1, 12, value=6)
contract_length_range = st.sidebar.slider("Contract length (months):", 2, 48, value=(2, 48))

# Convert input to pandas datetime if necessary (in case the input is not a datetime object)
if isinstance(start_month, pd.Timestamp):
    start_month = start_month
else:
    start_month = pd.to_datetime(start_month)

if isinstance(end_month, pd.Timestamp):
    end_month = end_month
else:
    end_month = pd.to_datetime(end_month)

# Ensure end_month is not earlier than start_month
if start_month > end_month:
    st.sidebar.error("End date must be after the start date.")

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

# Generate dummy customer data for Advanced Analytics
customer_data = generate_customer_data()

# Customer Segmentation Plot
st.markdown("<h3>Customer Segmentation</h3>", unsafe_allow_html=True)
plot_customer_segmentation(customer_data)

# Cohort Analysis Plot
st.markdown("<h3>Cohort Analysis</h3>", unsafe_allow_html=True)
plot_cohort_analysis(customer_data)

# AI-driven Q&A Section
st.markdown("<h2 style='text-align: center; color: black;'>AI-driven Q&A Section (placeholder)</h2>", unsafe_allow_html=True)
st.text("Cortex Analyst chat bot integration can be added here.")
