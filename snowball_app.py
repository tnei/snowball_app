import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from streamlit_sortable import st_sortable  # Import streamlit-sortable

# Function to load dummy data for the main dashboard
def load_dummy_data():
    # Generate dates for 24 months starting from January 2023
    dates = pd.date_range(start="2023-01-01", periods=24, freq='M')
    # Create a dictionary with random data for each metric
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
    # Return the data as a DataFrame
    return pd.DataFrame(data)

# Function to generate dummy customer data for segmentation and cohort analysis
def generate_customer_data():
    np.random.seed(42)
    customer_data = {
        'customer_id': range(1, 101),
        'spend': np.random.randint(500, 5000, 100),
        'contract_length': np.random.randint(6, 48, 100),
        'sign_up_date': pd.date_range(start="2020-01-01", periods=100, freq='7D'),
        'last_active': pd.date_range(start="2020-01-01", periods=100, freq='7D') +
                       pd.to_timedelta(np.random.randint(30, 365, 100), unit='D')
    }
    return pd.DataFrame(customer_data)

# Function to generate dummy cohort data with retention rates
def generate_cohort_data():
    np.random.seed(42)
    # Generate sign-up dates in monthly cohorts
    cohort_months = pd.date_range(start="2020-01-01", periods=12, freq='M')
    # List to store retention data for each cohort
    cohort_data = []
    for i, month in enumerate(cohort_months):
        cohort_size = np.random.randint(50, 100)  # Random cohort size
        retention = [cohort_size]  # Initial cohort size
        for m in range(1, 12):  # Generate retention for the next 11 months
            retention_rate = np.random.uniform(0.5, 0.95)  # Random retention rate
            retained_customers = int(retention[-1] * retention_rate)
            retention.append(retained_customers)
        cohort_data.append(retention)
    # Create DataFrame from the retention data
    cohort_df = pd.DataFrame(cohort_data, index=cohort_months.strftime('%Y-%m'),
                             columns=[f"Month {i}" for i in range(12)])
    # Calculate retention percentage
    cohort_size = cohort_df.iloc[:, 0]
    retention_rate_df = cohort_df.divide(cohort_size, axis=0) * 100
    return retention_rate_df

# Plotting functions for various charts

# Updated plot_revenue_forecast function with interactivity
def plot_revenue_forecast(df):
    fig = px.line(df, x='date', y='total_arr', title='Monthly Revenue Forecast',
                  labels={'total_arr': 'Revenue ($)'})
    fig.update_traces(line_color='green')
    # Add interactive range slider and selectors
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Updated plot_spend_forecast function with interactivity
def plot_spend_forecast(df):
    fig = px.line(df, x='date', y='spend_forecast', title='Spend Forecast by Product Type',
                  labels={'spend_forecast': 'Spend ($)'})
    fig.update_traces(line_color='blue')
    # Add interactive range slider and selectors
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Updated plot_product_profitability function with interactivity
def plot_product_profitability(df, product_selection):
    # Create a DataFrame based on product selection
    product_cols = {
        'Product 1': 'product_1_revenue',
        'Product 2': 'product_2_revenue',
        'Product 3': 'product_3_revenue'
    }
    selected_products = [product_cols[prod] for prod in product_selection if prod in product_cols]

    if not selected_products:
        st.warning("Please select at least one product to display the chart.")
        return

    # Melt the DataFrame for Plotly
    df_melted = df.melt(id_vars=['date'], value_vars=selected_products, var_name='Product', value_name='Revenue')
    df_melted['Product'] = df_melted['Product'].map({v: k for k, v in product_cols.items()})

    # Create interactive line chart with Plotly Express
    fig = px.line(
        df_melted, x='date', y='Revenue', color='Product',
        title='Product Profitability Trend',
        labels={'Revenue': 'Revenue ($)', 'date': 'Date'},
        hover_data={'Revenue': ':.2f', 'date': True}
    )

    # Add interactive range slider and selectors
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# Updated plot_product_mix function with interactivity
def plot_product_mix(df, product_selection):
    product_cols = {
        'Product 1': 'product_1_revenue',
        'Product 2': 'product_2_revenue',
        'Product 3': 'product_3_revenue'
    }
    selected_products = [product_cols[prod] for prod in product_selection if prod in product_cols]

    if not selected_products:
        st.warning("Please select at least one product to display the chart.")
        return

    # Melt the DataFrame for Plotly
    df_melted = df.melt(id_vars=['date'], value_vars=selected_products, var_name='Product', value_name='Revenue')
    df_melted['Product'] = df_melted['Product'].map({v: k for k, v in product_cols.items()})

    # Create stacked area chart
    fig = px.area(
        df_melted, x='date', y='Revenue', color='Product',
        title='Product Mix Distribution Over Time',
        labels={'Revenue': 'Revenue ($)', 'date': 'Date'},
        groupnorm='percent'
    )

    # Add interactive range slider and selectors
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        hovermode="x unified",
        yaxis=dict(ticksuffix='%')
    )

    st.plotly_chart(fig, use_container_width=True)

# Updated plot_contract_length_distribution function with interactivity
def plot_contract_length_distribution(df):
    fig = px.histogram(df, x='average_contract_length', nbins=10,
                       title='Contract Length Distribution',
                       labels={'average_contract_length': 'Contract Length (Months)'})
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

# Customer Segmentation using KMeans with interactivity
def plot_customer_segmentation(customer_data):
    X = customer_data[['spend', 'contract_length']]
    kmeans = KMeans(n_clusters=3)
    customer_data['segment'] = kmeans.fit_predict(X)

    fig = px.scatter(customer_data, x='spend', y='contract_length', color='segment',
                     title='Customer Segmentation',
                     labels={'spend': 'Customer Spend ($)', 'contract_length': 'Contract Length (Months)'},
                     hover_data=['customer_id'])
    st.plotly_chart(fig, use_container_width=True)

# Customer Cohort Analysis - Retention Rates Heatmap
def plot_cohort_retention_heatmap(cohort_df):
    fig = px.imshow(cohort_df,
                    labels=dict(x="Cohort Period (Months)", y="Cohort (Sign-up Month)",
                                color="Retention Rate (%)"),
                    x=[f"Month {i}" for i in range(12)],
                    y=cohort_df.index,
                    color_continuous_scale='Blues')
    fig.update_layout(title="Customer Cohort Analysis - Retention Rates",
                      xaxis_title="Cohort Period (Months)",
                      yaxis_title="Cohort (Sign-up Month)")
    st.plotly_chart(fig, use_container_width=True)

# Display data table with interactive filtering
def display_data_table(df):
    st.markdown("<h2 style='color: black;'>Detailed Data View</h2>", unsafe_allow_html=True)
    # Add filters
    with st.expander("Filter Data"):
        min_revenue = st.slider("Minimum Revenue ($)", int(df['total_arr'].min()), int(df['total_arr'].max()), int(df['total_arr'].min()))
        max_revenue = st.slider("Maximum Revenue ($)", int(df['total_arr'].min()), int(df['total_arr'].max()), int(df['total_arr'].max()))
        filtered_df = df[(df['total_arr'] >= min_revenue) & (df['total_arr'] <= max_revenue)]
    st.dataframe(filtered_df)

# Main function to run the Streamlit app
def main():
    # Set the title of the app
    st.markdown("<h1 style='text-align: center; color: black;'>SaaS Business Dashboard Demo</h1>",
                unsafe_allow_html=True)
    # Load data
    data = load_dummy_data()
    customer_data = generate_customer_data()
    cohort_data = generate_cohort_data()
    # Sidebar filters
    st.sidebar.header("Filter Options")
    start_month = st.sidebar.date_input("Start Date", value=data['date'].min())
    end_month = st.sidebar.date_input("End Date", value=data['date'].max())
    global product_selection
    product_selection = st.sidebar.multiselect("Select products:",
                                               ['Product 1', 'Product 2', 'Product 3'],
                                               default=['Product 1', 'Product 2', 'Product 3'])
    forecast_period = st.sidebar.slider("Forecast period (months):", 1, 12, value=6)
    contract_length_range = st.sidebar.slider("Contract length (months):", 2, 48, value=(2, 48))
    # Convert inputs to datetime if necessary
    start_month = pd.to_datetime(start_month)
    end_month = pd.to_datetime(end_month)
    # Validate date inputs
    if start_month > end_month:
        st.sidebar.error("End date must be after the start date.")
        return  # Stop execution if dates are invalid
    # Filter data based on user selections
    filtered_data = data[(data['date'] >= start_month) & (data['date'] <= end_month)]
    # Apply contract length filter
    filtered_data = filtered_data[(filtered_data['average_contract_length'] >= contract_length_range[0]) &
                                  (filtered_data['average_contract_length'] <= contract_length_range[1])]
    # Apply product selection filter (adjusted in plotting functions)

    # Allow the user to select which sections to display
    st.sidebar.header("Customize Dashboard")
    available_widgets = [
        "Overview",
        "Monthly Revenue Forecast",
        "Spend Forecast by Product Type",
        "Product Profitability Trend",
        "Product Mix Distribution Over Time",
        "Contract Length Distribution",
        "Customer Segmentation",
        "Customer Cohort Analysis - Retention Rates",
        "Detailed Data View",
        "AI-driven Q&A Section"
    ]

    # Use streamlit-sortable to allow the user to reorder the widgets
    with st.sidebar:
        st.write("Drag and drop to rearrange sections:")
        widget_selection = st.multiselect("Select sections to display:", options=available_widgets, default=available_widgets)
        widget_order = st_sortable(widget_selection, key='sortable_list')

    # Display the widgets based on user-selected order
    for widget_name in widget_order:
        if widget_name == "Overview":
            # Overview section
            st.markdown("<h2 style='text-align: center; color: black;'>Overview</h2>",
                        unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Customers", int(filtered_data['total_customers'].sum()))
            col2.metric("Churn Percentage", f"{filtered_data['churn_percentage'].mean():.2%}")
            col3.metric("Avg Contract Length", f"{filtered_data['average_contract_length'].mean():.1f} months")
            col4.metric("Total ARR", f"${int(filtered_data['total_arr'].sum()):,}")
        elif widget_name == "Monthly Revenue Forecast":
            # Monthly Revenue Forecast Chart
            st.markdown("<h2 style='color: black;'>Monthly Revenue Forecast</h2>",
                        unsafe_allow_html=True)
            plot_revenue_forecast(filtered_data)
        elif widget_name == "Spend Forecast by Product Type":
            st.markdown("<h2 style='color: black;'>Spend Forecast by Product Type</h2>",
                        unsafe_allow_html=True)
            plot_spend_forecast(filtered_data)
        elif widget_name == "Product Profitability Trend":
            st.markdown("<h2 style='color: black;'>Product Profitability Trend</h2>",
                        unsafe_allow_html=True)
            plot_product_profitability(filtered_data, product_selection)
        elif widget_name == "Product Mix Distribution Over Time":
            st.markdown("<h3 style='color: black;'>Product Mix Distribution Over Time</h3>",
                        unsafe_allow_html=True)
            plot_product_mix(filtered_data, product_selection)
        elif widget_name == "Contract Length Distribution":
            st.markdown("<h3 style='color: black;'>Contract Length Distribution</h3>",
                        unsafe_allow_html=True)
            plot_contract_length_distribution(filtered_data)
        elif widget_name == "Customer Segmentation":
            st.markdown("<h3>Customer Segmentation</h3>", unsafe_allow_html=True)
            plot_customer_segmentation(customer_data)
        elif widget_name == "Customer Cohort Analysis - Retention Rates":
            st.markdown("<h3>Customer Cohort Analysis - Retention Rates</h3>", unsafe_allow_html=True)
            plot_cohort_retention_heatmap(cohort_data)
        elif widget_name == "Detailed Data View":
            display_data_table(filtered_data)
        elif widget_name == "AI-driven Q&A Section":
            st.markdown("<h2 style='text-align: center; color: black;'>AI-driven Q&A Section</h2>",
                        unsafe_allow_html=True)
            st.text("Cortex Analyst chat bot integration can be added here.")

# Run the main function when the script is executed
if __name__ == '__main__':
    main()
