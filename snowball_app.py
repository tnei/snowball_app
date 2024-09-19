import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from prophet import Prophet  # For advanced predictive analytics
import base64
import hashlib

# Set page configuration with custom theme and logo
st.set_page_config(
    page_title="SaaS Business Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for fonts and other styling
st.markdown("""
    <style>
        /* Import custom fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* Apply font styles */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        /* Customize header */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2E4057;
        }

        /* Customize subheaders */
        .subheader {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2E4057;
            margin-top: 2rem;
        }

        /* Customize metrics */
        .metric {
            font-size: 1.25rem;
            font-weight: 600;
        }

        /* Customize sidebar */
        .sidebar .sidebar-content {
            background-color: #F4F4F4;
        }

        /* Customize tooltips */
        .tooltip {
            font-size: 0.9rem;
        }

        /* Customize buttons */
        .stButton>button {
            background-color: #2E86AB;
            color: #FFFFFF;
            border-radius: 8px;
        }

        /* Dark mode styles */
        body.dark-mode {
            background-color: #2E4057;
            color: #FFFFFF;
        }
        .dark-mode .sidebar .sidebar-content {
            background-color: #3E4E67;
        }
    </style>
""", unsafe_allow_html=True)

# Function to load dummy data for the main dashboard
@st.cache_data
def load_dummy_data():
    # Generate dates for 36 months starting from January 2021
    dates = pd.date_range(start="2021-01-01", periods=36, freq='MS')
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
@st.cache_data
def generate_customer_data():
    np.random.seed(42)
    customer_data = {
        'customer_id': range(1, 1001),
        'spend': np.random.randint(500, 5000, 1000),
        'contract_length': np.random.randint(6, 48, 1000),
        'sign_up_date': pd.date_range(start="2018-01-01", periods=1000, freq='D'),
        'last_active': pd.date_range(start="2018-01-01", periods=1000, freq='D') +
                       pd.to_timedelta(np.random.randint(30, 365 * 3, 1000), unit='D')
    }
    return pd.DataFrame(customer_data)

# Function to generate dummy cohort data with retention rates
@st.cache_data
def generate_cohort_data():
    np.random.seed(42)
    # Generate sign-up dates in monthly cohorts
    cohort_months = pd.date_range(start="2018-01-01", periods=36, freq='MS')
    # List to store retention data for each cohort
    cohort_data = []
    for i, month in enumerate(cohort_months):
        cohort_size = np.random.randint(50, 100)  # Random cohort size
        retention = [cohort_size]  # Initial cohort size
        for m in range(1, 36):  # Generate retention for the next 35 months
            retention_rate = np.random.uniform(0.5, 0.95)  # Random retention rate
            retained_customers = int(retention[-1] * retention_rate)
            retention.append(retained_customers)
        cohort_data.append(retention)
    # Create DataFrame from the retention data
    cohort_df = pd.DataFrame(cohort_data, index=cohort_months.strftime('%Y-%m'))
    # Calculate retention percentage
    cohort_size = cohort_df.iloc[:, 0]
    retention_rate_df = cohort_df.divide(cohort_size, axis=0) * 100
    return retention_rate_df

# Function for advanced predictive analytics using Prophet
def forecast_revenue(df, periods):
    # Prepare data for Prophet
    prophet_df = df[['date', 'total_arr']].rename(columns={'date': 'ds', 'total_arr': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    # Plot forecast
    fig = plot_plotly(model, forecast)
    fig.update_layout(
        title="Revenue Forecast",
        title_x=0.5,
        title_font_size=20,
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    return forecast

# Plotting functions with enhanced interactivity and customization options
def plot_revenue_forecast(df):
    fig = px.line(df, x='date', y='total_arr', title='Monthly Revenue',
                  labels={'total_arr': 'Revenue ($)'})
    fig.update_traces(line_color='#2E86AB')
    # Add interactive range slider and selectors
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ]),
                font=dict(size=12)
            ),
            rangeslider=dict(visible=True),
            type="date",
            title='Date'
        ),
        yaxis=dict(title='Revenue ($)'),
        hovermode="x unified"
    )
    # Add annotations for data storytelling
    max_value = df['total_arr'].max()
    max_date = df[df['total_arr'] == max_value]['date'].iloc[0]
    fig.add_annotation(x=max_date, y=max_value,
                       text="Peak Revenue",
                       showarrow=True, arrowhead=1)
    st.plotly_chart(fig, use_container_width=True)

def plot_spend_forecast(df):
    fig = px.line(df, x='date', y='spend_forecast', title='Spend Forecast by Product Type',
                  labels={'spend_forecast': 'Spend ($)'})
    fig.update_traces(line_color='#FF6F61')
    # Add interactive range slider and selectors
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ]),
                font=dict(size=12)
            ),
            rangeslider=dict(visible=True),
            type="date",
            title='Date'
        ),
        yaxis=dict(title='Spend ($)'),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

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
        title_x=0.5,
        title_font_size=20,
        legend_title_text='Product',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ]),
                font=dict(size=12)
            ),
            rangeslider=dict(visible=True),
            type="date",
            title='Date'
        ),
        yaxis=dict(title='Revenue ($)'),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

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
        groupnorm=None,
        hover_data={'Revenue': ':.2f', 'date': True}
    )

    # Add interactive range slider and selectors
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        legend_title_text='Product',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ]),
                font=dict(size=12)
            ),
            rangeslider=dict(visible=True),
            type="date",
            title='Date'
        ),
        yaxis=dict(title='Revenue ($)'),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_contract_length_distribution(df):
    fig = px.histogram(df, x='average_contract_length', nbins=10,
                       title='Contract Length Distribution',
                       labels={'average_contract_length': 'Contract Length (Months)'})
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        xaxis=dict(title='Contract Length (Months)'),
        yaxis=dict(title='Count'),
        bargap=0.1
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_customer_segmentation(customer_data):
    X = customer_data[['spend', 'contract_length']]
    kmeans = KMeans(n_clusters=3, n_init='auto')
    customer_data['segment'] = kmeans.fit_predict(X)

    fig = px.scatter(customer_data, x='spend', y='contract_length', color='segment',
                     title='Customer Segmentation',
                     labels={'spend': 'Customer Spend ($)', 'contract_length': 'Contract Length (Months)'},
                     hover_data=['customer_id'])
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        legend_title_text='Segment',
        xaxis=dict(title='Customer Spend ($)'),
        yaxis=dict(title='Contract Length (Months)')
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_cohort_retention_heatmap(cohort_df):
    fig = px.imshow(cohort_df,
                    labels=dict(x="Cohort Period (Months)", y="Cohort (Sign-up Month)",
                                color="Retention Rate (%)"),
                    x=[f"Month {i}" for i in range(len(cohort_df.columns))],
                    y=cohort_df.index,
                    color_continuous_scale='Blues')
    fig.update_layout(
        title="Customer Cohort Analysis - Retention Rates",
        title_x=0.5,
        title_font_size=20,
        xaxis_title="Cohort Period (Months)",
        yaxis_title="Cohort (Sign-up Month)"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_data_table(df):
    st.markdown("<h2 class='subheader'>Detailed Data View</h2>", unsafe_allow_html=True)
    # Add filters
    with st.expander("Filter Data"):
        min_revenue = st.slider("Minimum Revenue ($)", int(df['total_arr'].min()), int(df['total_arr'].max()), int(df['total_arr'].min()))
        max_revenue = st.slider("Maximum Revenue ($)", int(df['total_arr'].min()), int(df['total_arr'].max()), int(df['total_arr'].max()))
        filtered_df = df[(df['total_arr'] >= min_revenue) & (df['total_arr'] <= max_revenue)]
    st.dataframe(filtered_df)

    # Add option to download data
    csv = filtered_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Function to add user authentication
def authenticate(username, password):
    # In a real application, replace this with a secure authentication mechanism
    # Here, we use a simple hardcoded username and password
    if username == "admin" and password == "password":
        return True
    else:
        return False

# Main function to run the Streamlit app
def main():
    # Add logo and header
    st.image("https://i.imgur.com/UbOXYAU.png", width=200)  # Replace with your logo URL or local path
    st.markdown("<h1 class='main-header'>SaaS Business Dashboard</h1>", unsafe_allow_html=True)

    # User authentication
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        with st.form("Login"):
            st.write("Please log in to access the dashboard.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if authenticate(username, password):
                    st.session_state['authenticated'] = True
                    st.success("Login successful!")
                else:
                    st.error("Invalid username or password.")
        return  # Stop execution until user logs in

    # Load data
    data = load_dummy_data()
    customer_data = generate_customer_data()
    cohort_data = generate_cohort_data()

    # Sidebar filters
    st.sidebar.header("Filter Options")
    start_month = st.sidebar.date_input("Start Date", value=data['date'].min(), help="Select the start date for data filtering.")
    end_month = st.sidebar.date_input("End Date", value=data['date'].max(), help="Select the end date for data filtering.")
    global product_selection
    product_selection = st.sidebar.multiselect("Select products:",
                                               ['Product 1', 'Product 2', 'Product 3'],
                                               default=['Product 1', 'Product 2', 'Product 3'],
                                               help="Choose which products to include in the analysis.")
    forecast_period = st.sidebar.slider("Forecast period (months):", 1, 12, value=6, help="Adjust the forecast period.")
    contract_length_range = st.sidebar.slider("Contract length (months):", 2, 48, value=(2, 48), help="Filter data by contract length range.")

    # Theme selection
    st.sidebar.header("Theme Options")
    theme_choice = st.sidebar.selectbox("Choose a theme:", ["Light", "Dark"])
    if theme_choice == "Dark":
        # Apply dark mode by adding a CSS class
        st.markdown('<body class="dark-mode">', unsafe_allow_html=True)

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

    # Dashboard Customization
    st.sidebar.header("Customize Dashboard")
    available_widgets = [
        "Overview",
        "Monthly Revenue",
        "Revenue Forecast",
        "Spend Forecast by Product Type",
        "Product Profitability Trend",
        "Product Mix Distribution Over Time",
        "Contract Length Distribution",
        "Customer Segmentation",
        "Customer Cohort Analysis - Retention Rates",
        "Detailed Data View",
        "AI-driven Q&A Section"
    ]

    widget_selection = []
    for widget in available_widgets:
        if st.sidebar.checkbox(f"{widget}", value=True):
            widget_selection.append(widget)

    # Display the widgets based on user-selected options
    for widget_name in widget_selection:
        if widget_name == "Overview":
            # Overview section
            st.markdown("<h2 class='subheader'>Overview</h2>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Customers", int(filtered_data['total_customers'].sum()), help="Sum of total customers over the selected period.")
            col2.metric("Churn Percentage", f"{filtered_data['churn_percentage'].mean():.2%}", help="Average churn percentage over the selected period.")
            col3.metric("Avg Contract Length", f"{filtered_data['average_contract_length'].mean():.1f} months", help="Average contract length in months.")
            col4.metric("Total ARR", f"${int(filtered_data['total_arr'].sum()):,}", help="Total Annual Recurring Revenue.")
        elif widget_name == "Monthly Revenue":
            # Monthly Revenue Chart
            st.markdown("<h2 class='subheader'>Monthly Revenue</h2>", unsafe_allow_html=True)
            plot_revenue_forecast(filtered_data)
        elif widget_name == "Revenue Forecast":
            st.markdown("<h2 class='subheader'>Revenue Forecast</h2>", unsafe_allow_html=True)
            forecast = forecast_revenue(data, periods=forecast_period)
        elif widget_name == "Spend Forecast by Product Type":
            st.markdown("<h2 class='subheader'>Spend Forecast by Product Type</h2>", unsafe_allow_html=True)
            plot_spend_forecast(filtered_data)
        elif widget_name == "Product Profitability Trend":
            st.markdown("<h2 class='subheader'>Product Profitability Trend</h2>", unsafe_allow_html=True)
            plot_product_profitability(filtered_data, product_selection)
        elif widget_name == "Product Mix Distribution Over Time":
            st.markdown("<h2 class='subheader'>Product Mix Distribution Over Time</h2>", unsafe_allow_html=True)
            plot_product_mix(filtered_data, product_selection)
        elif widget_name == "Contract Length Distribution":
            st.markdown("<h2 class='subheader'>Contract Length Distribution</h2>", unsafe_allow_html=True)
            plot_contract_length_distribution(filtered_data)
        elif widget_name == "Customer Segmentation":
            st.markdown("<h2 class='subheader'>Customer Segmentation</h2>", unsafe_allow_html=True)
            plot_customer_segmentation(customer_data)
        elif widget_name == "Customer Cohort Analysis - Retention Rates":
            st.markdown("<h2 class='subheader'>Customer Cohort Analysis - Retention Rates</h2>", unsafe_allow_html=True)
            plot_cohort_retention_heatmap(cohort_data)
        elif widget_name == "Detailed Data View":
            display_data_table(filtered_data)
        elif widget_name == "AI-driven Q&A Section":
            st.markdown("<h2 class='subheader'>AI-driven Q&A Section</h2>", unsafe_allow_html=True)
            st.text("Cortex Analyst chat bot integration can be added here.")

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9rem;'>© 2023 SaaS Business Dashboard. All rights reserved.</p>", unsafe_allow_html=True)

# Run the main function when the script is executed
if __name__ == '__main__':
    main()
