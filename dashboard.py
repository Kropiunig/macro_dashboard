import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os
import io
from datetime import datetime, timedelta
from fpdf import FPDF
import base64
from global_economic_overview import GlobalEconomicOverview

# Set up paths
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Initialize session state for view mode
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'Dashboard Tabs'

# Create a cache for data loading
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_economic_data(period='1y'):
    """Load economic data with caching"""
    geo = GlobalEconomicOverview()
    
    # Fetch data
    indices_data = geo.fetch_stock_indices_data(period=period)
    currency_data = geo.fetch_currency_data(period=period)
    commodity_data = geo.fetch_commodity_data(period=period)
    indicator_data = geo.fetch_economic_indicators()
    
    # Calculate metrics
    indices_metrics = geo.calculate_performance_metrics(indices_data)
    currency_metrics = geo.calculate_performance_metrics(currency_data)
    commodity_metrics = geo.calculate_performance_metrics(commodity_data)
    
    # Generate report
    report_text = geo.generate_summary_report(indices_metrics, currency_metrics, commodity_metrics, indicator_data)
    
    return {
        'indices_data': indices_data,
        'currency_data': currency_data,
        'commodity_data': commodity_data,
        'indicator_data': indicator_data,
        'indices_metrics': indices_metrics,
        'currency_metrics': currency_metrics,
        'commodity_metrics': commodity_metrics,
        'report_text': report_text
    }

# Custom CSS with theme support
def get_css():
    if st.session_state.theme == 'dark':
        return """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                color: #4DA6FF;
                text-align: center;
                margin-bottom: 1rem;
            }
            .sub-header {
                font-size: 1.5rem;
                font-weight: 600;
                color: #E0E0E0;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
            }
            .card {
                padding: 1.5rem;
                border-radius: 0.5rem;
                background-color: #2C3E50;
                box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.2);
                margin-bottom: 1rem;
                border: 1px solid #34495E;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: 700;
                color: #4DA6FF;
            }
            .metric-label {
                font-size: 1rem;
                color: #B0B0B0;
            }
            .positive {
                color: #00CC96;
            }
            .negative {
                color: #EF553B;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #2C3E50;
                border-radius: 4px 4px 0px 0px;
                padding: 10px 16px;
                border: 1px solid #34495E;
            }
            .stTabs [aria-selected="true"] {
                background-color: #4DA6FF !important;
                color: white !important;
            }
        </style>
        """
    else:
        return """
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: 700;
                color: #1E88E5;
                text-align: center;
                margin-bottom: 1rem;
            }
            .sub-header {
                font-size: 1.5rem;
                font-weight: 600;
                color: #333;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
            }
            .card {
                padding: 1.5rem;
                border-radius: 0.5rem;
                background-color: #f8f9fa;
                box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
                border: 1px solid #e9ecef;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: 700;
                color: #1E88E5;
            }
            .metric-label {
                font-size: 1rem;
                color: #666;
            }
            .positive {
                color: #4CAF50;
            }
            .negative {
                color: #F44336;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #f8f9fa;
                border-radius: 4px 4px 0px 0px;
                padding: 10px 16px;
                border: 1px solid #e9ecef;
            }
            .stTabs [aria-selected="true"] {
                background-color: #1E88E5 !important;
                color: white !important;
            }
        </style>
        """

# Apply CSS
st.markdown(get_css(), unsafe_allow_html=True)

# Helper functions
def format_change(value):
    """Format change values with color and sign"""
    if value > 0:
        return f'<span class="positive">+{value:.2f}%</span>'
    else:
        return f'<span class="negative">{value:.2f}%</span>'

def create_download_link(df, filename, text):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def run_dashboard():
    # Header
    st.markdown('<div class="main-header">Global Economic Dashboard</div>', unsafe_allow_html=True)
    
    # Current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    st.markdown(f"<div style='text-align: center; color: #666;'>Last updated: {current_date}</div>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Add view mode selection
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Dashboard Tabs", "Custom Chart Builder"],
        index=0,
        key="view_mode"
    )
    
    time_period = st.sidebar.selectbox("Select Time Period", ["1m", "3m", "6m", "1y", "2y", "5y"], index=3)
    
    # Theme toggle
    st.sidebar.selectbox("Select Theme", ["light", "dark"], key="theme", on_change=lambda: st.experimental_rerun())
    
    # Add date range selector
    today = datetime.now()
    default_start = today - timedelta(days=365)
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", today)
    if start_date > end_date:
        st.sidebar.error("End date must be after start date")
    
    # Data loading options
    load_option = st.sidebar.radio("Data Source", ["Use Cached Data", "Load Fresh Data"])
    
    # Load data
    if load_option == "Load Fresh Data":
        with st.spinner("Fetching latest economic data..."):
            # Load data with caching
            data = load_economic_data(period=time_period)
            indices_data = data['indices_data']
            currency_data = data['currency_data']
            commodity_data = data['commodity_data']
            indicator_data = data['indicator_data']
            indices_metrics = data['indices_metrics']
            currency_metrics = data['currency_metrics']
            commodity_metrics = data['commodity_metrics']
            report_text = data['report_text']
            
            # Generate visualizations
            geo = GlobalEconomicOverview()
            geo.plot_stock_indices(indices_data)
            geo.plot_economic_indicators(indicator_data)
            geo.plot_currencies(currency_data)
            geo.plot_commodities(commodity_data)
            geo.generate_summary_report(indices_metrics, currency_metrics, commodity_metrics, indicator_data)
            
            st.success("Data refreshed successfully!")
    
    # Check if output files exist
    if not os.path.exists(f"{output_dir}/global_economic_overview.md"):
        st.error("No data available. Please select 'Load Fresh Data' to generate the dashboard.")
        return
    
    # Load report data
    try:
        # Parse markdown report to extract data
        with open(f"{output_dir}/global_economic_overview.md", "r") as f:
            report_text = f.read()
        
        # Load data
        data = load_economic_data(period=time_period)
        indices_data = data['indices_data']
        currency_data = data['currency_data']
        commodity_data = data['commodity_data']
        indicator_data = data['indicator_data']
        
        # Display content based on view mode
        if st.session_state.view_mode == "Dashboard Tabs":
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Market Performance", "Economic Indicators", "Currencies & Commodities", "Money Supply", "Custom Analysis"])
            
            with tab1:
                st.markdown('<div class="sub-header">Dashboard Overview</div>', unsafe_allow_html=True)
                
                # Create overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # Extract key metrics from the report
                try:
                    # Stock market performance
                    stock_section = report_text.split("## Stock Market Overview")[1].split("##")[0] if "## Stock Market Overview" in report_text else ""
                    stock_lines = [line for line in stock_section.strip().split("\n") if line.strip()]
                    
                    # Get S&P 500 data
                    sp500_line = next((line for line in stock_lines if "S&P 500" in line), None)
                    if sp500_line:
                        try:
                            parts = sp500_line.replace("- **S&P 500**: ", "").split(" (YTD: ")
                            if len(parts) >= 2:
                                sp500_value = float(parts[0])
                                sp500_ytd = float(parts[1].split("%")[0])
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="card">
                                        <div class="metric-label">S&P 500</div>
                                        <div class="metric-value">{sp500_value:.2f}</div>
                                        <div>YTD: {format_change(sp500_ytd)}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        except (ValueError, IndexError) as e:
                            st.warning(f"Could not parse S&P 500 data: {e}")
                    
                    # Inflation rate
                    econ_section = report_text.split("## Economic Indicators")[1].split("###")[0] if "## Economic Indicators" in report_text else ""
                    econ_lines = [line for line in econ_section.strip().split("\n") if line.strip() and line.startswith("-")]
                    
                    inflation_line = next((line for line in econ_lines if "Inflation Rate" in line), None)
                    if inflation_line:
                        try:
                            parts = inflation_line.replace("- **US Inflation Rate**: ", "").split("%")
                            if len(parts) >= 1:
                                inflation_value = float(parts[0])
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="card">
                                        <div class="metric-label">US Inflation Rate</div>
                                        <div class="metric-value">{inflation_value:.1f}%</div>
                                        <div>YoY Change</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        except (ValueError, IndexError) as e:
                            st.warning(f"Could not parse inflation data: {e}")
                    
                    # Unemployment rate
                    unemployment_line = next((line for line in econ_lines if "Unemployment Rate" in line), None)
                    if unemployment_line:
                        try:
                            parts = unemployment_line.replace("- **US Unemployment Rate**: ", "").split("%")
                            if len(parts) >= 1:
                                unemployment_value = float(parts[0])
                                
                                with col3:
                                    st.markdown(f"""
                                    <div class="card">
                                        <div class="metric-label">US Unemployment</div>
                                        <div class="metric-value">{unemployment_value:.1f}%</div>
                                        <div>Current Rate</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        except (ValueError, IndexError) as e:
                            st.warning(f"Could not parse unemployment data: {e}")
                    
                    # Fed Funds Rate
                    fed_rate_line = next((line for line in econ_lines if "Federal Funds Rate" in line), None)
                    if fed_rate_line:
                        try:
                            parts = fed_rate_line.replace("- **US Federal Funds Rate**: ", "").split("%")
                            if len(parts) >= 1:
                                fed_rate_value = float(parts[0])
                                
                                with col4:
                                    st.markdown(f"""
                                    <div class="card">
                                        <div class="metric-label">Fed Funds Rate</div>
                                        <div class="metric-value">{fed_rate_value:.2f}%</div>
                                        <div>Current Rate</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        except (ValueError, IndexError) as e:
                            st.warning(f"Could not parse Fed Funds Rate data: {e}")
            
                except Exception as e:
                    st.warning(f"Could not parse all overview metrics: {e}")
            
                # Add a brief description
                st.markdown("""
                This dashboard provides a comprehensive overview of global economic indicators, 
                market performance, and monetary conditions. Use the controls in the sidebar to 
                adjust the time period and refresh the data.
                """)
            
                # Add download links for the report
                st.markdown("### Download Reports")
                col1, col2 = st.columns(2)
                with col1:
                    with open(f"{output_dir}/global_economic_overview.md", "r") as f:
                        md_content = f.read()
                        b64_md = base64.b64encode(md_content.encode()).decode()
                        st.markdown(f'<a href="data:file/markdown;base64,{b64_md}" download="global_economic_overview.md">Download Markdown Report</a>', unsafe_allow_html=True)
            
                with col2:
                    # Create a PDF download link (if available)
                    try:
                        from fpdf import FPDF
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 10, txt=md_content.replace('#', '').replace('*', ''))
                        pdf_output = pdf.output(dest="S").encode("latin1")
                        b64_pdf = base64.b64encode(pdf_output).decode()
                        st.markdown(f'<a href="data:application/pdf;base64,{b64_pdf}" download="global_economic_overview.pdf">Download PDF Report</a>', unsafe_allow_html=True)
                    except:
                        st.info("PDF download requires fpdf package. Install with: pip install fpdf")
        
        with tab2:
            st.markdown('<div class="sub-header">Market Performance</div>', unsafe_allow_html=True)
            
            try:
                # Load the data
                data = load_economic_data(period=time_period)
                indices_data = data['indices_data']
                
                # Debug information
                st.write(f"Number of indices: {len(indices_data)}")
                
                # Create a DataFrame to store normalized data for all indices
                all_indices_data = pd.DataFrame()
                
                # Process each index
                for name, df in indices_data.items():
                    if df is None:
                        st.warning(f"No data for {name} (None)")
                        continue
                    
                    if df.empty:
                        st.warning(f"No data for {name} (Empty DataFrame)")
                        continue
                    
                    if 'Close' not in df.columns:
                        st.warning(f"No 'Close' column for {name}. Columns: {df.columns.tolist()}")
                        continue
                    
                    # Make sure index is datetime
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    # Calculate normalized values
                    try:
                        normalized = df['Close'] / df['Close'].iloc[0] * 100
                        all_indices_data[name] = normalized
                    except Exception as e:
                        st.warning(f"Error normalizing {name}: {e}")
                
                # Display the data shape for debugging
                st.write(f"Processed data shape: {all_indices_data.shape}")
                
                # Create and display the chart if we have data
                if not all_indices_data.empty:
                    # Use Plotly Express for simpler plotting
                    fig = px.line(
                        all_indices_data,
                        title='Major Stock Indices Performance (Normalized)',
                        labels={'value': 'Normalized Value (Start = 100)', 'index': 'Date', 'variable': 'Index'},
                        template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                    )
                    
                    fig.update_layout(
                        height=500,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show a sample of the data for debugging
                    with st.expander("Show Data Sample"):
                        st.dataframe(all_indices_data.head())
                else:
                    st.error("No stock indices data available for plotting")
                    
                    # Display raw data for debugging
                    with st.expander("Debug Raw Data"):
                        for name, df in indices_data.items():
                            st.write(f"Index: {name}")
                            if df is not None and not df.empty:
                                st.dataframe(df.head())
                            else:
                                st.write("No data available")
                
                # Extract stock indices data from report for table display
                if "## Stock Indices Overview" in report_text:
                    stock_section = report_text.split("## Stock Indices Overview")[1].split("##")[0]
                    stock_lines = [line for line in stock_section.strip().split("\n") if line.strip()]
                    
                    # Create a DataFrame for the stock data
                    stock_data = []
                    for line in stock_lines:
                        try:
                            parts = line.replace("- **", "").split("**: ")
                            if len(parts) < 2:
                                continue
                            
                            name = parts[0]
                            values = parts[1].split(" (YTD: ")
                            if len(values) < 2:
                                continue
                            
                            current = float(values[0].replace(",", ""))
                            ytd = float(values[1].split("%")[0])
                            
                            stock_data.append({
                                "Index": name,
                                "Current Value": current,
                                "YTD Change (%)": ytd
                            })
                        except (ValueError, IndexError) as e:
                            st.warning(f"Error parsing stock line: {line} - {e}")
                            continue
                    
                    if stock_data:
                        stock_df = pd.DataFrame(stock_data)
                        st.dataframe(stock_df, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error loading market performance data: {e}")
                st.error(f"Details: {str(e)}")
                
                # Fall back to static image
                try:
                    stock_img = Image.open(f"{output_dir}/stock_indices_performance.png")
                    st.image(stock_img, use_container_width=True)
                except Exception as img_error:
                    st.error(f"Could not load stock indices image: {img_error}")
        
        with tab3:
            st.markdown('<div class="sub-header">Economic Indicators</div>', unsafe_allow_html=True)
            
            try:
                # Load the data for interactive charts
                data = load_economic_data(period=time_period)
                indicator_data = data['indicator_data']
                
                # Create interactive economic indicators chart
                # Create tabs for different economic indicators
                ec_tab1, ec_tab2, ec_tab3 = st.tabs(["GDP", "Inflation & Interest Rates", "Unemployment"])
                
                with ec_tab1:
                    # GDP Chart
                    if 'US GDP' in indicator_data:
                        gdp_data = indicator_data['US GDP']
                        fig = px.line(
                            gdp_data, 
                            x='Date', 
                            y='Value',
                            title='US GDP (Billions of $)',
                            labels={'Value': 'GDP (Billions of $)', 'Date': 'Date'},
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate growth rate if possible
                        if len(gdp_data) > 4:  # Need at least 5 points for 4 quarters
                            gdp_data['YoY_Growth'] = gdp_data['Value'].pct_change(4) * 100
                            latest_growth = gdp_data['YoY_Growth'].dropna().iloc[-1]
                            st.metric("Latest GDP Growth (YoY)", f"{latest_growth:.2f}%")
                
                with ec_tab2:
                    # Inflation and Interest Rates
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add inflation data
                    if 'US Inflation Rate' in indicator_data and 'YoY_Inflation' in indicator_data['US Inflation Rate'].columns:
                        inflation_data = indicator_data['US Inflation Rate'].dropna(subset=['YoY_Inflation'])
                        fig.add_trace(
                            go.Scatter(
                                x=inflation_data['Date'],
                                y=inflation_data['YoY_Inflation'],
                                name="Inflation Rate (YoY %)",
                                line=dict(color='red')
                            )
                        )
                    
                    # Add Fed Funds Rate
                    if 'US Federal Funds Rate' in indicator_data:
                        fed_data = indicator_data['US Federal Funds Rate']
                        fig.add_trace(
                            go.Scatter(
                                x=fed_data['Date'],
                                y=fed_data['Value'],
                                name="Federal Funds Rate (%)",
                                line=dict(color='blue')
                            )
                        )
                    
                    # Add 10-Year Treasury Yield
                    if 'US 10-Year Treasury Yield' in indicator_data:
                        treasury_data = indicator_data['US 10-Year Treasury Yield']
                        fig.add_trace(
                            go.Scatter(
                                x=treasury_data['Date'],
                                y=treasury_data['Value'],
                                name="10-Year Treasury Yield (%)",
                                line=dict(color='green')
                            ),
                            secondary_y=False
                        )
                    
                    fig.update_layout(
                        title='Inflation and Interest Rates',
                        xaxis_title='Date',
                        yaxis_title='Percent (%)',
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=500,
                        template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with ec_tab3:
                    # Unemployment Rate
                    if 'US Unemployment Rate' in indicator_data:
                        unemployment_data = indicator_data['US Unemployment Rate']
                        fig = px.line(
                            unemployment_data, 
                            x='Date', 
                            y='Value',
                            title='US Unemployment Rate (%)',
                            labels={'Value': 'Unemployment Rate (%)', 'Date': 'Date'},
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show latest value
                        latest_unemployment = unemployment_data['Value'].iloc[-1]
                        st.metric("Latest Unemployment Rate", f"{latest_unemployment:.1f}%")
            except Exception as e:
                st.error(f"Error loading economic indicators: {e}")
                # Fall back to static image
                try:
                    econ_img = Image.open(f"{output_dir}/economic_indicators.png")
                    st.image(econ_img, use_container_width=True)
                except:
                    st.error("Could not load economic indicators image")
        
        with tab4:
            st.markdown('<div class="sub-header">Currencies & Commodities</div>', unsafe_allow_html=True)
            
            try:
                # Load the data
                data = load_economic_data(period=time_period)
                currency_data = data['currency_data']
                commodity_data = data['commodity_data']
                
                # Create tabs for currencies and commodities
                cc_tab1, cc_tab2 = st.tabs(["Currencies", "Commodities"])
                
                with cc_tab1:
                    # Debug information
                    st.write(f"Number of currencies: {len(currency_data)}")
                    for name, df in currency_data.items():
                        if df is None or df.empty:
                            st.warning(f"No data for {name}")
                        elif 'Close' not in df.columns:
                            st.warning(f"No 'Close' column for {name}. Columns: {df.columns.tolist()}")
                    
                    # Create a fresh DataFrame with the currency data
                    currency_plot_data = pd.DataFrame()
                    
                    # Process each currency
                    for name, df in currency_data.items():
                        if df is not None and not df.empty and 'Close' in df.columns:
                            # Normalize the data
                            normalized = df['Close'] / df['Close'].iloc[0] * 100
                            currency_plot_data[name] = normalized
                    
                    # Create a datetime index if needed
                    if not currency_plot_data.empty and not isinstance(currency_plot_data.index, pd.DatetimeIndex):
                        currency_plot_data.index = pd.to_datetime(df.index)
                    
                    # Create and display the chart if we have data
                    if not currency_plot_data.empty:
                        fig = px.line(
                            currency_plot_data,
                            title='Currency Performance vs USD (Normalized)',
                            labels={'value': 'Normalized Value (Start = 100)', 'index': 'Date', 'variable': 'Currency'},
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        fig.update_layout(
                            height=500,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No currency data available for plotting")
                    
                    # Extract currency data from report for table display
                    if "## Currency Overview" in report_text:
                        currency_section = report_text.split("## Currency Overview")[1].split("##")[0]
                        currency_lines = [line for line in currency_section.strip().split("\n") if line.strip()]
                        
                        # Create a DataFrame for the currency data
                        currency_data_table = []
                        for line in currency_lines:
                            try:
                                parts = line.replace("- **", "").split("**: ")
                                if len(parts) < 2:
                                    continue
                                
                                name = parts[0]
                                values = parts[1].split(" (YTD: ")
                                if len(values) < 2:
                                    continue
                                
                                current = float(values[0])
                                ytd = float(values[1].split("%")[0])
                                
                                currency_data_table.append({
                                    "Currency": name,
                                    "Current Rate": current,
                                    "YTD Change (%)": ytd
                                })
                            except (ValueError, IndexError) as e:
                                st.warning(f"Error parsing currency line: {line} - {e}")
                                continue
                        
                        if currency_data_table:
                            currency_df = pd.DataFrame(currency_data_table)
                            st.dataframe(currency_df, use_container_width=True)
                
                with cc_tab2:
                    # Debug information
                    st.write(f"Number of commodities: {len(commodity_data)}")
                    for name, df in commodity_data.items():
                        if df is None or df.empty:
                            st.warning(f"No data for {name}")
                        elif 'Close' not in df.columns:
                            st.warning(f"No 'Close' column for {name}. Columns: {df.columns.tolist()}")
                    
                    # Create a fresh DataFrame with the commodity data
                    commodity_plot_data = pd.DataFrame()
                    
                    # Process each commodity
                    for name, df in commodity_data.items():
                        if df is not None and not df.empty and 'Close' in df.columns:
                            # Normalize the data
                            normalized = df['Close'] / df['Close'].iloc[0] * 100
                            commodity_plot_data[name] = normalized
                    
                    # Create a datetime index if needed
                    if not commodity_plot_data.empty and not isinstance(commodity_plot_data.index, pd.DatetimeIndex):
                        commodity_plot_data.index = pd.to_datetime(df.index)
                    
                    # Create and display the chart if we have data
                    if not commodity_plot_data.empty:
                        fig = px.line(
                            commodity_plot_data,
                            title='Commodity Performance (Normalized)',
                            labels={'value': 'Normalized Value (Start = 100)', 'index': 'Date', 'variable': 'Commodity'},
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        fig.update_layout(
                            height=500,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No commodity data available for plotting")
                    
                    # Extract commodity data from report for table display
                    if "## Commodity Overview" in report_text:
                        commodity_section = report_text.split("## Commodity Overview")[1].split("##")[0]
                        commodity_lines = [line for line in commodity_section.strip().split("\n") if line.strip()]
                        
                        # Create a DataFrame for the commodity data
                        commodity_data_table = []
                        for line in commodity_lines:
                            try:
                                parts = line.replace("- **", "").split("**: $")
                                if len(parts) < 2:
                                    continue
                                
                                name = parts[0]
                                values = parts[1].split(" (YTD: ")
                                if len(values) < 2:
                                    continue
                                
                                current = float(values[0])
                                ytd = float(values[1].split("%")[0])
                                
                                commodity_data_table.append({
                                    "Commodity": name,
                                    "Current Price ($)": current,
                                    "YTD Change (%)": ytd
                                })
                            except (ValueError, IndexError) as e:
                                st.warning(f"Error parsing commodity line: {line} - {e}")
                                continue
                        
                        if commodity_data_table:
                            commodity_df = pd.DataFrame(commodity_data_table)
                            st.dataframe(commodity_df, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error loading currency and commodity data: {e}")
                st.error(f"Details: {str(e)}")
                
                # Fall back to static images
                try:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="sub-header">Currency Performance</div>', unsafe_allow_html=True)
                        currency_img = Image.open(f"{output_dir}/currency_performance.png")
                        st.image(currency_img, use_container_width=True)
                    
                    with col2:
                        st.markdown('<div class="sub-header">Commodity Performance</div>', unsafe_allow_html=True)
                        commodity_img = Image.open(f"{output_dir}/commodity_performance.png")
                        st.image(commodity_img, use_container_width=True)
                except Exception as img_error:
                    st.error(f"Could not load currency and commodity images: {img_error}")
        
        with tab5:
            st.markdown('<div class="sub-header">Money Supply</div>', unsafe_allow_html=True)
            
            try:
                # Load the data
                data = load_economic_data(period=time_period)
                indicator_data = data['indicator_data']
                
                # Create tabs for different money supply metrics
                ms_tab1, ms_tab2 = st.tabs(["US Money Supply", "Global Money Supply Growth"])
                
                with ms_tab1:
                    # US Money Supply Chart (M1 and M2)
                    fig = go.Figure()
                    
                    # Add M1 Money Supply
                    if 'US M1 Money Supply' in indicator_data:
                        m1_data = indicator_data['US M1 Money Supply']
                        fig.add_trace(go.Scatter(
                            x=m1_data['Date'],
                            y=m1_data['Value'] / 1000,  # Convert to trillions
                            mode='lines',
                            name='M1',
                            line=dict(color='blue')
                        ))
                    
                    # Add M2 Money Supply
                    if 'US M2 Money Supply' in indicator_data:
                        m2_data = indicator_data['US M2 Money Supply']
                        fig.add_trace(go.Scatter(
                            x=m2_data['Date'],
                            y=m2_data['Value'] / 1000,  # Convert to trillions
                            mode='lines',
                            name='M2',
                            line=dict(color='red')
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title='US Money Supply (Trillions of $)',
                        xaxis_title='Date',
                        yaxis_title='Trillions of $',
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=500,
                        template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show latest values
                    col1, col2 = st.columns(2)
                    
                    if 'US M1 Money Supply' in indicator_data:
                        m1_data = indicator_data['US M1 Money Supply']
                        latest_m1 = m1_data['Value'].iloc[-1] / 1000  # Convert to trillions
                        with col1:
                            st.metric("Latest M1 Money Supply", f"${latest_m1:.2f}T")
                    
                    if 'US M2 Money Supply' in indicator_data:
                        m2_data = indicator_data['US M2 Money Supply']
                        latest_m2 = m2_data['Value'].iloc[-1] / 1000  # Convert to trillions
                        with col2:
                            st.metric("Latest M2 Money Supply", f"${latest_m2:.2f}T")
                
                with ms_tab2:
                    # Global Money Supply Growth Chart
                    fig = go.Figure()
                    
                    # Add money supply growth rates for different regions
                    money_supply_data = []
                    
                    # US M2 Money Supply Growth
                    if 'US M2 Money Supply' in indicator_data and 'YoY_Growth' in indicator_data['US M2 Money Supply'].columns:
                        ms_data = indicator_data['US M2 Money Supply']
                        valid_data = ms_data.dropna(subset=['YoY_Growth'])
                        if not valid_data.empty:
                            fig.add_trace(go.Scatter(
                                x=valid_data['Date'],
                                y=valid_data['YoY_Growth'],
                                mode='lines',
                                name='US M2',
                                line=dict(color='blue')
                            ))
                            money_supply_data.append(("US M2", valid_data['YoY_Growth'].iloc[-1]))
                    
                    # Euro Area M3 Money Supply Growth
                    if 'Euro Area M3 Money Supply' in indicator_data and 'YoY_Growth' in indicator_data['Euro Area M3 Money Supply'].columns:
                        ms_data = indicator_data['Euro Area M3 Money Supply']
                        valid_data = ms_data.dropna(subset=['YoY_Growth'])
                        if not valid_data.empty:
                            fig.add_trace(go.Scatter(
                                x=valid_data['Date'],
                                y=valid_data['YoY_Growth'],
                                mode='lines',
                                name='Euro Area M3',
                                line=dict(color='red')
                            ))
                            money_supply_data.append(("Euro Area M3", valid_data['YoY_Growth'].iloc[-1]))
                    
                    # China M2 Money Supply Growth
                    if 'China M2 Money Supply' in indicator_data and 'YoY_Growth' in indicator_data['China M2 Money Supply'].columns:
                        ms_data = indicator_data['China M2 Money Supply']
                        valid_data = ms_data.dropna(subset=['YoY_Growth'])
                        if not valid_data.empty:
                            fig.add_trace(go.Scatter(
                                x=valid_data['Date'],
                                y=valid_data['YoY_Growth'],
                                mode='lines',
                                name='China M2',
                                line=dict(color='green')
                            ))
                            money_supply_data.append(("China M2", valid_data['YoY_Growth'].iloc[-1]))
                    
                    # Japan M2 Money Supply Growth
                    if 'Japan M2 Money Supply' in indicator_data and 'YoY_Growth' in indicator_data['Japan M2 Money Supply'].columns:
                        ms_data = indicator_data['Japan M2 Money Supply']
                        valid_data = ms_data.dropna(subset=['YoY_Growth'])
                        if not valid_data.empty:
                            fig.add_trace(go.Scatter(
                                x=valid_data['Date'],
                                y=valid_data['YoY_Growth'],
                                mode='lines',
                                name='Japan M2',
                                line=dict(color='purple')
                            ))
                            money_supply_data.append(("Japan M2", valid_data['YoY_Growth'].iloc[-1]))
                    
                    # Update layout
                    fig.update_layout(
                        title='Global Money Supply Growth (YoY %)',
                        xaxis_title='Date',
                        yaxis_title='Year-over-Year Growth (%)',
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=500,
                        template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show latest values
                    if money_supply_data:
                        cols = st.columns(len(money_supply_data))
                        for i, (name, value) in enumerate(money_supply_data):
                            with cols[i]:
                                st.metric(f"{name} Growth", f"{value:.2f}%")
            
            except Exception as e:
                st.error(f"Error loading money supply data: {e}")
                # Fall back to static data display
                try:
                    # Extract money supply data from report
                    money_section = report_text.split("### Global Money Supply")[1].split("##")[0] if "### Global Money Supply" in report_text else ""
                    money_lines = [line for line in money_section.strip().split("\n") if line.strip()]
                    
                    # Display money supply data in columns
                    cols = st.columns(len(money_lines) if len(money_lines) <= 3 else 3)
                    for i, line in enumerate(money_lines):
                        try:
                            parts = line.replace("- **", "").split("**: ")
                            if len(parts) < 2:
                                continue
                            
                            name = parts[0]
                            value = parts[1]
                            
                            with cols[i % 3]:
                                st.markdown(f"""
                                <div class="card">
                                    <div class="metric-label">{name}</div>
                                    <div class="metric-value">{value.split(' (YoY')[0]}</div>
                                    <div>{'(YoY ' + value.split('(YoY ')[1] if '(YoY ' in value else ''}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            continue
                except:
                    st.error("Could not load money supply data")
    
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        st.info("Try clicking 'Refresh Data' to regenerate the dashboard.")
        
    # Custom Chart Builder view mode
    if st.session_state.view_mode == "Custom Chart Builder":
        st.markdown('<div class="sub-header">Custom Chart Builder</div>', unsafe_allow_html=True)
        
        try:
            # Debug information
            st.write(f"Data loaded: indices={len(indices_data)}, currencies={len(currency_data)}, commodities={len(commodity_data)}, indicators={len(indicator_data)}")
            
            # Prepare data for selection
            indicator_options = []
            
            # Add stock indices
            for name in indices_data:
                indicator_options.append(f"Stock Index: {name}")
            
            # Add currencies
            for name in currency_data:
                indicator_options.append(f"Currency: {name}")
            
            # Add commodities
            for name in commodity_data:
                indicator_options.append(f"Commodity: {name}")
            
            # Add economic indicators
            for name in indicator_data:
                indicator_options.append(f"Indicator: {name}")
            
            # Create selection interface
            st.markdown("### Select Indicators to Compare")
            st.write("Choose multiple indicators to overlay on a single chart.")
            
            # Create multiselect for indicators
            selected_indicators = st.multiselect(
                "Select indicators to compare",
                options=indicator_options,
                default=indicator_options[:2] if len(indicator_options) >= 2 else indicator_options
            )
            
            # Normalization option
            normalization = st.radio(
                "Normalization Method",
                options=["Normalize to 100", "Raw Values"],
                index=0
            )
            
            # Create chart if indicators are selected
            if selected_indicators:
                st.write(f"Selected indicators: {selected_indicators}")
                
                # Create a DataFrame to store the data for plotting
                plot_data = pd.DataFrame()
                
                # Process each selected indicator
                for indicator in selected_indicators:
                    try:
                        # Parse indicator type and name
                        indicator_type, indicator_name = indicator.split(": ", 1)
                        
                        if indicator_type == "Stock Index":
                            if indicator_name in indices_data:
                                df = indices_data[indicator_name]
                                if df is not None and not df.empty and 'Close' in df.columns:
                                    # Make sure index is datetime
                                    if not isinstance(df.index, pd.DatetimeIndex):
                                        df.index = pd.to_datetime(df.index)
                                    
                                    # Apply normalization
                                    if normalization == "Normalize to 100":
                                        values = df['Close'] / df['Close'].iloc[0] * 100
                                    else:
                                        values = df['Close']
                                    
                                    # Add to plot data
                                    plot_data[indicator] = values
                        
                        elif indicator_type == "Currency":
                            if indicator_name in currency_data:
                                df = currency_data[indicator_name]
                                if df is not None and not df.empty and 'Close' in df.columns:
                                    # Make sure index is datetime
                                    if not isinstance(df.index, pd.DatetimeIndex):
                                        df.index = pd.to_datetime(df.index)
                                    
                                    # Apply normalization
                                    if normalization == "Normalize to 100":
                                        values = df['Close'] / df['Close'].iloc[0] * 100
                                    else:
                                        values = df['Close']
                                    
                                    # Add to plot data
                                    plot_data[indicator] = values
                        
                        elif indicator_type == "Commodity":
                            if indicator_name in commodity_data:
                                df = commodity_data[indicator_name]
                                if df is not None and not df.empty and 'Close' in df.columns:
                                    # Make sure index is datetime
                                    if not isinstance(df.index, pd.DatetimeIndex):
                                        df.index = pd.to_datetime(df.index)
                                    
                                    # Apply normalization
                                    if normalization == "Normalize to 100":
                                        values = df['Close'] / df['Close'].iloc[0] * 100
                                    else:
                                        values = df['Close']
                                    
                                    # Add to plot data
                                    plot_data[indicator] = values
                        
                        elif indicator_type == "Indicator":
                            if indicator_name in indicator_data:
                                df = indicator_data[indicator_name]
                                if df is not None and not df.empty and 'Value' in df.columns:
                                    # Apply normalization
                                    if normalization == "Normalize to 100":
                                        values = df['Value'] / df['Value'].iloc[0] * 100
                                    else:
                                        values = df['Value']
                                    
                                    # Add to plot data with date as index
                                    temp_df = pd.DataFrame({
                                        'Date': df['Date'],
                                        indicator: values
                                    })
                                    temp_df.set_index('Date', inplace=True)
                                    
                                    # Merge with plot data
                                    if plot_data.empty:
                                        plot_data = temp_df
                                    else:
                                        plot_data = plot_data.join(temp_df, how='outer')
                    
                    except Exception as e:
                        st.warning(f"Error processing {indicator}: {e}")
                
                # Display the data shape for debugging
                st.write(f"Plot data shape: {plot_data.shape}")
                
                # Check if we have data to plot
                if not plot_data.empty:
                    # Display a sample of the data
                    with st.expander("Show Data Sample"):
                        st.dataframe(plot_data.head())
                    
                    # Add option for dual y-axis
                    use_dual_axis = st.checkbox("Use Dual Y-Axis (for indicators with different scales)", value=True)
                    
                    if use_dual_axis and len(selected_indicators) > 1:
                        # Let user select which indicators go on which axis
                        col1, col2 = st.columns(2)
                        with col1:
                            left_axis = st.multiselect(
                                "Left Y-Axis Indicators",
                                options=selected_indicators,
                                default=[selected_indicators[0]] if selected_indicators else []
                            )
                        with col2:
                            right_axis = st.multiselect(
                                "Right Y-Axis Indicators",
                                options=[i for i in selected_indicators if i not in left_axis],
                                default=[i for i in selected_indicators if i not in left_axis and i != selected_indicators[0]][:1]
                            )
                        
                        # Create figure with secondary y-axis
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add traces for left y-axis
                        for indicator in left_axis:
                            if indicator in plot_data.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=plot_data.index,
                                        y=plot_data[indicator],
                                        name=indicator,
                                        line=dict(width=2)
                                    ),
                                    secondary_y=False
                                )
                        
                        # Add traces for right y-axis
                        for indicator in right_axis:
                            if indicator in plot_data.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=plot_data.index,
                                        y=plot_data[indicator],
                                        name=indicator,
                                        line=dict(width=2, dash='dash')
                                    ),
                                    secondary_y=True
                                )
                        
                        # Set titles
                        fig.update_layout(
                            title='Custom Indicator Comparison (Dual Y-Axis)',
                            xaxis_title='Date',
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            height=600,
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        
                        # Set y-axes titles
                        left_title = "Value" if normalization == "Raw Values" else "Normalized Value (Start = 100)"
                        right_title = "Value" if normalization == "Raw Values" else "Normalized Value (Start = 100)"
                        
                        if left_axis:
                            fig.update_yaxes(title_text=f"{left_title} - Left Axis", secondary_y=False)
                        if right_axis:
                            fig.update_yaxes(title_text=f"{right_title} - Right Axis", secondary_y=True)
                    else:
                        # Create and display the chart with a single y-axis
                        fig = px.line(
                            plot_data,
                            title='Custom Indicator Comparison',
                            labels={'value': 'Value', 'index': 'Date', 'variable': 'Indicator'},
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        
                        fig.update_layout(
                            height=600,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            yaxis_title="Normalized Value (Start = 100)" if normalization == "Normalize to 100" else "Value"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add correlation analysis
                    if len(selected_indicators) > 1 and st.checkbox("Show Correlation Analysis"):
                        st.markdown("### Correlation Analysis")
                        
                        # Calculate correlation matrix
                        correlation_matrix = plot_data.corr()
                        
                        # Create heatmap
                        fig_corr = px.imshow(
                            correlation_matrix,
                            text_auto=True,
                            color_continuous_scale='RdBu_r',
                            title='Correlation Matrix',
                            labels=dict(color="Correlation")
                        )
                        
                        fig_corr.update_layout(
                            height=500,
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Interpretation
                        st.markdown("""
                        **Correlation Interpretation:**
                        - Values close to 1: Strong positive correlation (indicators move together)
                        - Values close to -1: Strong negative correlation (indicators move in opposite directions)
                        - Values close to 0: No correlation (indicators move independently)
                        """)
                else:
                    st.error("No data available for the selected indicators")
            else:
                st.info("Please select at least one indicator to display")
        
        except Exception as e:
            st.error(f"Error in Custom Analysis: {e}")
            st.error(f"Details: {str(e)}")
            
            # Show traceback for debugging
            import traceback
            st.code(traceback.format_exc(), language="python")
            
    try:
        with tab6:
            st.markdown('<div class="sub-header">Custom Analysis</div>', unsafe_allow_html=True)
            
            try:
                # Debug information
                st.write(f"Data loaded: indices={len(indices_data)}, currencies={len(currency_data)}, commodities={len(commodity_data)}, indicators={len(indicator_data)}")
                
                # Prepare data for selection
                indicator_options = []
                
                # Add stock indices
                for name in indices_data:
                    indicator_options.append(f"Stock Index: {name}")
                
                # Add currencies
                for name in currency_data:
                    indicator_options.append(f"Currency: {name}")
                
                # Add commodities
                for name in commodity_data:
                    indicator_options.append(f"Commodity: {name}")
                
                # Add economic indicators
                for name in indicator_data:
                    indicator_options.append(f"Indicator: {name}")
                
                # Create selection interface
                st.markdown("### Select Indicators to Compare")
                st.write("Choose multiple indicators to overlay on a single chart.")
                
                # Create multiselect for indicators
                selected_indicators = st.multiselect(
                    "Select indicators to compare",
                    options=indicator_options,
                    default=indicator_options[:2] if len(indicator_options) >= 2 else indicator_options
                )
                
                # Normalization option
                normalization = st.radio(
                    "Normalization Method",
                    options=["Normalize to 100", "Raw Values"],
                    index=0
                )
                
                # Create chart if indicators are selected
                if selected_indicators:
                    st.write(f"Selected indicators: {selected_indicators}")
                    
                    # Create a DataFrame to store the data for plotting
                    plot_data = pd.DataFrame()
                    
                    # Process each selected indicator
                    for indicator in selected_indicators:
                        try:
                            # Parse indicator type and name
                            indicator_type, indicator_name = indicator.split(": ", 1)
                            
                            if indicator_type == "Stock Index":
                                if indicator_name in indices_data:
                                    df = indices_data[indicator_name]
                                    if df is not None and not df.empty and 'Close' in df.columns:
                                        # Make sure index is datetime
                                        if not isinstance(df.index, pd.DatetimeIndex):
                                            df.index = pd.to_datetime(df.index)
                                        
                                        # Apply normalization
                                        if normalization == "Normalize to 100":
                                            values = df['Close'] / df['Close'].iloc[0] * 100
                                        else:
                                            values = df['Close']
                                        
                                        # Add to plot data
                                        plot_data[indicator] = values
                        
                            elif indicator_type == "Currency":
                                if indicator_name in currency_data:
                                    df = currency_data[indicator_name]
                                    if df is not None and not df.empty and 'Close' in df.columns:
                                        # Make sure index is datetime
                                        if not isinstance(df.index, pd.DatetimeIndex):
                                            df.index = pd.to_datetime(df.index)
                                        
                                        # Apply normalization
                                        if normalization == "Normalize to 100":
                                            values = df['Close'] / df['Close'].iloc[0] * 100
                                        else:
                                            values = df['Close']
                                        
                                        # Add to plot data
                                        plot_data[indicator] = values
                        
                            elif indicator_type == "Commodity":
                                if indicator_name in commodity_data:
                                    df = commodity_data[indicator_name]
                                    if df is not None and not df.empty and 'Close' in df.columns:
                                        # Make sure index is datetime
                                        if not isinstance(df.index, pd.DatetimeIndex):
                                            df.index = pd.to_datetime(df.index)
                                        
                                        # Apply normalization
                                        if normalization == "Normalize to 100":
                                            values = df['Close'] / df['Close'].iloc[0] * 100
                                        else:
                                            values = df['Close']
                                        
                                        # Add to plot data
                                        plot_data[indicator] = values
                        
                            elif indicator_type == "Indicator":
                                if indicator_name in indicator_data:
                                    df = indicator_data[indicator_name]
                                    if df is not None and not df.empty and 'Value' in df.columns:
                                        # Apply normalization
                                        if normalization == "Normalize to 100":
                                            values = df['Value'] / df['Value'].iloc[0] * 100
                                        else:
                                            values = df['Value']
                                        
                                        # Add to plot data with date as index
                                        temp_df = pd.DataFrame({
                                            'Date': df['Date'],
                                            indicator: values
                                        })
                                        temp_df.set_index('Date', inplace=True)
                                        
                                        # Merge with plot data
                                        if plot_data.empty:
                                            plot_data = temp_df
                                        else:
                                            plot_data = plot_data.join(temp_df, how='outer')
                    
                        except Exception as e:
                            st.warning(f"Error processing {indicator}: {e}")
                
                # Display the data shape for debugging
                st.write(f"Plot data shape: {plot_data.shape}")
                
                # Check if we have data to plot
                if not plot_data.empty:
                    # Display a sample of the data
                    with st.expander("Show Data Sample"):
                        st.dataframe(plot_data.head())
                    
                    # Add option for dual y-axis
                    use_dual_axis = st.checkbox("Use Dual Y-Axis (for indicators with different scales)", value=True)
                    
                    if use_dual_axis and len(selected_indicators) > 1:
                        # Let user select which indicators go on which axis
                        col1, col2 = st.columns(2)
                        with col1:
                            left_axis = st.multiselect(
                                "Left Y-Axis Indicators",
                                options=selected_indicators,
                                default=[selected_indicators[0]] if selected_indicators else []
                            )
                        with col2:
                            right_axis = st.multiselect(
                                "Right Y-Axis Indicators",
                                options=[i for i in selected_indicators if i not in left_axis],
                                default=[i for i in selected_indicators if i not in left_axis and i != selected_indicators[0]][:1]
                            )
                        
                        # Create figure with secondary y-axis
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add traces for left y-axis
                        for indicator in left_axis:
                            if indicator in plot_data.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=plot_data.index,
                                        y=plot_data[indicator],
                                        name=indicator,
                                        line=dict(width=2)
                                    ),
                                    secondary_y=False
                                )
                        
                        # Add traces for right y-axis
                        for indicator in right_axis:
                            if indicator in plot_data.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=plot_data.index,
                                        y=plot_data[indicator],
                                        name=indicator,
                                        line=dict(width=2, dash='dash')
                                    ),
                                    secondary_y=True
                                )
                        
                        # Set titles
                        fig.update_layout(
                            title='Custom Indicator Comparison (Dual Y-Axis)',
                            xaxis_title='Date',
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            height=600,
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        
                        # Set y-axes titles
                        left_title = "Value" if normalization == "Raw Values" else "Normalized Value (Start = 100)"
                        right_title = "Value" if normalization == "Raw Values" else "Normalized Value (Start = 100)"
                        
                        if left_axis:
                            fig.update_yaxes(title_text=f"{left_title} - Left Axis", secondary_y=False)
                        if right_axis:
                            fig.update_yaxes(title_text=f"{right_title} - Right Axis", secondary_y=True)
                    else:
                        # Create and display the chart with a single y-axis
                        fig = px.line(
                            plot_data,
                            title='Custom Indicator Comparison',
                            labels={'value': 'Value', 'index': 'Date', 'variable': 'Indicator'},
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        
                        fig.update_layout(
                            height=600,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            yaxis_title="Normalized Value (Start = 100)" if normalization == "Normalize to 100" else "Value"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add correlation analysis
                    if len(selected_indicators) > 1 and st.checkbox("Show Correlation Analysis"):
                        st.markdown("### Correlation Analysis")
                        
                        # Calculate correlation matrix
                        correlation_matrix = plot_data.corr()
                        
                        # Create heatmap
                        fig_corr = px.imshow(
                            correlation_matrix,
                            text_auto=True,
                            color_continuous_scale='RdBu_r',
                            title='Correlation Matrix',
                            labels=dict(color="Correlation")
                        )
                        
                        fig_corr.update_layout(
                            height=500,
                            template='plotly_white' if st.session_state.theme == 'light' else 'plotly_dark'
                        )
                        
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Interpretation
                        st.markdown("""
                        **Correlation Interpretation:**
                        - Values close to 1: Strong positive correlation (indicators move together)
                        - Values close to -1: Strong negative correlation (indicators move in opposite directions)
                        - Values close to 0: No correlation (indicators move independently)
                        """)
                else:
                    st.error("No data available for the selected indicators")
            except Exception as e:
                st.error(f"Error in Custom Analysis: {e}")
                st.error(f"Details: {str(e)}")
                
                # Show traceback for debugging
                import traceback
                st.code(traceback.format_exc(), language="python")
    except Exception as e:
        st.error(f"Error in Custom Analysis Tab: {e}")
        st.error(f"Details: {str(e)}")
        
        # Show traceback for debugging
        import traceback
        st.code(traceback.format_exc(), language="python")

if __name__ == "__main__":
    run_dashboard()
