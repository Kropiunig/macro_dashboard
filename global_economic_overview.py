import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import io
import yfinance as yf
from matplotlib.gridspec import GridSpec
import seaborn as sns
from fredapi import Fred
import os
from dotenv import load_dotenv

# Load environment variables if available
load_dotenv()

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

class GlobalEconomicOverview:
    def __init__(self):
        """Initialize the GlobalEconomicOverview class with API keys and data sources"""
        # Try to get FRED API key from environment variable
        self.fred_api_key = os.getenv('FRED_API_KEY', None)
        if self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            print("Warning: FRED API key not found. Some data may not be available.")
            self.fred = None
        
        # Define major stock indices to track
        self.stock_indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'FTSE 100': '^FTSE',
            'DAX': '^GDAXI',
            'Nikkei 225': '^N225',
            'Shanghai Composite': '000001.SS',
            'Hang Seng': '^HSI'
        }
        
        # Define major currencies to track against USD
        self.currencies = {
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'JPY/USD': 'JPYUSD=X',
            'CNY/USD': 'CNYUSD=X',
            'CAD/USD': 'CADUSD=X'
        }
        
        # Define economic indicators to track
        self.indicators = {
            'US GDP': 'GDP',
            'US Unemployment Rate': 'UNRATE',
            'US Inflation Rate': 'CPIAUCSL',
            'US Federal Funds Rate': 'FEDFUNDS',
            'US 10-Year Treasury Yield': 'GS10',
            'Global GDP Growth': 'GDPCA',
            'US Consumer Sentiment': 'UMCSENT',
            # Add money supply indicators
            'US M1 Money Supply': 'M1SL',
            'US M2 Money Supply': 'M2SL',
            'Euro Area M3 Money Supply': 'MYAGM3EZM196N',
            'China M2 Money Supply': 'MYAGM2CNM189N',  
            'Japan M2 Money Supply': 'MANMM101JPM189S'
        }
        
        # Define commodities to track
        self.commodities = {
            'Crude Oil': 'CL=F',
            'Gold': 'GC=F',
            'Silver': 'SI=F',
            'Natural Gas': 'NG=F',
            'Copper': 'HG=F'
        }
        
    def fetch_stock_indices_data(self, period='1y'):
        """Fetch data for major stock indices"""
        print("Fetching stock indices data...")
        indices_data = {}
        
        for name, ticker in self.stock_indices.items():
            try:
                data = yf.download(ticker, period=period, progress=False)
                if not data.empty:
                    indices_data[name] = data
                    print(f"  ✓ {name}")
                else:
                    print(f"  ✗ {name} - No data available")
            except Exception as e:
                print(f"  ✗ {name} - Error: {e}")
        
        return indices_data
    
    def fetch_currency_data(self, period='1y'):
        """Fetch data for major currencies against USD"""
        print("Fetching currency data...")
        currency_data = {}
        
        for name, ticker in self.currencies.items():
            try:
                data = yf.download(ticker, period=period, progress=False)
                if not data.empty:
                    currency_data[name] = data
                    print(f"  ✓ {name}")
                else:
                    print(f"  ✗ {name} - No data available")
            except Exception as e:
                print(f"  ✗ {name} - Error: {e}")
        
        return currency_data
    
    def fetch_commodity_data(self, period='1y'):
        """Fetch data for major commodities"""
        print("Fetching commodity data...")
        commodity_data = {}
        
        for name, ticker in self.commodities.items():
            try:
                data = yf.download(ticker, period=period, progress=False)
                if not data.empty:
                    commodity_data[name] = data
                    print(f"  ✓ {name}")
                else:
                    print(f"  ✗ {name} - No data available")
            except Exception as e:
                print(f"  ✗ {name} - Error: {e}")
        
        return commodity_data
    
    def fetch_economic_indicators(self):
        """Fetch economic indicators from FRED"""
        print("Fetching economic indicators...")
        indicator_data = {}
        
        if self.fred is None:
            print("  ✗ FRED API key not available. Using alternative data source.")
            # Use alternative data source for key indicators
            try:
                # Fetch US GDP data from FRED's public CSV
                gdp_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP"
                gdp_data = pd.read_csv(io.StringIO(requests.get(gdp_url).text))
                gdp_data.columns = ['Date', 'Value']
                gdp_data['Date'] = pd.to_datetime(gdp_data['Date'])
                indicator_data['US GDP'] = gdp_data
                print("  ✓ US GDP")
                
                # Fetch US Unemployment Rate data
                unrate_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
                unrate_data = pd.read_csv(io.StringIO(requests.get(unrate_url).text))
                unrate_data.columns = ['Date', 'Value']
                unrate_data['Date'] = pd.to_datetime(unrate_data['Date'])
                indicator_data['US Unemployment Rate'] = unrate_data
                print("  ✓ US Unemployment Rate")
                
                # Fetch US Inflation Rate data
                inflation_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
                inflation_data = pd.read_csv(io.StringIO(requests.get(inflation_url).text))
                inflation_data.columns = ['Date', 'Value']
                inflation_data['Date'] = pd.to_datetime(inflation_data['Date'])
                # Calculate year-over-year percentage change for inflation
                inflation_data['YoY_Inflation'] = inflation_data['Value'].pct_change(12) * 100
                indicator_data['US Inflation Rate'] = inflation_data
                print("  ✓ US Inflation Rate")
                
                # Fetch US M1 Money Supply data
                m1_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M1SL"
                m1_data = pd.read_csv(io.StringIO(requests.get(m1_url).text))
                m1_data.columns = ['Date', 'Value']
                m1_data['Date'] = pd.to_datetime(m1_data['Date'])
                # Calculate year-over-year percentage change for money supply
                m1_data['YoY_Growth'] = m1_data['Value'].pct_change(12) * 100
                indicator_data['US M1 Money Supply'] = m1_data
                print("  ✓ US M1 Money Supply")
                
                # Fetch US M2 Money Supply data
                m2_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"
                m2_data = pd.read_csv(io.StringIO(requests.get(m2_url).text))
                m2_data.columns = ['Date', 'Value']
                m2_data['Date'] = pd.to_datetime(m2_data['Date'])
                # Calculate year-over-year percentage change for money supply
                m2_data['YoY_Growth'] = m2_data['Value'].pct_change(12) * 100
                indicator_data['US M2 Money Supply'] = m2_data
                print("  ✓ US M2 Money Supply")
                
            except Exception as e:
                print(f"  ✗ Error fetching alternative data: {e}")
        else:
            # Use FRED API
            for name, series_id in self.indicators.items():
                try:
                    data = self.fred.get_series(series_id)
                    if not data.empty:
                        # Convert to DataFrame
                        df = pd.DataFrame(data).reset_index()
                        df.columns = ['Date', 'Value']
                        
                        # Calculate year-over-year percentage change for inflation
                        if series_id == 'CPIAUCSL':
                            df['YoY_Inflation'] = df['Value'].pct_change(12) * 100
                        
                        # Calculate year-over-year percentage change for money supply
                        if series_id in ['M1SL', 'M2SL', 'MYAGM3EZM196N', 'MYAGM2CNM189N', 'MANMM101JPM189S']:
                            df['YoY_Growth'] = df['Value'].pct_change(12) * 100
                        
                        indicator_data[name] = df
                        print(f"  ✓ {name}")
                    else:
                        print(f"  ✗ {name} - No data available")
                except Exception as e:
                    print(f"  ✗ {name} - Error: {e}")
        
        return indicator_data
    
    def calculate_performance_metrics(self, data_dict):
        """Calculate performance metrics for the provided data"""
        metrics = {}
        
        for name, data in data_dict.items():
            if data.empty:
                continue
                
            # Make sure we have 'Close' column
            if 'Close' in data.columns:
                # Calculate metrics
                current_price = data['Close'].iloc[-1]
                start_price = data['Close'].iloc[0]
                max_price = data['Close'].max()
                min_price = data['Close'].min()
                
                # Calculate returns
                total_return = (current_price - start_price) / start_price * 100
                
                # Calculate volatility (standard deviation of daily returns)
                daily_returns = data['Close'].pct_change().dropna()
                volatility = daily_returns.std() * 100
                
                # Store metrics
                metrics[name] = {
                    'current_price': current_price,
                    'total_return': total_return,
                    'volatility': volatility,
                    'max_price': max_price,
                    'min_price': min_price
                }
        
        return metrics
    
    def plot_stock_indices(self, indices_data):
        """Plot major stock indices performance"""
        print("Generating stock indices plot...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize all indices to 100 at the start for comparison
        for name, data in indices_data.items():
            if not data.empty and 'Close' in data.columns:
                normalized = data['Close'] / data['Close'].iloc[0] * 100
                ax.plot(data.index, normalized, label=name, linewidth=2)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price (Start = 100)')
        ax.set_title('Major Stock Indices Performance (Normalized)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add current date to the plot
        current_date = datetime.now().strftime('%Y-%m-%d')
        ax.annotate(f'Generated on: {current_date}', 
                   xy=(0.02, 0.02), xycoords='figure fraction',
                   fontsize=8, alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig('output/stock_indices_performance.png')
        plt.close()
        
        print("  ✓ Stock indices plot saved as 'output/stock_indices_performance.png'")
    
    def plot_economic_indicators(self, indicator_data):
        """Plot key economic indicators"""
        print("Generating economic indicators plots...")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(14, 16))
        gs = GridSpec(4, 2, figure=fig)
        
        # GDP Plot
        if 'US GDP' in indicator_data:
            ax1 = fig.add_subplot(gs[0, 0])
            gdp_data = indicator_data['US GDP']
            ax1.plot(gdp_data['Date'], gdp_data['Value'], 'b-', linewidth=2)
            ax1.set_title('US GDP (Billions of $)')
            ax1.grid(True, alpha=0.3)
            # Calculate and show growth rate
            gdp_data['YoY_Growth'] = gdp_data['Value'].pct_change(4) * 100  # Quarterly data
            latest_growth = gdp_data['YoY_Growth'].dropna().iloc[-1]
            ax1.annotate(f'Latest YoY Growth: {latest_growth:.2f}%',
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Unemployment Rate Plot
        if 'US Unemployment Rate' in indicator_data:
            ax2 = fig.add_subplot(gs[0, 1])
            unemployment_data = indicator_data['US Unemployment Rate']
            ax2.plot(unemployment_data['Date'], unemployment_data['Value'], 'r-', linewidth=2)
            ax2.set_title('US Unemployment Rate (%)')
            ax2.grid(True, alpha=0.3)
            latest_rate = unemployment_data['Value'].iloc[-1]
            ax2.annotate(f'Latest Rate: {latest_rate:.1f}%',
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Inflation Rate Plot
        if 'US Inflation Rate' in indicator_data:
            ax3 = fig.add_subplot(gs[1, 0])
            inflation_data = indicator_data['US Inflation Rate']
            if 'YoY_Inflation' in inflation_data.columns:
                # Fix: Make sure we're using aligned data by creating a new DataFrame with only valid values
                valid_inflation = inflation_data.dropna(subset=['YoY_Inflation']).copy()
                ax3.plot(valid_inflation['Date'], valid_inflation['YoY_Inflation'], 'g-', linewidth=2)
                ax3.set_title('US Inflation Rate (YoY %)')
                ax3.grid(True, alpha=0.3)
                latest_inflation = valid_inflation['YoY_Inflation'].iloc[-1]
                ax3.annotate(f'Latest Inflation: {latest_inflation:.1f}%',
                            xy=(0.05, 0.05), xycoords='axes fraction',
                            fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Federal Funds Rate Plot
        if 'US Federal Funds Rate' in indicator_data:
            ax4 = fig.add_subplot(gs[1, 1])
            fed_rate_data = indicator_data['US Federal Funds Rate']
            ax4.plot(fed_rate_data['Date'], fed_rate_data['Value'], 'c-', linewidth=2)
            ax4.set_title('US Federal Funds Rate (%)')
            ax4.grid(True, alpha=0.3)
            latest_rate = fed_rate_data['Value'].iloc[-1]
            ax4.annotate(f'Latest Rate: {latest_rate:.2f}%',
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # 10-Year Treasury Yield Plot
        if 'US 10-Year Treasury Yield' in indicator_data:
            ax5 = fig.add_subplot(gs[2, 0])
            treasury_data = indicator_data['US 10-Year Treasury Yield']
            ax5.plot(treasury_data['Date'], treasury_data['Value'], 'm-', linewidth=2)
            ax5.set_title('US 10-Year Treasury Yield (%)')
            ax5.grid(True, alpha=0.3)
            latest_yield = treasury_data['Value'].iloc[-1]
            ax5.annotate(f'Latest Yield: {latest_yield:.2f}%',
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Consumer Sentiment Plot
        if 'US Consumer Sentiment' in indicator_data:
            ax6 = fig.add_subplot(gs[2, 1])
            sentiment_data = indicator_data['US Consumer Sentiment']
            ax6.plot(sentiment_data['Date'], sentiment_data['Value'], 'y-', linewidth=2)
            ax6.set_title('US Consumer Sentiment Index')
            ax6.grid(True, alpha=0.3)
            latest_sentiment = sentiment_data['Value'].iloc[-1]
            ax6.annotate(f'Latest Index: {latest_sentiment:.1f}',
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Money Supply Plot - Compare M1 and M2
        ax7 = fig.add_subplot(gs[3, 0])
        if 'US M1 Money Supply' in indicator_data and 'US M2 Money Supply' in indicator_data:
            m1_data = indicator_data['US M1 Money Supply']
            m2_data = indicator_data['US M2 Money Supply']
            
            # Plot both M1 and M2 on the same axis
            ax7.plot(m1_data['Date'], m1_data['Value'] / 1000, 'b-', linewidth=2, label='M1')
            ax7.plot(m2_data['Date'], m2_data['Value'] / 1000, 'r-', linewidth=2, label='M2')
            ax7.set_title('US Money Supply (Trillions of $)')
            ax7.grid(True, alpha=0.3)
            ax7.legend(loc='best')
            
            # Show latest values
            latest_m1 = m1_data['Value'].iloc[-1] / 1000
            latest_m2 = m2_data['Value'].iloc[-1] / 1000
            ax7.annotate(f'Latest M1: ${latest_m1:.2f}T\nLatest M2: ${latest_m2:.2f}T',
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        elif 'US M2 Money Supply' in indicator_data:
            m2_data = indicator_data['US M2 Money Supply']
            ax7.plot(m2_data['Date'], m2_data['Value'] / 1000, 'r-', linewidth=2)
            ax7.set_title('US M2 Money Supply (Trillions of $)')
            ax7.grid(True, alpha=0.3)
            latest_m2 = m2_data['Value'].iloc[-1] / 1000
            ax7.annotate(f'Latest M2: ${latest_m2:.2f}T',
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Global Money Supply Growth Rates
        ax8 = fig.add_subplot(gs[3, 1])
        money_supply_series = []
        
        # Add available money supply growth rates
        for ms_name in ['US M2 Money Supply', 'Euro Area M3 Money Supply', 'China M2 Money Supply', 'Japan M2 Money Supply']:
            if ms_name in indicator_data and 'YoY_Growth' in indicator_data[ms_name].columns:
                ms_data = indicator_data[ms_name]
                valid_data = ms_data.dropna(subset=['YoY_Growth']).copy()
                if not valid_data.empty:
                    label = ms_name.replace(' Money Supply', '')
                    ax8.plot(valid_data['Date'], valid_data['YoY_Growth'], linewidth=2, label=label)
                    money_supply_series.append((label, valid_data['YoY_Growth'].iloc[-1]))
        
        if money_supply_series:
            ax8.set_title('Global Money Supply Growth (YoY %)')
            ax8.grid(True, alpha=0.3)
            ax8.legend(loc='best')
            
            # Show latest values
            annotation_text = '\n'.join([f'{name}: {value:.1f}%' for name, value in money_supply_series])
            ax8.annotate(annotation_text,
                        xy=(0.05, 0.05), xycoords='axes fraction',
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add current date to the plot
        current_date = datetime.now().strftime('%Y-%m-%d')
        fig.text(0.02, 0.02, f'Generated on: {current_date}', fontsize=8, alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig('output/economic_indicators.png')
        plt.close()
        
        print("  ✓ Economic indicators plot saved as 'output/economic_indicators.png'")
    
    def plot_currencies(self, currency_data):
        """Plot major currencies against USD"""
        print("Generating currency plot...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize all currencies to 100 at the start for comparison
        for name, data in currency_data.items():
            if not data.empty and 'Close' in data.columns:
                normalized = data['Close'] / data['Close'].iloc[0] * 100
                ax.plot(data.index, normalized, label=name, linewidth=2)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Value (Start = 100)')
        ax.set_title('Major Currencies Performance vs USD (Normalized)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add current date to the plot
        current_date = datetime.now().strftime('%Y-%m-%d')
        ax.annotate(f'Generated on: {current_date}', 
                   xy=(0.02, 0.02), xycoords='figure fraction',
                   fontsize=8, alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig('output/currency_performance.png')
        plt.close()
        
        print("  ✓ Currency plot saved as 'output/currency_performance.png'")
    
    def plot_commodities(self, commodity_data):
        """Plot major commodities performance"""
        print("Generating commodities plot...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize all commodities to 100 at the start for comparison
        for name, data in commodity_data.items():
            if not data.empty and 'Close' in data.columns:
                normalized = data['Close'] / data['Close'].iloc[0] * 100
                ax.plot(data.index, normalized, label=name, linewidth=2)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price (Start = 100)')
        ax.set_title('Major Commodities Performance (Normalized)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add current date to the plot
        current_date = datetime.now().strftime('%Y-%m-%d')
        ax.annotate(f'Generated on: {current_date}', 
                   xy=(0.02, 0.02), xycoords='figure fraction',
                   fontsize=8, alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig('output/commodity_performance.png')
        plt.close()
        
        print("  ✓ Commodities plot saved as 'output/commodity_performance.png'")
    
    def generate_summary_report(self, indices_metrics, currency_metrics, commodity_metrics, indicator_data):
        """Generate a summary report of the global economic overview"""
        print("Generating summary report...")
        
        # Create report
        report = []
        
        # Add title and date
        current_date = datetime.now().strftime('%Y-%m-%d')
        report.append("# Global Economic Overview")
        report.append(f"**Generated on: {current_date}**\n")
        
        # Stock Market Overview
        report.append("## Stock Market Overview")
        
        for name, metrics in indices_metrics.items():
            # Ensure we're dealing with scalar values, not Series
            current_price = metrics['current_price']
            total_return = metrics['total_return']
            volatility = metrics['volatility']
            
            if hasattr(current_price, 'item'):
                current_price = current_price.item()
            if hasattr(total_return, 'item'):
                total_return = total_return.item()
            if hasattr(volatility, 'item'):
                volatility = volatility.item()
                
            report.append(f"- **{name}**: {current_price:.2f} (YTD: {total_return:.2f}%, Volatility: {volatility:.2f}%)")
        
        report.append("")
        
        # Currency Overview
        report.append("## Currency Overview (vs USD)")
        
        for name, metrics in currency_metrics.items():
            # Ensure we're dealing with scalar values, not Series
            current_price = metrics['current_price']
            total_return = metrics['total_return']
            
            if hasattr(current_price, 'item'):
                current_price = current_price.item()
            if hasattr(total_return, 'item'):
                total_return = total_return.item()
                
            report.append(f"- **{name}**: {current_price:.4f} (YTD: {total_return:.2f}%)")
        
        report.append("")
        
        # Commodity Overview
        report.append("## Commodity Overview")
        
        for name, metrics in commodity_metrics.items():
            # Ensure we're dealing with scalar values, not Series
            current_price = metrics['current_price']
            total_return = metrics['total_return']
            
            if hasattr(current_price, 'item'):
                current_price = current_price.item()
            if hasattr(total_return, 'item'):
                total_return = total_return.item()
                
            report.append(f"- **{name}**: ${current_price:.2f} (YTD: {total_return:.2f}%)")
        
        report.append("")
        
        # Economic Indicators
        report.append("## Economic Indicators")
        
        # US GDP
        if 'US GDP' in indicator_data:
            gdp_data = indicator_data['US GDP']
            latest_gdp = gdp_data['Value'].iloc[-1]
            if 'YoY_Growth' in gdp_data.columns:
                latest_growth = gdp_data['YoY_Growth'].dropna().iloc[-1]
                report.append(f"- **US GDP**: ${latest_gdp:.1f} billion (YoY Growth: {latest_growth:.2f}%)")
            else:
                report.append(f"- **US GDP**: ${latest_gdp:.1f} billion")
        
        # US Unemployment Rate
        if 'US Unemployment Rate' in indicator_data:
            unemployment_data = indicator_data['US Unemployment Rate']
            latest_rate = unemployment_data['Value'].iloc[-1]
            report.append(f"- **US Unemployment Rate**: {latest_rate:.1f}%")
        
        # US Inflation Rate
        if 'US Inflation Rate' in indicator_data and 'YoY_Inflation' in indicator_data['US Inflation Rate'].columns:
            inflation_data = indicator_data['US Inflation Rate']
            latest_inflation = inflation_data['YoY_Inflation'].dropna().iloc[-1]
            report.append(f"- **US Inflation Rate**: {latest_inflation:.1f}% (YoY)")
        
        # US Federal Funds Rate
        if 'US Federal Funds Rate' in indicator_data:
            fed_rate_data = indicator_data['US Federal Funds Rate']
            latest_rate = fed_rate_data['Value'].iloc[-1]
            report.append(f"- **US Federal Funds Rate**: {latest_rate:.2f}%")
        
        # US 10-Year Treasury Yield
        if 'US 10-Year Treasury Yield' in indicator_data:
            treasury_data = indicator_data['US 10-Year Treasury Yield']
            latest_yield = treasury_data['Value'].iloc[-1]
            report.append(f"- **US 10-Year Treasury Yield**: {latest_yield:.2f}%")
        
        # US Consumer Sentiment
        if 'US Consumer Sentiment' in indicator_data:
            sentiment_data = indicator_data['US Consumer Sentiment']
            latest_sentiment = sentiment_data['Value'].iloc[-1]
            report.append(f"- **US Consumer Sentiment Index**: {latest_sentiment:.1f}")
        
        # Money Supply Indicators
        report.append("\n### Global Money Supply")
        
        # US M1 Money Supply
        if 'US M1 Money Supply' in indicator_data:
            m1_data = indicator_data['US M1 Money Supply']
            latest_m1 = m1_data['Value'].iloc[-1] / 1000  # Convert to trillions
            if 'YoY_Growth' in m1_data.columns:
                latest_growth = m1_data['YoY_Growth'].dropna().iloc[-1]
                report.append(f"- **US M1 Money Supply**: ${latest_m1:.2f} trillion (YoY Growth: {latest_growth:.2f}%)")
            else:
                report.append(f"- **US M1 Money Supply**: ${latest_m1:.2f} trillion")
        
        # US M2 Money Supply
        if 'US M2 Money Supply' in indicator_data:
            m2_data = indicator_data['US M2 Money Supply']
            latest_m2 = m2_data['Value'].iloc[-1] / 1000  # Convert to trillions
            if 'YoY_Growth' in m2_data.columns:
                latest_growth = m2_data['YoY_Growth'].dropna().iloc[-1]
                report.append(f"- **US M2 Money Supply**: ${latest_m2:.2f} trillion (YoY Growth: {latest_growth:.2f}%)")
            else:
                report.append(f"- **US M2 Money Supply**: ${latest_m2:.2f} trillion")
        
        # Euro Area M3 Money Supply
        if 'Euro Area M3 Money Supply' in indicator_data:
            euro_m3_data = indicator_data['Euro Area M3 Money Supply']
            latest_euro_m3 = euro_m3_data['Value'].iloc[-1]
            if 'YoY_Growth' in euro_m3_data.columns:
                latest_growth = euro_m3_data['YoY_Growth'].dropna().iloc[-1]
                report.append(f"- **Euro Area M3 Money Supply**: YoY Growth: {latest_growth:.2f}%")
            else:
                report.append(f"- **Euro Area M3 Money Supply**: Latest value available")
        
        # China M2 Money Supply
        if 'China M2 Money Supply' in indicator_data:
            china_m2_data = indicator_data['China M2 Money Supply']
            latest_china_m2 = china_m2_data['Value'].iloc[-1]
            if 'YoY_Growth' in china_m2_data.columns:
                latest_growth = china_m2_data['YoY_Growth'].dropna().iloc[-1]
                report.append(f"- **China M2 Money Supply**: YoY Growth: {latest_growth:.2f}%")
            else:
                report.append(f"- **China M2 Money Supply**: Latest value available")
        
        # Japan M2 Money Supply
        if 'Japan M2 Money Supply' in indicator_data:
            japan_m2_data = indicator_data['Japan M2 Money Supply']
            latest_japan_m2 = japan_m2_data['Value'].iloc[-1]
            if 'YoY_Growth' in japan_m2_data.columns:
                latest_growth = japan_m2_data['YoY_Growth'].dropna().iloc[-1]
                report.append(f"- **Japan M2 Money Supply**: YoY Growth: {latest_growth:.2f}%")
            else:
                report.append(f"- **Japan M2 Money Supply**: Latest value available")
        
        report.append("")
        
        # Update visualization references to point to the output folder
        report.append("## Generated Visualizations")
        report.append("- Stock Indices Performance: `output/stock_indices_performance.png`")
        report.append("- Economic Indicators: `output/economic_indicators.png`")
        report.append("- Currency Performance: `output/currency_performance.png`")
        report.append("- Commodity Performance: `output/commodity_performance.png`")
        
        # Write report to file
        with open('output/global_economic_overview.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("  ✓ Summary report saved as 'output/global_economic_overview.md'")
        
        # Print a brief summary to console
        print("\nBrief Economic Overview:")
        # Add money supply indicators to the brief summary
        if 'US M1 Money Supply' in indicator_data:
            m1_data = indicator_data['US M1 Money Supply']
            latest_m1 = m1_data['Value'].iloc[-1] / 1000  # Convert to trillions
            if 'YoY_Growth' in m1_data.columns:
                latest_growth = m1_data['YoY_Growth'].dropna().iloc[-1]
                print(f"- **US M1 Money Supply**: ${latest_m1:.2f} trillion (YoY Growth: {latest_growth:.2f}%)")
            else:
                print(f"- **US M1 Money Supply**: ${latest_m1:.2f} trillion")
        
        if 'US M2 Money Supply' in indicator_data:
            m2_data = indicator_data['US M2 Money Supply']
            latest_m2 = m2_data['Value'].iloc[-1] / 1000  # Convert to trillions
            if 'YoY_Growth' in m2_data.columns:
                latest_growth = m2_data['YoY_Growth'].dropna().iloc[-1]
                print(f"- **US M2 Money Supply**: ${latest_m2:.2f} trillion (YoY Growth: {latest_growth:.2f}%)")
            else:
                print(f"- **US M2 Money Supply**: ${latest_m2:.2f} trillion")
        
        if 'Euro Area M3 Money Supply' in indicator_data:
            euro_m3_data = indicator_data['Euro Area M3 Money Supply']
            latest_euro_m3 = euro_m3_data['Value'].iloc[-1]
            if 'YoY_Growth' in euro_m3_data.columns:
                latest_growth = euro_m3_data['YoY_Growth'].dropna().iloc[-1]
                print(f"- **Euro Area M3 Money Supply**: YoY Growth: {latest_growth:.2f}%")
            else:
                print(f"- **Euro Area M3 Money Supply**: Latest value available")
        
        if 'China M2 Money Supply' in indicator_data:
            china_m2_data = indicator_data['China M2 Money Supply']
            latest_china_m2 = china_m2_data['Value'].iloc[-1]
            if 'YoY_Growth' in china_m2_data.columns:
                latest_growth = china_m2_data['YoY_Growth'].dropna().iloc[-1]
                print(f"- **China M2 Money Supply**: YoY Growth: {latest_growth:.2f}%")
            else:
                print(f"- **China M2 Money Supply**: Latest value available")
        
        if 'Japan M2 Money Supply' in indicator_data:
            japan_m2_data = indicator_data['Japan M2 Money Supply']
            latest_japan_m2 = japan_m2_data['Value'].iloc[-1]
            if 'YoY_Growth' in japan_m2_data.columns:
                latest_growth = japan_m2_data['YoY_Growth'].dropna().iloc[-1]
                print(f"- **Japan M2 Money Supply**: YoY Growth: {latest_growth:.2f}%")
            else:
                print(f"- **Japan M2 Money Supply**: Latest value available")
    
    def run_analysis(self, period='1y'):
        """Run the full economic analysis"""
        print("\n=== GLOBAL ECONOMIC OVERVIEW ANALYSIS ===\n")
        
        # Fetch data
        indices_data = self.fetch_stock_indices_data(period)
        currency_data = self.fetch_currency_data(period)
        commodity_data = self.fetch_commodity_data(period)
        indicator_data = self.fetch_economic_indicators()
        
        # Calculate metrics
        indices_metrics = self.calculate_performance_metrics(indices_data)
        currency_metrics = self.calculate_performance_metrics(currency_data)
        commodity_metrics = self.calculate_performance_metrics(commodity_data)
        
        # Generate visualizations
        self.plot_stock_indices(indices_data)
        self.plot_economic_indicators(indicator_data)
        self.plot_currencies(currency_data)
        self.plot_commodities(commodity_data)
        
        # Generate summary report
        self.generate_summary_report(indices_metrics, currency_metrics, commodity_metrics, indicator_data)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Summary report and visualizations have been generated.")
        print("Check 'output/global_economic_overview.md' for the full report.")

def main():
    """Main function to run the global economic overview analysis"""
    # Create an instance of the GlobalEconomicOverview class
    economic_overview = GlobalEconomicOverview()
    
    # Run the analysis with data from the past year
    economic_overview.run_analysis(period='1y')

if __name__ == "__main__":
    main()
