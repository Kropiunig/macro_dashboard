import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import io

def fetch_usd_eur_historical_data():
    """
    Fetches historical USD/EUR exchange rate data from the Federal Reserve Economic Data (FRED)
    Returns a pandas DataFrame with date and exchange rate
    """
    # Using FRED API to get USD/EUR exchange rate data
    # This is a public dataset that doesn't require authentication
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the CSV data
        df = pd.read_csv(io.StringIO(response.text))
        
        # Rename columns for clarity
        df.columns = ['Date', 'USD_EUR']
        
        # Convert date to datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Sort by date
        df = df.sort_values('Date')
        
        return df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_30day_fluctuations(df):
    """
    Calculates the 30-day fluctuation in exchange rates
    Returns a DataFrame with date and fluctuation percentage
    """
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Calculate the 30-day price difference
    result_df['30d_price'] = result_df['USD_EUR'].shift(-30)
    
    # Calculate the absolute percentage change over 30 days
    result_df['30d_fluctuation_pct'] = abs((result_df['30d_price'] - result_df['USD_EUR']) / result_df['USD_EUR'] * 100)
    
    # Calculate the actual percentage change (not absolute)
    result_df['30d_actual_change_pct'] = (result_df['30d_price'] - result_df['USD_EUR']) / result_df['USD_EUR'] * 100
    
    # Drop rows with NaN values (last 30 days)
    result_df = result_df.dropna()
    
    return result_df

def find_highest_fluctuation(df):
    """
    Finds the period with the highest 30-day fluctuation
    Returns information about the highest fluctuation period
    """
    # Find the row with the maximum fluctuation
    max_fluctuation_row = df.loc[df['30d_fluctuation_pct'].idxmax()]
    
    # Get the start date
    start_date = max_fluctuation_row['Date']
    
    # Get the end date (30 days later)
    end_date = start_date + timedelta(days=30)
    
    # Get the start and end prices
    start_price = max_fluctuation_row['USD_EUR']
    end_price = max_fluctuation_row['30d_price']
    
    # Calculate the actual change (not absolute)
    actual_change_pct = (end_price - start_price) / start_price * 100
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'start_price': start_price,
        'end_price': end_price,
        'fluctuation_pct': max_fluctuation_row['30d_fluctuation_pct'],
        'actual_change_pct': actual_change_pct
    }

def plot_highest_fluctuation(df, highest_fluctuation):
    """
    Creates a plot showing the highest fluctuation period
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the entire exchange rate history
    ax.plot(df['Date'], df['USD_EUR'], color='blue', alpha=0.5, label='USD/EUR Exchange Rate')
    
    # Highlight the highest fluctuation period
    start_date = highest_fluctuation['start_date']
    end_date = highest_fluctuation['end_date']
    
    # Get data for the highest fluctuation period
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    period_data = df.loc[mask]
    
    # Plot the highest fluctuation period
    ax.plot(period_data['Date'], period_data['USD_EUR'], color='red', linewidth=2, 
            label=f'Highest 30-day Fluctuation: {highest_fluctuation["fluctuation_pct"]:.2f}%')
    
    # Mark the start and end points
    ax.scatter([start_date, end_date], 
               [highest_fluctuation['start_price'], highest_fluctuation['end_price']], 
               color='red', s=100)
    
    # Add annotations
    ax.annotate(f"{highest_fluctuation['start_price']:.4f}",
                (start_date, highest_fluctuation['start_price']),
                xytext=(10, -20), textcoords='offset points')
    
    ax.annotate(f"{highest_fluctuation['end_price']:.4f}",
                (end_date, highest_fluctuation['end_price']),
                xytext=(10, -20), textcoords='offset points')
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('USD/EUR Exchange Rate')
    ax.set_title('USD/EUR Exchange Rate with Highest 30-day Fluctuation Period')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('usd_eur_highest_fluctuation.png')
    plt.close()

def plot_30day_distribution(df):
    """
    Creates a histogram showing the distribution of 30-day price moves
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot histogram of absolute fluctuations
    ax1.hist(df['30d_fluctuation_pct'], bins=50, color='blue', alpha=0.7)
    ax1.set_xlabel('Absolute 30-day Fluctuation (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Absolute 30-day USD/EUR Fluctuations')
    ax1.grid(True, alpha=0.3)
    
    # Add vertical line for the highest fluctuation
    max_fluctuation = df['30d_fluctuation_pct'].max()
    ax1.axvline(x=max_fluctuation, color='red', linestyle='--', 
                label=f'Max Fluctuation: {max_fluctuation:.2f}%')
    ax1.legend()
    
    # Plot histogram of actual changes (not absolute)
    ax2.hist(df['30d_actual_change_pct'], bins=50, color='green', alpha=0.7)
    ax2.set_xlabel('Actual 30-day Change (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Actual 30-day USD/EUR Changes')
    ax2.grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    mean_change = df['30d_actual_change_pct'].mean()
    median_change = df['30d_actual_change_pct'].median()
    
    ax2.axvline(x=mean_change, color='red', linestyle='--', 
                label=f'Mean: {mean_change:.2f}%')
    ax2.axvline(x=median_change, color='orange', linestyle=':', 
                label=f'Median: {median_change:.2f}%')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.legend()
    
    # Add some statistics as text
    stats_text = (
        f"Statistics for 30-day USD/EUR Changes:\n"
        f"Mean: {mean_change:.2f}%\n"
        f"Median: {median_change:.2f}%\n"
        f"Min: {df['30d_actual_change_pct'].min():.2f}%\n"
        f"Max: {df['30d_actual_change_pct'].max():.2f}%\n"
        f"Std Dev: {df['30d_actual_change_pct'].std():.2f}%\n"
    )
    
    # Position the text box in figure coords
    fig.text(0.15, 0.01, stats_text, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add time period information
    time_period = (
        f"Time Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}\n"
        f"Total Observations: {len(df)}"
    )
    fig.text(0.65, 0.01, time_period, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the text at the bottom
    plt.savefig('usd_eur_30day_distribution.png')
    plt.close()
    
    # Create a time series plot of the 30-day changes
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the time series of 30-day changes
    ax.plot(df['Date'], df['30d_actual_change_pct'], color='blue', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('30-day Change (%)')
    ax.set_title('Time Series of 30-day USD/EUR Changes')
    ax.grid(True, alpha=0.3)
    
    # Add a horizontal line at zero
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('usd_eur_30day_timeseries.png')
    plt.close()

def main():
    print("Fetching historical USD/EUR exchange rate data...")
    df = fetch_usd_eur_historical_data()
    
    if df is None:
        print("Failed to fetch data. Exiting.")
        return
    
    print(f"Data fetched successfully. Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    print("Calculating 30-day fluctuations...")
    fluctuations_df = calculate_30day_fluctuations(df)
    
    print("Finding highest 30-day fluctuation...")
    highest_fluctuation = find_highest_fluctuation(fluctuations_df)
    
    # Print the results
    print("\nHighest 30-day Fluctuation in USD/EUR Exchange Rate:")
    print(f"Period: {highest_fluctuation['start_date'].strftime('%Y-%m-%d')} to {highest_fluctuation['end_date'].strftime('%Y-%m-%d')}")
    print(f"Starting Rate: {highest_fluctuation['start_price']:.4f}")
    print(f"Ending Rate: {highest_fluctuation['end_price']:.4f}")
    print(f"Absolute Fluctuation: {highest_fluctuation['fluctuation_pct']:.2f}%")
    print(f"Actual Change: {highest_fluctuation['actual_change_pct']:.2f}%")
    
    print("\nCreating plots...")
    plot_highest_fluctuation(df, highest_fluctuation)
    print("Plot saved as 'usd_eur_highest_fluctuation.png'")
    
    print("Creating distribution plots of 30-day price moves...")
    plot_30day_distribution(fluctuations_df)
    print("Distribution plots saved as 'usd_eur_30day_distribution.png' and 'usd_eur_30day_timeseries.png'")

if __name__ == "__main__":
    main()
