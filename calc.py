import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from scipy.stats import skew, kurtosis 
import numpy as np

# List of CSV file paths of 2 years of historical data for 3 cryptocurrencies
csv_files = ['Bitcoin_cash.csv', 'Ethereum_classic.csv', 'Solana.csv']

# Dictionary to hold 'change' columns and statistics from each DataFrame
change_columns = {}
statistics = {}
portfolio_weights = {'Bitcoin_cash': 0.3, 'Ethereum_classic': 0.3, 'Solana': 0.4}

# Read each CSV file into a dataframe and store in list
for file in csv_files:

    # Step 1: Read the CSV into a DataFrame
    df = pd.read_csv(file)

    # Step 2: Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Step 3: Sort the DataFrame by the 'date' column, oldest first
    df_sorted = df.sort_values(by='date')

    # Display the sorted DataFrame
    print(f"{file}")
    print(f"Top  5: {df_sorted.head()}")
    print("")     
    # print(f"Average Return: {df_sorted['change']}")
    
    # Step 4: Prepare the 'change' column: remove percentage signs and convert to float
    df_sorted['change'] = df_sorted['change'].str.replace('%', '').astype(float) / 100



    # Calculate the average daily return
    avg_daily_return = df_sorted['change'].mean()

    # Assume 252 trading days in a year for most financial markets
    trading_days = 252
    annualized_return = avg_daily_return * trading_days

    # Step 5: Calculate the required statistics on the 'change' column
    std_dev = df_sorted['change'].std()
    skewness = df_sorted['change'].skew()
    kurt = df_sorted['change'].kurt()

    # Print the statistics
    print(f"Annualized Return (without compounding): {annualized_return * 100}%")

    # print(f"Standard Deviation: {std_dev}")
    print(f"Standard Deviation: {std_dev * 100}%")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurt}")
    
    # Step 6: Use the file name without '.csv' as the key and store the 'change' column
    dataframe_key = file.replace('.csv', '')
    change_columns[dataframe_key] = df_sorted['change']
    
    print("-----------------------------------------")
   
# Step 7: After processing all files, create a DataFrame from the 'change' columns for correlation and covariance calculation
combined_change_df = pd.DataFrame(change_columns)

print("Combined DataFrame: {}".format(combined_change_df))

# Step 8: Calculate correlation matrix
correlation_matrix = combined_change_df.corr()

# Step 9: Calculate covariance matrix
covariance_matrix = combined_change_df.cov()

# Calculate portfolio statistics
# Assign the weightings to each cryptocurrency and calculate the portfolio's combined average return, standard deviation, skewness, and kurtosis.

# Weightings
weights = np.array([0.3, 0.3, 0.4])  # 30% BCH, 30% ETC, 40% SOL

# Calculate weighted returns
df['Portfolio']= combined_change_df.dot(weights)

# Combined Average Return
combined_avg_return = df['Portfolio'].mean()

# Combined Standard Deviation
combined_std = df['Portfolio'].std()

# Combined Skewness
combined_skew = skew(df['Portfolio'])

# Combined Kurtosis
combined_kurtosis = kurtosis(df['Portfolio'])

print(f"Combined Average Return: {combined_avg_return * 100}%")
print(f"Combined Standard Deviation: {combined_std * 100}%")
print(f"Combined Skewness: {combined_skew}")
print(f"Combined Kurtosis: {combined_kurtosis}")

# portfolio_return = sum([statistics[crypto]['Annualized Return'] * weight for crypto, weight in portfolio_weights.items()])
# portfolio_std_dev = sum([statistics[crypto]['Standard Deviation'] * weight for crypto, weight in portfolio_weights.items()])
# portfolio_skewness = sum([statistics[crypto]['Skewness'] * weight for crypto, weight in portfolio_weights.items()])
# portfolio_kurtosis = sum([statistics[crypto]['Kurtosis'] * weight for crypto, weight in portfolio_weights.items()])

# # Print portfolio statistics
# print(f"Portfolio Annualized Return (without compounding): {portfolio_return * 100}%")
# print(f"Portfolio Standard Deviation: {portfolio_std_dev * 100}%")
# print(f"Portfolio Skewness: {portfolio_skewness}")
# print(f"Portfolio Kurtosis: {portfolio_kurtosis}")

# Step 10: Optionally display the matrices
print("Correlation Matrix:")
print(correlation_matrix)
print("\nCovariance Matrix:")
print(covariance_matrix)