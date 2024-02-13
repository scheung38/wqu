import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from scipy.stats import skew, kurtosis 

# List of CSV file paths of 2 years of historical data for 3 cryptocurrencies
csv_files = ['Bitcoin_cash.csv', 'Ethereum_classic.csv', 'Solana.csv']

# Dictionary to hold 'change' columns from each DataFrame
change_columns = {}

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
    print(f"Average Return: {df_sorted['change'].mean()}")
    
    # Step 4: Prepare the 'change' column: remove percentage signs and convert to float
    df_sorted['change'] = df_sorted['change'].str.replace('%', '').astype(float) / 100

    # Step 5: Calculate the required statistics on the 'change' column
    std_dev = df_sorted['change'].std()
    skewness = df_sorted['change'].skew()
    kurt = df_sorted['change'].kurt()

    # Print the statistics
    print(f"Standard Deviation: {std_dev}")
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

# Step 10: Optionally display the matrices
print("Correlation Matrix:")
print(correlation_matrix)
print("\nCovariance Matrix:")
print(covariance_matrix)   
