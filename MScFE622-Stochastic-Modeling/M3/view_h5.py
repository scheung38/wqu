# import pandas as pd

# file_path = 'Stochastic-Modeling/M3/option_data_wqu.h5'
# # List all keys (dataframes/groups) in the file
# with pd.HDFStore(file_path, 'r') as store:
#     print(store.keys())
    
# # Read a specific dataframe (replace 'key_name' with the actual key)
# df = pd.read_hdf(file_path, key='data')
# print(df.head())