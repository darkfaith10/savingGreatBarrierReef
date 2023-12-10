import pandas as pd

# read the CSV file into a Pandas DataFrame object
df_sift = pd.read_csv('/Users/vijaykumarsingh/Desktop/onemore/ORB/ORB_combined.csv')
df_orb = pd.read_csv('/Users/vijaykumarsingh/Desktop/onemore/SIFT/SIFT_combined.csv')

# print the shape of the DataFrame
print(df_sift.shape)
print(df_orb.shape)
