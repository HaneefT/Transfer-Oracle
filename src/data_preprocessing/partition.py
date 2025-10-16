import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pd.set_option('future.no_silent_downcasting', True)
# Define the path to your CSV file and the desired Parquet output file
csv_file_path = './data/raw/players_data_2024_2025.csv'
parquet_file_path = './data/processed/players_data_processed.parquet'
parquet_file_path_GK = './data/processed/players_data_GK.parquet'
parquet_file_path_DF = './data/processed/players_data_DF.parquet'
parquet_file_path_MF = './data/processed/players_data_MF.parquet'
parquet_file_path_FW = './data/processed/players_data_FW.parquet'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)
# Write a sample of the dataframe to a CSV file for inspection
df.sample(5).to_csv('./data/raw/sample_players_data.csv', index=False)
# Assign an exposure score for each player based on minutes played
def calculate_exposure_score(comp, minutes):
    if comp in ["de Bundesliga", "Fr Ligue 1"]:
        return float("{:.2f}".format(minutes / (34 * 90))) # 34 game season
    else:
        return float("{:.2f}".format(minutes / (38 * 90)))  # 38 game season

# Assign exposure scores to players for penalising low playtime
df['exposure_score'] = df.apply(lambda row: calculate_exposure_score(row['Comp'], row['Min']), axis=1)
# Drop players with an exposure score below 0.3
df = df[df['exposure_score'] >= 0.3]
# Make 0 s into NaNs for the purpose of finding columns with no missing values
df.replace(0, pd.NA, inplace=True)
# Remove duplicate rows and columns
df = df.T.drop_duplicates().T

# make a sample of the dataframe to a CSV file for inspection
df.sample(5).to_csv('./data/processed/sample_players_data_processed.csv', index=False)

# partition the data into sets based on the Position column
df_GK= df[df['Pos'].str.split(',').str[0] == 'GK']
df_DF = df[(df['Pos'].str.split(',').str[0] == 'DF') | (df['Pos'].str.split(',').str[1] == 'DF')]
df_MF = df[(df['Pos'].str.split(',').str[0] == 'MF') | (df['Pos'].str.split(',').str[1] == 'MF')]
df_FW = df[(df['Pos'].str.split(',').str[0] == 'FW') | (df['Pos'].str.split(',').str[1] == 'FW')]

# clean the partitioned dataframes
df_GK = df_GK.T.drop_duplicates().T
df_GK = df_GK.dropna(thresh=len(df_GK)*0.6, axis=1)

df_DF = df_DF.T.drop_duplicates().T
df_DF = df_DF.dropna(thresh=len(df_DF)*0.6, axis=1)

df_MF = df_MF.T.drop_duplicates().T
df_MF = df_MF.dropna(thresh=len(df_MF)*0.6, axis=1)

df_FW = df_FW.T.drop_duplicates().T
df_FW = df_FW.dropna(thresh=len(df_FW)*0.6, axis=1)

# Convert the Pandas DataFrames to PyArrow Tables
table = pa.Table.from_pandas(df)
table_GK = pa.Table.from_pandas(df_GK)
table_DF = pa.Table.from_pandas(df_DF)
table_MF = pa.Table.from_pandas(df_MF)
table_FW = pa.Table.from_pandas(df_FW)

# # Write the PyArrow Tables to a Parquet file
pq.write_table(table, parquet_file_path)
pq.write_table(table_GK, parquet_file_path_GK)
pq.write_table(table_DF, parquet_file_path_DF)
pq.write_table(table_MF, parquet_file_path_MF)
pq.write_table(table_FW, parquet_file_path_FW)



