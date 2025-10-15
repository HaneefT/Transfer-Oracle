import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Define the path to your CSV file and the desired Parquet output file
csv_file_path = './data/raw/players_data-2024_2025.csv'
parquet_file_path = './data/processed/players_data_processed.parquet'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)
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
# Remove duplicate rows and columns
df = df.T.drop_duplicates().T
df_cleaned = df.dropna(axis=1, how='all')

# Convert the Pandas DataFrame to a PyArrow Table
table = pa.Table.from_pandas(df)

# Write the PyArrow Table to a Parquet file
pq.write_table(table, parquet_file_path)