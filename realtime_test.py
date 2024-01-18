import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('packets.csv', encoding='latin1', on_bad_lines='warn', usecols=range(87), dtype={17: str})

print(df.iloc[:, 17].unique())

# Basic preprocessing
# Select numerical features or features you deem relevant
# For the sake of the example, let's assume you've identified some features
#selected_features = ['feature1', 'feature2', 'feature3']  # replace with your actual feature names
#df = df[selected_features]

# Fill missing values if any
df.fillna(0, inplace=True)

# Select only numeric data
df_numeric = df._get_numeric_data()

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)


# Save preprocessed data
pd.DataFrame(scaled_data, columns=df_numeric.columns).to_csv('preprocessed_packets.csv', index=False)

