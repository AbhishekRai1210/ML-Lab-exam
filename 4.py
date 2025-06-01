/*For a given set of training data examples stored in a .CSV file, implement and demonstrate the Find-S
algorithm to output a description of the set of all hypotheses consistent with the training examples. */
import pandas as pd

# Corrected file path
df = pd.read_csv(r"C:\Users\abhir\Downloads\training_data.csv")

X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Initialize hypothesis based on first positive instance
hypothesis = X.iloc[0] if y.iloc[0] == 'Yes' else ['ϕ']

# Find-S Algorithm
for i in range(1, len(y)):
    if y.iloc[i] == 'Yes':  # Use .iloc to ensure scalar comparison
        hypothesis = [X.iloc[i] if h == '?' else h for h in hypothesis]

print("Most specific hypothesis:", hypothesis)
------------------------------------------------------------------------------
training_data.csv
Sky,Temp,Humidity,Wind,Water,Forecast,EnjoySport Sunny,Warm,Normal,Strong,Warm,Same,Yes Sunny,Warm,High,Strong,Warm,Same,Yes Rainy,Cold,High,Strong,Warm,Change,No Sunny,Warm,High,Strong,Cool,Change,Yes



