import pandas as pd
from faker import Faker


# EJERCICIO 1
# Function
def filter_dataframe(dataframe, col_name, val):
    filtered_df = dataframe.loc[dataframe[col_name] > val]
    return filtered_df


# Create example dataframe with 4 columns.
df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Region': ['NA', 'LATAM', 'NA', 'EMEA', 'APAC'],
    'Country': ['US', 'MX', 'CA', 'RO', 'PH'],
    'Value': [1.5, 2.5, 0.5, 3.2, 1.2]

})
print('This is the unfiltered df.')
print(df)

# Call function and print result
print('\n This is the filtered df.')
print(
    filter_dataframe(df, 'Value', 1.4)
)




