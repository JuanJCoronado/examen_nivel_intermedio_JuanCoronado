import pandas as pd
from faker import Faker


# EJERCICIO 1: Filtrar DataFrame con Pandas
# Function
def filter_dataframe(dataframe, col, val):
    filtered_df = dataframe.loc[dataframe[col] > val]
    return filtered_df


# Create sample dataframe with 4 columns.
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


# Ejercicio 2
# Function
# ________
def generate_regression_data(n_samples):
    return 223


fake = Faker()
n_samples = 3
for i in range(n_samples):
    print(
        fake.random_number(digits=2)
    )


# _________

# EJERCICIO 3


# _________

# EJERCICIO 4: List comprehension anidado
# Function
def flatten_list(list_of_lists):
    flat_list = []
    for x in list_of_lists:
        flat_list.extend(x)
    return flat_list


# Create sample list of lists
list_of_lists = [['a', 'b', 'c'], [1, 2, 3], ['x', 'y', 'z'], [7, 8, 9]]
print('\n This is a sample list of lists.')
print(list_of_lists)

# Call function and print result
print('\n This is a flattened list.')
print(
    flatten_list(list_of_lists)
)


# EJERCICIO 5: Agrupar y agregar con Pandas
# FUNCTION
def group_and_aggregate(dataframe, col):
    grouped_df = df.groupby(col)['Price'].mean()
    return grouped_df


# Create sample dataframe with 4 columns.
df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Region': ['LATAM', 'LATAM', 'LATAM', 'EMEA', 'EMEA'],
    'Country': ['CO', 'MX', 'PE', 'RO', 'GE'],
    'Price': [200, 300, 100, 200, 400]
})

print('\n This is the sample df.')
print(df)

# Call function and print result
print('\n This is the grouped and aggregated df.')
print(
    group_and_aggregate(df, 'Region')
)


# EJERCICIO 6: Modelo de clasificacion logistica


# EJERCICIO 7: Aplicar funcion una columna con Pandas
# Function
def apply_function_to_column(dataframe, col):
    def modify_salary(row):
        if row[col] == 'John':
            return row['Salary'] * 1.1
        elif row[col] == 'Erin' or row[col] == 'Danielle':
            return row['Salary'] * 1.05
        return row['Salary']

    dataframe['Salary'] = dataframe.apply(modify_salary, axis=1)


# Create sample dataframe with 3 columns.
df = pd.DataFrame({
    'ID': [1, 2, 3],
    'Name': ['John', 'Erin', 'Danielle'],
    'Salary': [50000, 70000, 80000]
})

print('\n This is the sample df.')
print(df)

# Call function and print result
print('\n This is the modified df.')
apply_function_to_column(df, 'Name')
print(df)

# EJERCICIO 8: