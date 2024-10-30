import pandas as pd
import numpy as np
from faker import Faker
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
print('This is a sample df.')
print(df)

# Call function and print result
print('\n This is a filtered df.')
print(
    filter_dataframe(df, 'Value', 1.4)
)


# EJERCICIO 2: Generar datos para regresion
# Function
def generate_regression_data(n_samples):
    data = pd.DataFrame()
    for i in range(0, n_samples):
        data.loc[i, 'ID'] = i+1
        data.loc[i, 'Independent_Var1'] = fake.random_number(digits=2)
        data.loc[i, 'Independent_Var2'] = fake.random_number(digits=2)
        data.loc[i, 'Dependant_Var'] = fake.random_number(digits=2)
    return data


# Create faker instance and define sample size
fake = Faker()
n_sample = 20

# Call function and print result
print('\n This is a sample df generated using Faker.')
data = generate_regression_data(n_sample)
print(data)


# EJERCICIO 3: Entrenar modelo de regresion multiple.
# Function
def train_multiple_linear_regression():
    return 3


X = data[['Independent_Var1']]
y = data[['Dependant_Var']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)

# Evaluating the model:
print(mean_squared_error(y_test, predictions)) # mean squared error (MSE)
print(np.sqrt(mean_squared_error(y_test, predictions))) # root mean square error (RMSE)
print(model.score(X_test, y_test))
print(r2_score(y_test, predictions))

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

print('\n This is a sample df.')
print(df)

# Call function and print result
print('\n This is a grouped and aggregated df.')
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

print('\n This is a sample df.')
print(df)

# Call function and print result
print('\n This is a modified column from the df.')
apply_function_to_column(df, 'Name')
print(df)


# EJERCICIO 8: Comprehensions con condiciones
# Function
def filter_and_square(number_list):
    new_list = []
    for x in number_list:
        if x > 5:
            new_list.append(x)

    return [x ** 2 for x in new_list]


# Create sample list
n_list = [4, 6, 8, 3, 9]

print('\n This is a sample list.')
print(n_list)

# Call function and print result
print('\n This is a filtered and squared list.')
print(
    filter_and_square(n_list)
)
