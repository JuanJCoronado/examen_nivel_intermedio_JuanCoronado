import pandas as pd
import numpy as np
from faker import Faker
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report


# FUNCTIONS
# Function 1
def filter_dataframe(dataframe, column, val):
    filtered_df = dataframe.loc[dataframe[column] > val]
    return filtered_df


# Function 2
def generate_regression_data(num_samples):
    data = pd.DataFrame()
    for i in range(0, num_samples):
        data.loc[i, 'ID'] = i+1
        data.loc[i, 'Independent_Var1'] = fake.random_number(digits=2)
        data.loc[i, 'Independent_Var2'] = fake.random_number(digits=2)
        data.loc[i, 'Dependant_Var'] = fake.random_number(digits=2)
    return data


# Function 3
def train_multiple_linear_regression(dataframe, independent_vars, dependent_var):
    X = independent_vars
    y = dependent_var
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluate the model
    print('\n These are the metrics that evaluate the multiple regression model.')
    print('Mean squared error (MSE): ', mean_squared_error(y_test, predictions))
    print('Root mean squared error (RMSE): ', np.sqrt(mean_squared_error(y_test, predictions)))
    print('R squared (R2): ', r2_score(y_test, predictions))

    return model


# Function 4
def flatten_list(list_of_lists):
    flat_list = []
    for x in list_of_lists:
        flat_list.extend(x)
    return flat_list


# Function 5
def group_and_aggregate(dataframe, col_to_group, col_to_aggregate):
    grouped_df = dataframe.groupby(col_to_group)[col_to_aggregate].mean()
    return grouped_df


# Function 6
def train_logistic_regression(independent_var, dependant_var):
    X = independent_var
    y = dependant_var
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Create new df.
    new_df = pd.DataFrame({
        'glucose': X_test['Glucose'],
        'outcome': y_test,
        'prob': model.predict_proba(X_test)[:, 1]
    })
    new_df = new_df.sort_values('prob')
    new_df['predictions'] = predictions

    # Print the test df with prob and pred.
    print('\n This is the df that contains the test data used, along with probabilities and predictions.')
    print(new_df)

    # Evaluate the model.
    print('\n These are metrics that evaluate the logistic regression model.')
    print('Accuracy', model.score(X_test, y_test))
    print('\n Classification report')
    print(classification_report(y_test, model.predict(X_test)))

    return model


# Function 7
def apply_function_to_column(dataframe, col):
    def modify_salary(row):
        if row[col] == 'John':
            return row['Salary'] * 1.1
        elif row[col] == 'Erin' or row[col] == 'Danielle':
            return row['Salary'] * 1.05
        return row['Salary']

    dataframe['Salary'] = dataframe.apply(modify_salary, axis=1)


# Function 8
def filter_and_square(number_list):
    new_list = []
    for x in number_list:
        if x > 5:
            new_list.append(x)
    return [x ** 2 for x in new_list]


# _____________________________________________________
# EJERCICIOS
# EJERCICIO 1: Filtrar DataFrame con Pandas.
# Create sample df and define value to filter.
df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Region': ['NA', 'LATAM', 'NA', 'EMEA', 'APAC'],
    'Country': ['US', 'MX', 'CA', 'RO', 'PH'],
    'Price': [10.5, 20.5, 5.7, 30.2, 13.2]
})
Value = 11.1

# Print sample df.
print('This is a sample df.')
print(df)

# Call function and print filtered df.
print('\n This is a filtered df.')
print(filter_dataframe(df, 'Price', Value))


# EJERCICIO 2: Generar datos para regresion.
# Create faker instance and define sample size.
fake = Faker()
Faker.seed(0)
n_samples = 500

# Call function and print generated fake data df.
print('\n This is a sample df generated using Faker.')
data = generate_regression_data(n_samples)
print(data)


# EJERCICIO 3: Entrenar modelo de regresión multiple.
# Define independent and dependant variables from previous df.
X = data[['Independent_Var1', 'Independent_Var2']]
y = data[['Dependant_Var']]

# Call function.
train_multiple_linear_regression(data, X, y)


# EJERCICIO 4: List comprehension anidado.
# Create sample list of lists
listOfLists = [['a', 'b', 'c'], [1, 2, 3], ['x', 'y', 'z'], [7, 8, 9]]

# Print sample list of lists.
print('\n This is a sample list of lists.')
print(listOfLists)

# Call function and print new flattened list.
print('\n This is a flattened list.')
print(flatten_list(listOfLists))


# EJERCICIO 5: Agrupar y agregar con Pandas.
# Create sample df.
df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Region': ['LATAM', 'LATAM', 'LATAM', 'EMEA', 'EMEA'],
    'Country': ['CO', 'MX', 'PE', 'RO', 'GE'],
    'Price': [200, 300, 100, 200, 400]
})

# Print sample df
print('\n This is a sample df.')
print(df)

# Call function and print grouped and aggregated df.
print('\n This is a grouped and aggregated df.')
print(group_and_aggregate(df, 'Region', 'Price'))


# EJERCICIO 6: Modelo de clasificacion logistica.
# Load data from diabetes.csv and print sample df.
diabetes_df = pd.read_csv('diabetes.csv')
print('\n This is a sample diabetes df')
print(diabetes_df)

# Assign variables.
X = diabetes_df[['Glucose']]
y = diabetes_df['Outcome']

# Call function.
train_logistic_regression(X, y)

# EJERCICIO 7: Aplicar funcion una columna con Pandas.
# Create sample df.
df = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['John', 'Erin', 'Danielle', 'Chris'],
    'Salary': [50000, 70000, 80000, 90000]
})

# Print sample df
print('\n This is a sample df.')
print(df)

# Call function and print df with modified column.
print('\n This is a modified column from the df.')
apply_function_to_column(df, 'Name')
print(df)


# EJERCICIO 8: Comprehensions con condiciones.
# Create sample list
n_list = [4, 6, 8, 3, 9]

# Print sample list
print('\n This is a sample list.')
print(n_list)

# Call function and print filtered and squared list.
print('\n This is a filtered and squared list.')
print(filter_and_square(n_list))
