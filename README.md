# Examen Nivel Intermedio Juan Coronado

## Descripción
Bienvenido/a al ReadMe del examen nivel intermedio en donde se realizaron los 8 ejercicios solicitados. Para una explicación clara de lo que se hizo en cada ejercicio, favor de ver la información a continuación.

___
## Tecnologías utilizadas
Para crear los siguientes ejercicios, se utilizó el IDE PyCharm y las siguientes librerías:
* Pandas
* Numpy
* Faker
* Sklearn

___
## Descripción de los ejercicios realizados

### Ejercicio 1: Filtrar DataFrame con Pandas.

#### Descripción:

Implementa una función llamada ``filter_dataframe`` que recibe un DataFrame, el nombre de una columna, y un valor de umbral. La función devuelve un DataFrame filtrado donde los valores en la columna especificada son mayores que el umbral.

#### Ejemplo:

Se crea un DataFrame con 4 columnas: ID, Region, Country y Price, y se define el umbral en la variable de Value. Al llamar la función, se especifica que se utilizará este DataFrame, la columna específica llamada Price, y la variable de Value para llevar acabo el filtrado del DataFrame.

Con esta información, la función utiliza .loc para evaluar cuales filas tienen un número mayor en la columna que la variable declarada y regresa un DataFrame filtrado.

Se usa print() para primero mostrar el DataFrame sin filtrar y posteriormente el DataFrame filtrado.

### Ejercicio 2: Generar datos para regresión.

#### Descripción:

Crea una función  ``generate_regression_data`` que simule un conjunto de datos para un problema de regresión. Utiliza la librería **Faker** para generar datos numéricos aleatorios. Debe devolver un DataFrame con variables independientes y una Serie con la variable dependiente.

#### Ejemplo:

Se crea una instancia de Faker y se declara un número de muestras en la variable n_samples. Al llamar la función, se especifica la variable de n_samples para saber cuantas muestras tendrá el DataFrame resultante.

Con esta información, la función crea un DataFrame con 4 columnas: ID, Independent_Var1, Independent_Var2 y Dependant_Var. La primera columna es un secuencial y para las demás se utiliza fake.random_number para generar un número aleatorio de 2 dígitos. Esto se encuentra dentro de un For loop, que se repite de acorde a lo declarado en la variable de n_samples. La función regresa el DataFrame generado.

Se usa print() para mostrar el DataFrame resultante.

### Ejercicio 3: Entrenar modelo de regresión múltiple.

#### Descripción:

Implementa una función  ``train_multiple_linear_regression`` que entrene un modelo de **regresión lineal múltiple** utilizando los datos simulados. La función debe devolver el modelo entrenado.

#### Ejemplo:

Utilizando el DataFrame del ejercicio anterior, se selecciona cual va a ser la variable X (independientes) y cual la variable y (dependiente). Al llamar la función, se especifica las variables X y y.

Con esta información, la función separa los datos en subconjuntos para entrenamiento y para testing, lo cual son las variables X_train, X_test, y_train, y_test utilizando train_test_split. Después se crea y entrena al modelo con los subconjuntos de entrenamiento, se hacen predicciones con el subconjunto para testing y finalmente se crean métricas para evaluar el modelo. La función regresa el modelo entrenado.

Se usa print() para mostrar tres medidas de evaluación del modelo: MSE, RMSE y R2.

* **Ejercicio 4: List comprehension anidado.**

#### Descripción:

Crea una función  ``flatten_list`` que tome una lista de listas y la convierta en una lista plana utilizando **list comprehensions anidados**.

#### Ejemplo:

Se crea una variable llamada list_of_lists, que es una lista de listas. Al llamar la función, se especifica la variable list_of_lists.

Con esta información, la función crea una lista vacía primero y utilizando un for loop, utiliza .extend para agregar cada elemento a la nueva lista. La función regresa la lista aplanada.

Se usa print() para mostrar primero la lista de listas y posteriormente la lista aplanada.

* **Ejercicio 5: Agrupar y agregar con Pandas.**

Descripcion


* **Ejercicio 6: Modelo de clasificación logística.**

Descripcion


* **Ejercicio 7: Aplicar función a una columna con Pandas.**

Descripcion


* **Ejercicio 8: Comprehensions con condiciones.**

Descripcion

___
## Contacto

Para cualquier aclaración o duda, favor de contactarse con Juan Coronado, al siguiente correo: jjcoronado91@gmail.com.