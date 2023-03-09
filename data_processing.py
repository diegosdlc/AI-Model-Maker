#Preprocesador de datos para entrenamiento de modelos
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.DataFrame()
X = pd.DataFrame()
y = pd.DataFrame()
X_train = pd.DataFrame()
y_train = pd.DataFrame()
X_test = pd.DataFrame()
y_test = pd.DataFrame()

#importar datos de entrenamiento
def import_data(data_path, goal_column):
    global data
    data = pd.read_csv(data_path, sep=';')

    #Separar los datos de entrenamiento de los resultados conocidos
    global X
    global y
    X = data.drop(goal_column, axis='columns')
    y = data[goal_column]
def split_data(test_size):
    global X_train
    global X_test
    global y_train
    global y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size))
