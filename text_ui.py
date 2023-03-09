#Interfaz de usuario basada en texto

import RFC_logic
import data_processing

print('MODEL MAKER V 1.0')

#Solicita los datos para el entrenamiento
data_path = input('Introduzca ruta de datos de entrenamiento: ')
goal_column = input('Introduzca columna sobre la que desea realizar predicciones: ')

data_processing.import_data(data_path, goal_column)

#Especifica que cantidad de los datos se destinan al testeo
test_size = input('Especifique tamaño del set de test (0 al 1): ')

data_processing.split_data(test_size)

print('IMPORTACION DE DATOS REALIZADA CON EXITO')

#Seleccion del modelo
print("""Seleccione modelo:
1: Random Forest Classifier
2: Regresion logistica
3: Regresion lineal
""")
model = int(input('Introduzca seleccion: '))

if model == 1:
    print('SELECCIONADO MODELO DE CLASIFICACION RFC. Seleccion de parametros')
    # Parametros para modelo RFC
    n_estimators = input('Seleccione numero de arboles: ')
    max_depth = input('Seleccione profundidad de cada arbol: ')
    #Creacion del modelo con los parametros seleccionados
    RFC_logic.RFC_model(n_estimators, max_depth)

    #Entrenamiento del modelo con los datos aportados
    RFC_logic.RFC_train()

    print('ENTRENAMIENTO TERMINADO.')

    print('RESULTADOS DEL MODELO')
    #Testeo basico de funcionamiento
    accuracy, conf_matrix = RFC_logic.RFC_test()

    print('Precision del modelo: ' + str(accuracy))
    print('Matriz de confusion: \n')
    print(conf_matrix)

elif model == 2:
    print('SELECCIONADO MODELO DE CLASIFICACION LINEAL. Proximamente')
elif model == 3:
    print('SELECCIONADO MODELO DE REGRESION LINEAL. Proximamente')
else:
    print('ERROR: seleccion no valida')


