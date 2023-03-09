#Creacion, entrenamiento, feature engineering y testing para un Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import data_processing

#Creamos un modelo generico
rfc = RandomForestClassifier()

#Definimos el modelo en base a los parametros solicitados a usuario en text_ui.py
def RFC_model(n_estimators, max_depth):
  global rfc

  rfc.set_params(n_estimators=int(n_estimators))
  rfc.set_params(max_depth=int(max_depth))

#Se entrena el modelo en base a los datos aportados por data_processing
def RFC_train():
  global rfc

  # Entrenamiento del modelo en base a los datos proporcionados por el usuario
  rfc.fit(data_processing.X_train, data_processing.y_train)

def RFC_test():
  global rfc

  #Realizamos predicciones sobre el set de test
  predictions = rfc.predict(data_processing.X_test)

  #Medida de precision
  accuracy = accuracy_score(data_processing.y_test, predictions)

  #Matriz de confusion
  conf_matrix = confusion_matrix(data_processing.y_test, predictions)

  return accuracy, conf_matrix