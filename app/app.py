from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, LabelEncoder

import joblib
import numpy as np
import pandas as pd

# Cargar el modelo
model = joblib.load('app/final/random_forest_model.pkl')

# Inicializar Flask
app = Flask(__name__)

# Lista de características en el mismo orden que durante el entrenamiento
feature_columns = [
    'fami_educacionpadre_Educación profesional completa',
    'fami_educacionpadre_Educación profesional incompleta',
    'fami_educacionpadre_Ninguno',
    'fami_educacionpadre_No Aplica',
    'fami_educacionpadre_No sabe',
    'fami_educacionpadre_Postgrado',
    'fami_educacionpadre_Primaria completa',
    'fami_educacionpadre_Primaria incompleta',
    'fami_educacionpadre_Secundaria (Bachillerato) completa',
    'fami_educacionpadre_Secundaria (Bachillerato) incompleta',
    'fami_educacionpadre_Técnica o tecnológica completa',
    'fami_educacionpadre_Técnica o tecnológica incompleta',
    'fami_tienecomputador_No',
    'fami_tienecomputador_Si',
    'fami_tieneinternet_No',
    'fami_tieneinternet_Si',
    'fami_tieneautomovil_No',
    'fami_tieneautomovil_Si',
    'fami_estratovivienda_Estrato 1',
    'fami_estratovivienda_Estrato 2',
    'fami_estratovivienda_Estrato 3',
    'fami_estratovivienda_Estrato 4',
    'fami_estratovivienda_Estrato 5',
    'fami_estratovivienda_Estrato 6',
    'fami_estratovivienda_Sin Estrato',
    'fami_educacionmadre_Educación profesional completa',
    'fami_educacionmadre_Educación profesional incompleta',
    'fami_educacionmadre_Ninguno',
    'fami_educacionmadre_No Aplica',
    'fami_educacionmadre_No sabe',
    'fami_educacionmadre_Postgrado',
    'fami_educacionmadre_Primaria completa',
    'fami_educacionmadre_Primaria incompleta',
    'fami_educacionmadre_Secundaria (Bachillerato) completa',
    'fami_educacionmadre_Secundaria (Bachillerato) incompleta',
    'fami_educacionmadre_Técnica o tecnológica completa',
    'fami_educacionmadre_Técnica o tecnológica incompleta',
    'fami_tienelavadora_No',
    'fami_tienelavadora_Si',
    'estu_genero_F',
    'estu_genero_M'
]
@app.route('/', methods=['GET'])
def health():
    return jsonify('True')
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del request
    data = request.get_json(force=True)
    
    # Convertir datos a un DataFrame asegurando las columnas correctas
    df = pd.DataFrame(data, index=[0])  # Cambiar el índice a [0]
    
    # Asegurarse de que las columnas estén en el mismo orden que durante el entrenamiento
    df = df[feature_columns]
    
    # Preprocesar los datos de la misma manera que los datos de entrenamiento
    df = df.applymap(lambda x: 1 if x is True else 0 if x is False else x)
    
    # Normalizar los datos
    scaler = StandardScaler()
    nuevos_datos_normalizados = scaler.fit_transform(df)
    
    # Hacer predicciones
    prediction = model.predict(nuevos_datos_normalizados)
    
    # Transformar las predicciones numéricas a etiquetas originales si es necesario
    label_encoder = LabelEncoder()
    label_encoder.fit(['bajo', 'medio', 'alto'])
    predicciones_etiquetas = label_encoder.inverse_transform(prediction)
    
    # Convertir la predicción de nuevo a formato JSON
    result = {'prediction': predicciones_etiquetas.tolist()}  # Convertir a lista
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
