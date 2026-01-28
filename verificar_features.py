import joblib
import pandas as pd
import numpy as np

modelo = joblib.load('models/modelo_entrenado.pkl')
print("Clases:", modelo.classes_)
if hasattr(modelo, 'feature_names_in_'):
    print("Características esperadas:", modelo.feature_names_in_)
else:
    print("El modelo no guardó los nombres de características.")
    
# Cargar info_flask.json para ver el orden de features_modelo
import json
with open('models/info_flask.json', 'r') as f:
    info = json.load(f)
print("Features en info_flask.json:", info['features_para_modelo'])