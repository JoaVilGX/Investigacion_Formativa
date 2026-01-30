import joblib

modelo = joblib.load('models/modelo_entrenado.pkl')
print("Clases del modelo:", modelo.classes_)
print("Tipo de clases:", type(modelo.classes_[0]))