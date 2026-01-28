from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import json
import os

# Importar módulos propios
from preprocess import cargar_datos, obtener_info_dataset
from model import cargar_modelo, cargar_escalador, predecir, obtener_metricas
from visualize import crear_grafica_distribucion, crear_matriz_confusion, crear_grafica_importancia

app = Flask(__name__)
app.secret_key = 'clave_secreta_para_sesiones'  # Cambia esto

# Cargar datos y modelo al iniciar
print("🔄 Cargando dataset y modelo...")
df = cargar_datos('data/dataset_limpio.csv')
modelo = cargar_modelo('models/modelo_entrenado.pkl')
escalador = cargar_escalador('models/scaler.pkl')

# Cargar información del JSON
with open('models/info_flask.json', 'r') as f:
    info_dataset = json.load(f)

# Variables globales (se pueden guardar en sesión)
target_col = info_dataset.get('variable_objetivo', 'target')
X_train = None
X_test = None
y_train = None
y_test = None

@app.route('/')
def index():
    """Página principal con información del dataset"""
    dataset_info = obtener_info_dataset(df)
    
    # Obtener ejemplo de fila para el formulario
    ejemplo_fila = df.iloc[0].to_dict() if not df.empty else {}
    
    return render_template('index.html', 
                          dataset_info=dataset_info,
                          ejemplo_fila=ejemplo_fila,
                          target_col=target_col)

@app.route('/configurar', methods=['POST'])
def configurar():
    """Configura la muestra de datos según parámetros del usuario"""
    data = request.json
    
    semilla = data.get('semilla', 42)
    porcentaje = float(data.get('porcentaje', 1.0))
    split = float(data.get('split', 0.8))
    
    # 1. Tomar muestra del dataset
    if porcentaje < 1.0:
        df_muestra = df.sample(frac=porcentaje, random_state=semilla)
    else:
        df_muestra = df
    
    # 2. Preparar para modelo
    from sklearn.model_selection import train_test_split
    
    X = df_muestra.drop(columns=[target_col])
    y = df_muestra[target_col]
    
    # Codificar categóricas si es necesario
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # 3. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-split, random_state=semilla
    )
    
    # Guardar en variables globales (o en sesión)
    global global_X_train, global_X_test, global_y_train, global_y_test
    global_X_train, global_X_test = X_train, X_test
    global_y_train, global_y_test = y_train, y_test
    
    return jsonify({
        'mensaje': '✅ Configuración aplicada',
        'muestra_filas': len(df_muestra),
        'entrenamiento_filas': len(X_train),
        'prueba_filas': len(X_test)
    })

@app.route('/evaluar', methods=['GET'])
def evaluar():
    """Evalúa el modelo y genera gráficas"""
    if global_X_test is None or global_y_test is None:
        return jsonify({'error': 'Primero configura los datos'}), 400
    
    # Realizar predicciones
    y_pred = predecir(modelo, global_X_test, escalador)
    
    # Calcular métricas
    metricas = obtener_metricas(global_y_test, y_pred)
    
    # Generar gráficas
    matriz_conf = crear_matriz_confusion(global_y_test, y_pred)
    
    # Gráfica de distribución de la variable objetivo
    grafica_dist = crear_grafica_distribucion(
        df, 
        info_dataset.get('columnas_numericas', [df.columns[0]])[0],
        target_col
    )
    
    # Gráfica de importancia (si aplica)
    grafica_imp = crear_grafica_importancia(
        modelo, 
        global_X_test.columns.tolist() if hasattr(global_X_test, 'columns') else []
    )
    
    return jsonify({
        'metricas': metricas,
        'matriz_confusion': matriz_conf,
        'grafica_distribucion': grafica_dist,
        'grafica_importancia': grafica_imp
    })

@app.route('/predecir_manual', methods=['POST'])
def predecir_manual():
    """Recibe datos manuales y devuelve predicción"""
    datos = request.json
    
    # Crear DataFrame con los datos recibidos
    datos_df = pd.DataFrame([datos])
    
    # Asegurar que tenga las mismas columnas que el modelo espera
    # Esto puede requerir transformaciones adicionales
    # Por ahora, asumimos que los datos vienen en el formato correcto
    
    prediccion = predecir(modelo, datos_df, escalador)
    
    return jsonify({
        'prediccion': int(prediccion[0]),
        'clase': str(prediccion[0])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)