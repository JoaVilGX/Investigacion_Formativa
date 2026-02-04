from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURACIÓN DE IMPORTS Y PATHS
# ============================================================================

sys.path.append('.')

# Importamos módulos propios
try:
    from preprocess import (
        cargar_datos, 
        limpiar_datos, 
        transformar_categoricas, 
        estandarizar_numericas,
        obtener_info_dataset,
        preparar_datos_para_modelo_sin_duplicados,
        asegurar_caracteristicas
    )
    from model import (
        cargar_modelo, 
        cargar_escalador, 
        predecir_con_preprocesamiento,
        obtener_metricas,
        obtener_probabilidades
    )
    from visualize import (
        crear_grafica_distribucion,
        crear_grafica_rendimiento_por_clase,
        crear_matriz_confusion,
        crear_grafica_importancia
    )
    print("✅ Todos los módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    # Definimos placeholders
    cargar_datos = limpiar_datos = transformar_categoricas = estandarizar_numericas = None
    obtener_info_dataset = preparar_datos_para_modelo_sin_duplicados = asegurar_caracteristicas = None
    cargar_modelo = cargar_escalador = predecir_con_preprocesamiento = obtener_metricas = obtener_probabilidades = None
    crear_grafica_distribucion = crear_matriz_confusion = crear_grafica_importancia = None

# ============================================================================
# INICIALIZACIÓN DE FLASK
# ============================================================================

app = Flask(__name__)
app.secret_key = 'clave_secreta_para_sesiones_12345'
CORS(app)

# ============================================================================
# CONFIGURACIÓN GLOBAL - CARGA DE DATOS Y MODELO
# ============================================================================

print("\n" + "="*60)
print("🚀 INICIALIZANDO SISTEMA DE PREDICCIÓN")
print("="*60)

# 1. CARGAMOS LA CONFIGURACIÓN
try:
    with open('models/info_flask.json', 'r') as f:
        info_dataset = json.load(f)
    features_modelo = info_dataset['features_para_modelo']
    target_col = info_dataset['variable_objetivo']
    accuracy_entrenamiento = info_dataset['accuracy']
    print(f"✅ Configuración cargada: {len(features_modelo)} características")
    print(f"   Modelo: {info_dataset['modelo_utilizado']}")
    print(f"   Accuracy: {accuracy_entrenamiento:.4f}")
except Exception as e:
    print(f"❌ Error cargando configuración: {e}")
    print("⚠️  Usando configuración por defecto...")
    info_dataset = {
        'nombre_dataset': 'car_price_cleaned.csv',
        'variable_objetivo': 'Condition',
        'modelo_utilizado': 'Regresión Logística',
        'accuracy': 0,
        'features_para_modelo': []
    }
    features_modelo = []
    target_col = 'Condition'
# 1. CARGAMOS DATOS PREPROCESADOS
try:
    print("\n🔄 Preparando datos...")
    df_preprocesado, condition_map, brand_mapping, scalers = preparar_datos_para_modelo_sin_duplicados()
    
    print(f"✅ Datos preparados: {df_preprocesado.shape}")
    print(f"   - Filas: {df_preprocesado.shape[0]}")
    print(f"   - Columnas: {df_preprocesado.shape[1]}")
    
    # Verificamos que tenemos Condition_encoded
    if 'Condition_encoded' not in df_preprocesado.columns:
        print("⚠️  Creando Condition_encoded...")
        df_preprocesado['Condition_encoded'] = df_preprocesado[target_col].map(
            {'New': 'New', 'Like New': 'Like New', 'Used': 'Used'}
        )
    
except Exception as e:
    print(f"❌ Error preparando datos: {e}")
    df_preprocesado = None

# 3. CARGAMOS MODELO ENTRENADO
try:
    modelo = cargar_modelo('models/modelo_entrenado.pkl')
    print("✅ Modelo cargado exitosamente")
    
    # VERIFICAMOS QUÉ CLASES TIENE EL MODELO
    if hasattr(modelo, 'classes_'):
        print(f"📊 Clases del modelo: {modelo.classes_}")
        print(f"📊 Tipo de clases: {type(modelo.classes_[0])}")
    
    # VERIFICAMOS LOS NOMBRES DE CARACTERÍSTICAS QUE EL MODELO ESPERA
    if hasattr(modelo, 'feature_names_in_'):
        print(f"🔤 Características esperadas por el modelo: {modelo.feature_names_in_}")
    else:
        print("⚠️  El modelo no tiene atributo 'feature_names_in_'. Asegúrate de usar scikit-learn >= 1.0.")
        
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    print("⚠️  Algunas funciones no estarán disponibles")
    modelo = None

# 4. CARGAMOS DATOS ORIGINALES PARA GRÁFICAS
try:
    df_original = cargar_datos('data/car_price_cleaned.csv')
    print(f"✅ Datos originales cargados: {df_original.shape}")
except Exception as e:
    print(f"⚠️  Error cargando datos originales: {e}")
    df_original = None

# Variables globales
global_X_train = None
global_X_test = None
global_y_train = None
global_y_test = None

escalador = None  # Placeholder en caso de necesitar

print("="*60)
print("✅ INICIALIZACIÓN COMPLETADA")
print("="*60 + "\n")

# ============================================================================
# RUTAS PRINCIPALES
# ============================================================================

@app.route('/')
def index():
    """Página principal con información del dataset"""
    # Preparamos información para mostrar
    if df_preprocesado is not None:
        dataset_info = obtener_info_dataset(df_preprocesado)
        info_display = {
            'nombre': info_dataset.get('nombre_dataset', 'car_price_cleaned.csv'),
            'filas': dataset_info.get('filas', 0),
            'columnas': dataset_info.get('columnas', 0),
            'caracteristicas': len(features_modelo),
            'modelo': info_dataset.get('modelo_utilizado', 'Desconocido'),
            'accuracy': round(info_dataset.get('accuracy', 0), 4)
        }
    else:
        info_display = {
            'nombre': 'No disponible',
            'filas': 0,
            'columnas': 0,
            'caracteristicas': 0,
            'modelo': 'No disponible',
            'accuracy': 0
        }
    
    return render_template(
        'index.html',
        dataset_info=info_display,
        target_col=target_col
    )

@app.route('/configurar', methods=['POST'])
def configurar():
    global global_X_train, global_X_test, global_y_train, global_y_test
    
    if df_preprocesado is None:
        return jsonify({'error': '❌ Dataset no disponible. Verifique la carga de datos.'}), 400
    
    data = request.json
    
    try:
        semilla = int(data.get('semilla', 42))
        porcentaje = float(data.get('porcentaje', 1.0))
        split = float(data.get('split', 0.8))
        
        print(f"⚙️ Configurando: semilla={semilla}, porcentaje={porcentaje}, split={split}")
        
        # 1. Tomamos muestra del dataset preprocesado
        if porcentaje < 1.0:
            df_muestra = df_preprocesado.sample(frac=porcentaje, random_state=semilla)
        else:
            df_muestra = df_preprocesado.copy()
        
        print(f"📊 Muestra seleccionada: {len(df_muestra)} filas")
        
        # 2. Separamos características y variable objetivo
        # Determinamos el orden de características que el modelo espera
        if modelo is not None and hasattr(modelo, 'feature_names_in_'):
            features_ordenadas = list(modelo.feature_names_in_)
        else:
            features_ordenadas = features_modelo
        
        X = df_muestra[features_ordenadas]
        y = df_muestra['Condition_encoded']
        
        # 3. Dividimos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=1-split, 
            random_state=semilla, 
            stratify=y
        )
        
        # Guardamos en variables globales
        global_X_train = X_train
        global_X_test = X_test
        global_y_train = y_train
        global_y_test = y_test
        
        print(f"✅ División creada: Entrenamiento={len(X_train)}, Prueba={len(X_test)}")
        print(f"   Características en X_test: {list(X_test.columns)}")
        
        return jsonify({
            'success': True,
            'mensaje': '✅ Configuración aplicada correctamente',
            'muestra_filas': len(df_muestra),
            'entrenamiento_filas': len(X_train),
            'prueba_filas': len(X_test)
        }), 200
        
    except Exception as e:
        print(f"❌ Error en configurar: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'❌ Error: {str(e)}'}), 500

@app.route('/evaluar', methods=['GET'])
def evaluar():
    """Evalúa el modelo y genera gráficas"""
    global global_X_test, global_y_test
    
    print("📊 Iniciando evaluación del modelo...")
    
    # Validaciones
    if global_X_test is None or global_y_test is None:
        return jsonify({
            'success': False,
            'error': '❌ Primero debe configurar los datos usando "Aplicar Configuración"'
        }), 400
    
    if modelo is None:
        return jsonify({
            'success': False,
            'error': '❌ Modelo no disponible. Verifique la carga del modelo.'
        }), 400
    
    try:
        # 1. REALIZAMOS PREDICCIONES
        print("🤖 Realizando predicciones...")
        y_pred = predecir_con_preprocesamiento(
            modelo, 
            global_X_test, 
            scalers=scalers,
            features_requeridas=features_modelo
        )
        
        if y_pred is None:
            return jsonify({
                'success': False,
                'error': '❌ No se pudieron realizar las predicciones'
            }), 400
        
        print(f"✅ {len(y_pred)} predicciones realizadas")
        
        # 2. CALCULAMOS LAS MÉTRICAS
        print("📈 Calculando métricas...")
        metricas = obtener_metricas(global_y_test, y_pred)
        
        if metricas is None:
            return jsonify({
                'success': False,
                'error': '❌ No se pudieron calcular las métricas'
            }), 400
        
        print(f"✅ Métricas calculadas (Accuracy: {metricas['accuracy']:.4f})")
        
        print("🔍 DIAGNÓSTICO DE ETIQUETAS:")
        print(f"Tipo de global_y_test: {type(global_y_test)}")
        print(f"Tipo de y_pred: {type(y_pred)}")

        if hasattr(global_y_test, '__len__'):
            print(f"Primeros 10 valores de global_y_test: {list(global_y_test[:10])}")
        if hasattr(y_pred, '__len__'):
            print(f"Primeros 10 valores de y_pred: {list(y_pred[:10])}")

        # Contar valores únicos
        unique_y_test = set(str(x) for x in global_y_test)
        unique_y_pred = set(str(x) for x in y_pred)
        print(f"Valores únicos en global_y_test (como strings): {unique_y_test}")
        print(f"Valores únicos en y_pred (como strings): {unique_y_pred}")

        # Ver cuáles NO están en etiquetas_validas
        etiquetas_validas_set = set(['Used', 'Like New', 'New', '0', '1', '2'])
        invalid_y_test = [x for x in unique_y_test if str(x) not in etiquetas_validas_set]
        invalid_y_pred = [x for x in unique_y_pred if str(x) not in etiquetas_validas_set]

        print(f"Valores NO VÁLIDOS en global_y_test: {invalid_y_test}")
        print(f"Valores NO VÁLIDOS en y_pred: {invalid_y_pred}")
        
        # 3. GENERAR MATRIZ DE CONFUSIÓN
        print("📊 Generando matriz de confusión...")
        matriz_conf = None
        try:
            # Convertir a Series de pandas para facilitar manipulación
            y_test_series = pd.Series(global_y_test)
            y_pred_series = pd.Series(y_pred)
            
            print(f"📊 Datos para matriz de confusión:")
            print(f"   y_test shape: {y_test_series.shape}")
            print(f"   y_pred shape: {y_pred_series.shape}")
            print(f"   Valores únicos en y_test: {y_test_series.unique()}")
            print(f"   Valores únicos en y_pred: {y_pred_series.unique()}")
            
            # PRIMERO: Normalizar todos los valores a strings consistentes
            def normalizar_etiquetas(series):
                # Convertir a string
                series_str = series.astype(str)
                
                # Remover espacios extra, puntos decimales, etc.
                series_str = series_str.str.strip()
                
                # Mapear variantes comunes
                mapeo = {
                    '0': 'Used',
                    '1': 'Like New', 
                    '2': 'New',
                    '0.0': 'Used',
                    '1.0': 'Like New',
                    '2.0': 'New',
                    'used': 'Used',
                    'like new': 'Like New',
                    'new': 'New',
                    'Like_New': 'Like New',
                    'Like-New': 'Like New'
                }
                
                # Aplicar mapeo
                series_normalizado = series_str.replace(mapeo)
                
                # Solo mantener las 3 categorías principales
                categorias_validas = ['Used', 'Like New', 'New']
                series_normalizado = series_normalizado.where(
                    series_normalizado.isin(categorias_validas), 
                    'Used'  # Valor por defecto si no es válido
                )
                
                return series_normalizado
            
            y_test_norm = normalizar_etiquetas(y_test_series)
            y_pred_norm = normalizar_etiquetas(y_pred_series)
            
            print(f"✅ Etiquetas normalizadas:")
            print(f"   y_test_norm únicas: {y_test_norm.unique()}")
            print(f"   y_pred_norm únicas: {y_pred_norm.unique()}")
            print(f"   Conteo: {len(y_test_norm)} muestras (deberían ser {len(global_y_test)})")
            
            # Verificar que tenemos todas las muestras
            perdidas = len(y_test_series) - len(y_test_norm)
            if perdidas > 0:
                print(f"⚠️  Se perdieron {perdidas} muestras en la normalización")
            
            # Crear matriz de confusión con TODAS las muestras
            matriz_conf = crear_matriz_confusion(
                y_test_norm, 
                y_pred_norm,
                labels=['Used', 'Like New', 'New']
            )
            
            print(f"✅ Matriz de confusión generada con {len(y_test_norm)} muestras")
            
        except Exception as e:
            print(f"⚠️  Error generando matriz de confusión: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"📊 Matriz de confusión generada: {type(matriz_conf)}, longitud: {len(matriz_conf) if matriz_conf else 0}")
                
        # 4. GRÁFICA DE DISTRIBUCIÓN
        print("📈 Generando gráfica de distribución...")
        grafica_dist = None
        try:
            if df_original is not None:
                if 'Year' in df_original.columns and target_col in df_original.columns:
                    grafica_dist = crear_grafica_distribucion(
                        df_original, 
                        'Year', 
                        target_col
                    )
                    print("✅ Gráfica de distribución generada")
        except Exception as e:
            print(f"⚠️  Error generando gráfica de distribución: {e}")
        
        print(f"📈 Gráfica distribución generada: {type(grafica_dist)}, longitud: {len(grafica_dist) if grafica_dist else 0}")
        
        # 5. GRÁFICA DE RENDIMIENTO POR CLASE
        print("📊 Generando gráfica de rendimiento por clase...")
        grafica_rendimiento = None
        try:
            # Usar las versiones normalizadas
            y_test_labels = y_test_norm if 'y_test_norm' in locals() else pd.Series(global_y_test).astype(str)
            y_pred_labels = y_pred_norm if 'y_pred_norm' in locals() else pd.Series(y_pred).astype(str)
            
            print(f"📊 Datos para gráfica de rendimiento:")
            print(f"   Muestras: {len(y_test_labels)}")
            print(f"   Distribución y_test: {y_test_labels.value_counts().to_dict()}")
            print(f"   Distribución y_pred: {y_pred_labels.value_counts().to_dict()}")
            
            grafica_rendimiento = crear_grafica_rendimiento_por_clase(
                y_test_labels.tolist(), 
                y_pred_labels.tolist(),
                labels=['Used', 'Like New', 'New']
            )
            
            if grafica_rendimiento:
                print("✅ Gráfica de rendimiento por clase generada")
            else:
                print("⚠️  No se pudo generar gráfica de rendimiento")
                
        except Exception as e:
            print(f"⚠️  Error generando gráfica de rendimiento por clase: {e}")
        
        print("✅ Evaluación completada exitosamente")

        return jsonify({
            'success': True,
            'metricas': metricas,
            'matriz_confusion': matriz_conf,
            'grafica_distribucion': grafica_dist,
            'grafica_rendimiento': grafica_rendimiento,
            'resumen': {
                'accuracy': round(metricas['accuracy'], 4),
                'precision': round(metricas['precision'], 4),
                'recall': round(metricas['recall'], 4),
                'f1': round(metricas['f1'], 4),
                'muestra_prueba': len(global_X_test)
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Error en evaluación: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'❌ Error en evaluación: {str(e)}'
        }), 500

@app.route('/test_image', methods=['GET'])
def test_image():
    """Endpoint de prueba para verificar que las imágenes se generan correctamente"""
    try:
        # Creamos una imagen simple de prueba
        import plotly.graph_objects as go
        import plotly.io as pio
        import base64
        
        fig = go.Figure(data=[go.Bar(y=[1, 3, 2])])
        fig.update_layout(title='Gráfica de prueba', width=400, height=300)
        
        img_bytes = pio.to_image(fig, format='png', width=400, height=300)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Devolvemos como HTML para probar
        html = f"""
        <html>
        <body>
            <h1>Test de Imagen Base64</h1>
            <p>Longitud: {len(img_base64)} caracteres</p>
            <p>Primeros 100 chars: {img_base64[:100]}...</p>
            <h2>Imagen renderizada:</h2>
            <img src="data:image/png;base64,{img_base64}" style="border: 1px solid black;">
            <h2>En JSON:</h2>
            <textarea rows="10" cols="80">{json.dumps({'test': f'data:image/png;base64,{img_base64}'})}</textarea>
        </body>
        </html>
        """
        
        return html
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/predecir_manual', methods=['POST'])
def predecir_manual():
    """Recibe datos manuales y devuelve predicción - VERSIÓN CORREGIDA"""
    if modelo is None:
        return jsonify({'success': False, 'error': '❌ Modelo no disponible'}), 400
    
    datos = request.json
    
    try:
        print(f"🔍 Realizando predicción manual...")
        
        # 1. Validamos y convertimos datos
        datos_validados = {
            'Year': float(datos.get('Year', 2020)),
            'Engine Size': float(datos.get('Engine Size', 2.0)),
            'Mileage': float(datos.get('Mileage', 50000)),
            'Brand': str(datos.get('Brand', 'Toyota')),
            'Fuel Type': str(datos.get('Fuel Type', 'Petrol')),
            'Transmission': str(datos.get('Transmission', 'Automatic'))
        }
        
        print(f"📝 Datos validados: {datos_validados}")
        
        # 2. Creamos DataFrame temporal para procesamiento
        temp_df = pd.DataFrame([datos_validados])
        
        # 3. Procesamos cada característica manualmente
        X_final = pd.DataFrame()
        
        # A. Variables numéricas básicas
        X_final['Year'] = [datos_validados['Year']]
        X_final['Engine Size'] = [datos_validados['Engine Size']]
        X_final['Mileage'] = [datos_validados['Mileage']]
        
        # B. Brand_encoded (simplificado - usar el mismo mapeo que en entrenamiento)
        # En una app real, deberías cargar el brand_mapping guardado durante el entrenamiento
        brand_value = 1  # Valor por defecto
        X_final['Brand_encoded'] = [brand_value]
        
        # C. Fuel Type (one-hot) - aseguramos que solo una sea 1
        fuel_types = ['Diesel', 'Electric', 'Hybrid', 'Petrol']
        for ft in fuel_types:
            X_final[f'Fuel_Type_{ft}'] = [1 if datos_validados['Fuel Type'] == ft else 0]
        
        # D. Transmission (one-hot) - asegurar que solo una sea 1
        trans_types = ['Automatic', 'Manual']
        for tt in trans_types:
            X_final[f'Transmission_{tt}'] = [1 if datos_validados['Transmission'] == tt else 0]
        
        # E. Estandarizamos (si hay escalador)
        if escalador and isinstance(escalador, dict) and 'means' in escalador:
            for col in ['Year', 'Engine Size', 'Mileage']:
                if col in escalador['means']:
                    mean_val = escalador['means'][col]
                    std_val = escalador['stds'].get(col, 1.0)
                    if std_val == 0:
                        std_val = 1.0
                    valor_original = datos_validados[col]
                    valor_estandarizado = (valor_original - mean_val) / std_val
                    X_final[f'{col}_standardized'] = [valor_estandarizado]
                else:
                    X_final[f'{col}_standardized'] = [0.0]
        else:
            # Sin escalador, usamos 0
            for col in ['Year', 'Engine Size', 'Mileage']:
                X_final[f'{col}_standardized'] = [0.0]
        
        # 4. Verificamos que tenemos todas las características en el orden CORRECTO
        # Aseguramos que X_final tenga TODAS las características que el modelo espera
        for feature in features_modelo:
            if feature not in X_final.columns:
                print(f"⚠️  Característica faltante '{feature}', agregando con valor 0")
                X_final[feature] = 0.0  # Usar 0.0 en lugar de NaN
        
        # Reordenamos columnas en el orden EXACTO que el modelo espera
        X_final = X_final[features_modelo]
        
        # 5. Verificamos que no hay NaN
        if X_final.isnull().any().any():
            print(f"❌ ¡HAY VALORES NaN EN LOS DATOS!")
            print(f"   Columnas con NaN: {X_final.columns[X_final.isnull().any()].tolist()}")
            # Rellenamos NaN con 0
            X_final = X_final.fillna(0.0)
        
        print(f"📤 Datos finales para modelo: {X_final.shape}")
        print(f"   Valores (primeras 5 columnas):")
        for i, col in enumerate(X_final.columns[:5]):
            print(f"     {col}: {X_final[col].iloc[0]}")
        
        # 6. Predecimos
        prediccion = modelo.predict(X_final)
        etiqueta = prediccion[0]
        
        # 7. Obtenemos probabilidades
        if hasattr(modelo, 'predict_proba'):
            probabilidades = modelo.predict_proba(X_final)[0]
            confianza = float(max(probabilidades))
            
            # Creamos diccionario de probabilidades
            prob_dict = {}
            for i, clase in enumerate(modelo.classes_):
                prob_dict[clase] = round(float(probabilidades[i]), 3)
        else:
            confianza = 0.0
            prob_dict = {}
        
        print(f"✅ Predicción: {etiqueta} (confianza: {confianza:.2%})")
        
        return jsonify({
            'success': True,
            'prediccion': etiqueta,
            'confianza': round(confianza, 3),
            'probabilidades': prob_dict
        }), 200
        
    except Exception as e:
        print(f"❌ Error en predicción manual: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'❌ Error en predicción: {str(e)}'
        }), 500

@app.route('/info_dataset', methods=['GET'])
def info_dataset_route():
    """Devuelve información detallada del dataset"""
    try:
        if df_preprocesado is not None:
            info = obtener_info_dataset(df_preprocesado)
            
            # Información adicional
            info_adicional = {
                'features_modelo': features_modelo,
                'target_col': target_col,
                'modelo': info_dataset.get('modelo_utilizado', 'Desconocido'),
                'accuracy_entrenamiento': info_dataset.get('accuracy', 0),
                'condition_mapping': condition_map
            }
            
            return jsonify({
                'success': True,
                'info_basica': info,
                'info_adicional': info_adicional
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': '❌ Dataset no disponible'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'❌ Error obteniendo información: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de salud para verificar que el servidor está funcionando"""
    return jsonify({
        'status': 'healthy',
        'service': 'car-price-predictor',
        'dataset_loaded': df_preprocesado is not None,
        'model_loaded': modelo is not None,
        'features_count': len(features_modelo),
        'timestamp': pd.Timestamp.now().isoformat()
    }), 200

# ============================================================================
# MANEJO DE ERRORES
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': '❌ Ruta no encontrada',
        'message': 'La URL solicitada no existe en el servidor.'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': '❌ Método no permitido',
        'message': 'El método HTTP no está permitido para esta URL.'
    }), 405

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        'success': False,
        'error': '❌ Error interno del servidor',
        'message': 'Ocurrió un error inesperado. Por favor, intente nuevamente.'
    }), 500

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    print(f"❌ Error no manejado: {error}")
    return jsonify({
        'success': False,
        'error': '❌ Error inesperado',
        'message': str(error)
    }), 500

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SERVICIO FLASK LISTO")
    print("="*60)
    print(f"📊 Dataset: {len(df_preprocesado) if df_preprocesado is not None else 0} filas")
    print(f"🎯 Características del modelo: {len(features_modelo)}")
    print(f"🤖 Modelo: {'✅ CARGADO' if modelo is not None else '❌ NO DISPONIBLE'}")
    print(f"📈 Escalador: {'✅ CARGADO' if escalador is not None else '⚠️  NO DISPONIBLE'}")
    print("="*60)
    print("🌐 URL Principal: http://localhost:5000")
    print("🔧 Health Check: http://localhost:5000/health")
    print("📚 API Info: http://localhost:5000/info_dataset")
    print("="*60)
    print("📋 Endpoints disponibles:")
    print("  POST /configurar    - Configurar muestra de datos")
    print("  GET  /evaluar       - Evaluar modelo y generar gráficas")
    print("  POST /predecir_manual - Realizar predicción manual")
    print("="*60 + "\n")
    
    # Obtenemos puerto de Codespaces o usar 5000 por defecto
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    
    app.run(
        debug=True, 
        port=port, 
        host=host,
        use_reloader=True
    )
