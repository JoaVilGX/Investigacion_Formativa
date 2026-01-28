import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def entrenar_y_guardar_modelo_completo(ruta_datos='data/car_price_cleaned.csv', 
                                     ruta_modelo='models/modelo_entrenado.pkl',
                                     ruta_scaler='models/scaler.pkl',
                                     ruta_info='models/info_flask.json'):
    """
    Entrena un modelo completo desde cero y guarda todo lo necesario.
    
    Returns:
        Tupla (modelo, scaler, info_dataset)
    """
    print("\n" + "="*60)
    print("🤖 ENTRENANDO MODELO COMPLETO DESDE CERO")
    print("="*60)
    
    try:
        from preprocess import cargar_datos, limpiar_datos
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        import joblib
        import json
        import pandas as pd
        import numpy as np
        
        # 1. Cargar datos
        df = cargar_datos(ruta_datos)
        df_clean = limpiar_datos(df)
        
        print(f"📊 Dataset original: {df_clean.shape}")
        
        # 3. Definir características y variable objetivo
        target_col = 'Condition'
        
        # Características exactas para el modelo (las 13 que necesitamos)
        features_unicas = [
            'Year',
            'Engine Size', 
            'Mileage',
            'Brand_encoded',
            'Fuel_Type_Diesel',
            'Fuel_Type_Electric', 
            'Fuel_Type_Hybrid',
            'Fuel_Type_Petrol',
            'Transmission_Automatic',
            'Transmission_Manual',
            'Year_standardized',
            'Engine Size_standardized',
            'Mileage_standardized'
        ]

        df_procesado = pd.DataFrame(index=df_clean.index)
        
        # A. Variables numéricas originales
        df_procesado['Year'] = df_clean['Year'].astype(float)
        df_procesado['Engine Size'] = df_clean['Engine Size'].astype(float)
        df_procesado['Mileage'] = df_clean['Mileage'].astype(float)
        
        # B. Codificar Brand (simplificado)
        unique_brands = sorted(df_clean['Brand'].dropna().unique())
        brand_mapping = {brand: idx for idx, brand in enumerate(unique_brands)}
        df_procesado['Brand_encoded'] = df_clean['Brand'].map(brand_mapping).fillna(0).astype(int)
        
        # C. One-Hot Encoding para Fuel Type (sin duplicados)
        fuel_types = ['Diesel', 'Electric', 'Hybrid', 'Petrol']
        for fuel in fuel_types:
            df_procesado[f'Fuel_Type_{fuel}'] = (df_clean['Fuel Type'] == fuel).astype(int)
        
        # D. One-Hot Encoding para Transmission (sin duplicados)
        transmissions = ['Automatic', 'Manual']
        for trans in transmissions:
            df_procesado[f'Transmission_{trans}'] = (df_clean['Transmission'] == trans).astype(int)
        
        # E. Estandarización
        for col in ['Year', 'Engine Size', 'Mileage']:
            mean_val = df_procesado[col].mean()
            std_val = df_procesado[col].std()
            df_procesado[f'{col}_standardized'] = (df_procesado[col] - mean_val) / (std_val if std_val != 0 else 1)
        
        # F. Variable objetivo
        condition_map = {'New': 'New', 'Like New': 'Like New', 'Used': 'Used'}
        df_procesado['Condition_encoded'] = df_clean['Condition'].map(condition_map)
        
        # 4. VERIFICAR que no hay duplicados
        columnas = df_procesado.columns.tolist()
        if len(columnas) != len(set(columnas)):
            print("❌ ¡HAY COLUMNAS DUPLICADAS!")
            from collections import Counter
            duplicados = [item for item, count in Counter(columnas).items() if count > 1]
            print(f"   Duplicados: {duplicados}")
            return None, None, None
        
        # 5. Seleccionar solo las 13 características + objetivo
        X = df_procesado[features_unicas]
        y = df_procesado['Condition_encoded']
        
        print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
        print(f"   Características: {list(X.columns)}")
        print(f"   Clases en y: {y.unique().tolist()}")
        
        # 6. Dividir y entrenar
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        modelo = LogisticRegression(max_iter=10, random_state=42)
        modelo.fit(X_train, y_train)
        
        # 7. Evaluar
        from sklearn.metrics import accuracy_score
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"📈 Accuracy: {accuracy:.4f}")
        
        # 8. Guardar
        joblib.dump(modelo, 'models/modelo_entrenado.pkl')
        
        # 9. Guardar escalador
        scaler_data = {
            'means': {col: float(df_procesado[col].mean()) for col in ['Year', 'Engine Size', 'Mileage']},
            'stds': {col: float(df_procesado[col].std()) for col in ['Year', 'Engine Size', 'Mileage']},
            'features': features_unicas
        }
        joblib.dump(scaler_data, 'models/scaler.pkl')
        
        # 10. Guardar info
        info_data = {
            'nombre_dataset': 'car_price_cleaned.csv',
            'variable_objetivo': 'Condition',
            'modelo_utilizado': 'Regresión Logística',
            'accuracy': float(accuracy),
            'features_para_modelo': features_unicas
        }
        
        with open('models/info_flask.json', 'w') as f:
            json.dump(info_data, f, indent=4)
        
        print("\n✅ Modelo guardado:")
        print(f"   - models/modelo_entrenado.pkl")
        print(f"   - models/scaler.pkl")
        print(f"   - models/info_flask.json")
        
        return modelo, scaler_data, info_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def cargar_modelo(ruta='models/modelo_entrenado.pkl'):
    """Carga el modelo entrenado desde un archivo .pkl"""
    try:
        modelo = joblib.load(ruta)
        print(f"✅ Modelo cargado desde {ruta}")
        return modelo
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {ruta}")
        return None
    except Exception as e:
        print(f"❌ Error al cargar modelo: {str(e)}")
        return None

def cargar_escalador(ruta='models/scaler.pkl'):
    """Carga el escalador desde un archivo .pkl"""
    try:
        escalador = joblib.load(ruta)
        print(f"✅ Escalador cargado desde {ruta}")
        return escalador
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {ruta}")
        return None
    except Exception as e:
        print(f"❌ Error al cargar escalador: {str(e)}")
        return None

def predecir_con_preprocesamiento(modelo, X, scalers=None, features_requeridas=None):
    """
    Realiza predicciones asegurando que los datos tengan EXACTAMENTE las características requeridas.
    """
    try:
        print(f"📥 Datos recibidos para predicción: {X.shape}")
        
        # Si X es un DataFrame, verificar sus columnas
        if hasattr(X, 'columns'):
            # Eliminar columnas duplicadas
            X = X.loc[:, ~X.columns.duplicated()]
            print(f"   Columnas recibidas (sin duplicados): {list(X.columns)}")
        
        # Determinar las características que el modelo espera
        # Si el modelo tiene feature_names_in_, usarlas (son las usadas en el entrenamiento)
        if hasattr(modelo, 'feature_names_in_'):
            features_esperadas = modelo.feature_names_in_
            print(f"   Características esperadas por el modelo: {features_esperadas}")
        elif features_requeridas is not None:
            features_esperadas = features_requeridas
            print(f"   Características esperadas (de la lista): {features_esperadas}")
        else:
            # Si no hay información, usar las columnas de X (asumiendo que ya están bien)
            features_esperadas = list(X.columns) if hasattr(X, 'columns') else None
            print(f"   Características esperadas (de los datos): {features_esperadas}")
        
        # Si tenemos features_esperadas, forzar a que X las tenga en el orden correcto
        if features_esperadas is not None:
            X_prepared = pd.DataFrame()
            
            # Para cada característica esperada, obtenerla de X o crear con 0
            for feature in features_esperadas:
                if feature in X.columns:
                    X_prepared[feature] = X[feature]
                else:
                    print(f"⚠️  Característica faltante: {feature} - creando con valor 0")
                    X_prepared[feature] = 0
            
            # Asegurar el orden correcto (según features_esperadas)
            X_prepared = X_prepared[features_esperadas]
        else:
            X_prepared = X
        
        print(f"📤 Datos preparados para modelo: {X_prepared.shape}")
        print(f"   Columnas finales: {list(X_prepared.columns)}")
        
        # Aplicar escalado si hay scalers
        if scalers and isinstance(scalers, dict) and 'means' in scalers:
            for col in scalers.get('means', {}):
                col_std = f'{col}_standardized'
                if col_std in X_prepared.columns:
                    mean_val = scalers['means'][col]
                    std_val = scalers['stds'][col]
                    X_prepared[col_std] = (X_prepared[col] - mean_val) / (std_val if std_val != 0 else 1)
        
        # Realizar predicción
        predicciones = modelo.predict(X_prepared)
        print(f"✅ Predicciones realizadas: {len(predicciones)}")
        return predicciones
        
    except Exception as e:
        print(f"❌ Error en predecir_con_preprocesamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predecir(modelo, X, escalador=None):
    """
    Realiza predicciones con el modelo
    
    Args:
        modelo: modelo entrenado
        X: características de entrada (DataFrame o array)
        escalador: escalador opcional para normalizar los datos
        
    Returns:
        Predicciones (array)
    """
    try:
        # Si hay escalador, escalar los datos
        if escalador is not None:
            X_scaled = escalador.transform(X)
        else:
            X_scaled = X
            
        # Realizar predicción
        predicciones = modelo.predict(X_scaled)
        return predicciones
        
    except Exception as e:
        print(f"❌ Error al realizar predicciones: {str(e)}")
        return None

def obtener_probabilidades(modelo, X, escalador=None):
    """
    Obtiene las probabilidades de predicción (si el modelo lo soporta)
    
    Args:
        modelo: modelo entrenado
        X: características de entrada
        escalador: escalador opcional
        
    Returns:
        Probabilidades de predicción
    """
    try:
        # Si hay escalador, escalar los datos
        if escalador is not None:
            X_scaled = escalador.transform(X)
        else:
            X_scaled = X
            
        # Obtener probabilidades si el modelo las soporta
        if hasattr(modelo, 'predict_proba'):
            probabilidades = modelo.predict_proba(X_scaled)
            return probabilidades
        else:
            print("⚠️ El modelo no soporta predict_proba()")
            return None
            
    except Exception as e:
        print(f"❌ Error al obtener probabilidades: {str(e)}")
        return None

def obtener_metricas(y_true, y_pred):
    """
    Calcula métricas de evaluación del modelo
    
    Args:
        y_true: valores reales
        y_pred: predicciones
        
    Returns:
        Diccionario con métricas
    """
    try:
        metricas = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'matriz_confusion': confusion_matrix(y_true, y_pred).tolist(),
            'reporte': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }
        return metricas
        
    except Exception as e:
        print(f"❌ Error al calcular métricas: {str(e)}")
        return None

def entrenar_modelo(X, y, test_size=0.2, random_state=42):
    """Entrena un modelo de clasificación y devuelve el modelo y métricas."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Entrenar ambos modelos
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_model.fit(X_train, y_train)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_model.fit(X_train, y_train)
    
    # Evaluar
    y_pred_rf = rf_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test)
    
    # Métricas
    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    
    # Elegir mejor modelo
    if acc_rf > acc_lr:
        best_model = rf_model
        best_name = 'RandomForest'
        best_acc = acc_rf
    else:
        best_model = lr_model
        best_name = 'LogisticRegression'
        best_acc = acc_lr
    
    # Guardar ambos modelos
    modelo_completo = {
        'rf_model': rf_model,
        'logreg_model': lr_model,
        'X_test': X_test,
        'y_test': y_test,
        'best_model': best_model,
        'best_model_name': best_name,
        'accuracy': best_acc
    }
    
    return modelo_completo