import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import joblib

def cargar_datos(ruta='data/car_price_cleaned.csv'):
    """Carga el dataset limpio desde la carpeta data/"""
    return pd.read_csv(ruta)

def limpiar_datos(df):
    """Realiza limpieza completa del dataset"""
    # Crear copia
    df_clean = df.copy()
    
    # Eliminar filas completamente vacías
    df_clean = df_clean.dropna(how='all')
    
    # Eliminar filas sin identificador
    df_clean = df_clean.dropna(subset=['Car ID'])
    
    # Convertir Car ID a entero
    df_clean['Car ID'] = df_clean['Car ID'].astype(int)
    
    # Imputación de valores faltantes numéricos
    numeric_cols = ['Year', 'Engine Size', 'Mileage', 'Price']
    for col in numeric_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    # Imputación de valores faltantes categóricos
    categorical_cols = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
    for col in categorical_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Desconocido'
            df_clean[col].fillna(mode_val, inplace=True)
    
    return df_clean

def transformar_categoricas(df_clean):
    """Transforma y codifica variables categóricas"""
    # Codificación de variables ordinales
    condition_mapping = {'New': 2, 'Like New': 1, 'Used': 0}
    df_clean['Condition_encoded'] = df_clean['Condition'].map(condition_mapping)
    
    # One-Hot Encoding para variables nominales
    nominal_vars = ['Fuel Type', 'Transmission']
    for var in nominal_vars:
        if var in df_clean.columns:
            dummies = pd.get_dummies(df_clean[var], prefix=var.replace(' ', '_'))
            df_clean = pd.concat([df_clean, dummies], axis=1)
    
    # Label Encoding para marcas
    if 'Brand' in df_clean.columns:
        brands = df_clean['Brand'].unique()
        brand_mapping = {brand: i for i, brand in enumerate(brands)}
        df_clean['Brand_encoded'] = df_clean['Brand'].map(brand_mapping)
    
    return df_clean, condition_mapping, brand_mapping

def estandarizar_numericas(df_clean):
    """Estandariza variables numéricas"""
    numeric_to_scale = ['Year', 'Engine Size', 'Mileage']
    
    # Estandarización manual (Z-score)
    scalers = {}
    for col in numeric_to_scale:
        if col in df_clean.columns:
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            df_clean[f'{col}_standardized'] = (df_clean[col] - mean_val) / std_val
            scalers[col] = {'mean': mean_val, 'std': std_val}
    
    # Normalización Min-Max para Price
    if 'Price' in df_clean.columns:
        min_price = df_clean['Price'].min()
        max_price = df_clean['Price'].max()
        df_clean['Price_normalized'] = (df_clean['Price'] - min_price) / (max_price - min_price)
        scalers['Price'] = {'min': min_price, 'max': max_price}
    
    return df_clean, scalers

def guardar_escalador(scaler, ruta='models/scaler.pkl'):
    """Guarda el escalador en un archivo .pkl."""
    joblib.dump(scaler, ruta)
    return True

def obtener_info_dataset(df):
    """Obtiene información básica del dataset"""
    return {
        'filas': df.shape[0],
        'columnas': df.shape[1],
        'columnas_numericas': df.select_dtypes(include=[np.number]).columns.tolist(),
        'columnas_categoricas': df.select_dtypes(include=['object']).columns.tolist(),
        'valores_faltantes': df.isnull().sum().sum(),
        'duplicados': df.duplicated().sum()
    }

# ============================================================================
# FUNCIONES DE PREPARACIÓN PARA FLASK
# ============================================================================

def preparar_datos_completos(ruta='data/car_price_cleaned.csv'):
    """
    Carga y preprocesa completamente los datos para el modelo Flask.
    Devuelve el DataFrame preprocesado y las transformaciones aplicadas.
    """
    print("🔄 Preparando datos completos...")
    
    # 1. Cargar y limpiar
    df = cargar_datos(ruta)
    df_clean = limpiar_datos(df)
    
    # 2. Transformar categóricas
    df_clean, condition_mapping, brand_mapping = transformar_categoricas(df_clean)
    
    # 3. Estandarizar numéricas
    df_clean, scalers = estandarizar_numericas(df_clean)
    
    # 4. Codificar variable objetivo
    condition_map = {'New': 2, 'Like New': 1, 'Used': 0}
    df_clean['Condition_encoded'] = df_clean['Condition'].map(condition_map)
    
    print(f"✅ Datos preparados: {len(df_clean)} filas")
    return df_clean, condition_map, brand_mapping, scalers

def preparar_datos_para_modelo(ruta='data/car_price_cleaned.csv', ruta_json='models/info_flask.json'):
    """
    Prepara datos específicamente para el modelo Flask, asegurando solo las 13 columnas requeridas.
    """
    print("🔄 Preparando datos para el modelo Flask...")
    
    # 1. Cargar configuración del modelo
    try:
        with open(ruta_json, 'r') as f:
            info = json.load(f)
        features_requeridas = info.get('features_para_modelo', [])
        target_col = info.get('variable_objetivo', 'Condition')
        print(f"📋 Características requeridas: {len(features_requeridas)}")
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        return None, None, [], {}
    
    # 2. Cargar y limpiar datos
    df = cargar_datos(ruta)
    df_clean = limpiar_datos(df)
    
    # 3. Crear un DataFrame vacío para las características finales
    df_final = pd.DataFrame(index=df_clean.index)
    
    # 4. Añadir columnas originales necesarias
    df_final['Year'] = df_clean['Year']
    df_final['Engine Size'] = df_clean['Engine Size']
    df_final['Mileage'] = df_clean['Mileage']
    df_final['Brand'] = df_clean['Brand']
    df_final['Fuel Type'] = df_clean['Fuel Type']
    df_final['Transmission'] = df_clean['Transmission']
    df_final['Condition'] = df_clean['Condition']
    
    # 5. Crear las 13 características EXACTAS en el orden correcto
    # Codificación de Brand
    unique_brands = sorted(df_final['Brand'].unique())
    brand_mapping = {brand: idx for idx, brand in enumerate(unique_brands)}
    df_final['Brand_encoded'] = df_final['Brand'].map(brand_mapping)
    
    # One-Hot Encoding para Fuel Type
    fuel_categories = ['Diesel', 'Electric', 'Hybrid', 'Petrol']
    for fuel in fuel_categories:
        df_final[f'Fuel_Type_{fuel}'] = (df_final['Fuel Type'] == fuel).astype(int)
    
    # One-Hot Encoding para Transmission
    transmission_categories = ['Automatic', 'Manual']
    for trans in transmission_categories:
        df_final[f'Transmission_{trans}'] = (df_final['Transmission'] == trans).astype(int)
    
    # Estandarización
    scalers = {}
    for col in ['Year', 'Engine Size', 'Mileage']:
        mean_val = df_final[col].mean()
        std_val = df_final[col].std()
        df_final[f'{col}_standardized'] = (df_final[col] - mean_val) / std_val if std_val != 0 else 0
        scalers[col] = {'mean': mean_val, 'std': std_val}
    
    # 6. Codificar variable objetivo
    condition_map = {'New': 2, 'Like New': 1, 'Used': 0}
    df_final['Condition_encoded'] = df_final['Condition'].map(condition_map)
    
    # 7. VERIFICAR que tenemos todas las características requeridas
    missing = [f for f in features_requeridas if f not in df_final.columns]
    if missing:
        print(f"❌ Características faltantes: {missing}")
        for f in missing:
            df_final[f] = 0  # Crear con valor 0
    
    # 8. Seleccionar SOLO las 13 características + objetivo, en el orden correcto
    columnas_finales = [col for col in features_requeridas if col in df_final.columns]
    columnas_finales.append('Condition_encoded')
    
    # Eliminar posibles duplicados
    columnas_finales = list(dict.fromkeys(columnas_finales))
    
    df_resultado = df_final[columnas_finales].copy()
    
    # VERIFICACIÓN FINAL
    print(f"\n✅ DATOS FINALES: {df_resultado.shape}")
    print(f"   - Filas: {df_resultado.shape[0]}")
    print(f"   - Columnas: {df_resultado.shape[1]}")
    print(f"   - Características: {[c for c in df_resultado.columns if c != 'Condition_encoded']}")
    
    # Verificar duplicados
    if len(df_resultado.columns) != len(set(df_resultado.columns)):
        print("⚠️  ¡ADVERTENCIA: Hay columnas duplicadas!")
        print(f"   Columnas únicas: {len(set(df_resultado.columns))}")
        print(f"   Columnas totales: {len(df_resultado.columns)}")
    
    return df_resultado, condition_map, brand_mapping, scalers

def asegurar_caracteristicas(df, features_requeridas):
    """
    Asegura que el DataFrame tenga todas las características requeridas.
    Crea las características faltantes si es necesario.
    """
    for feature in features_requeridas:
        if feature not in df.columns:
            print(f"⚠️ Creando característica faltante: {feature}")
            
            if feature.startswith('Fuel_Type_'):
                fuel_type = feature.replace('Fuel_Type_', '')
                df[feature] = (df['Fuel Type'] == fuel_type).astype(int)
                
            elif feature.startswith('Transmission_'):
                transmission = feature.replace('Transmission_', '')
                df[feature] = (df['Transmission'] == transmission).astype(int)
                
            elif feature == 'Brand_encoded' and 'Brand' in df.columns:
                # Si no existe Brand_encoded pero sí Brand, codificar
                if 'Brand' in df.columns:
                    unique_brands = df['Brand'].unique()
                    brand_mapping = {brand: i for i, brand in enumerate(sorted(unique_brands))}
                    df[feature] = df['Brand'].map(brand_mapping)
                else:
                    df[feature] = 0
                    
            elif feature.endswith('_standardized'):
                base_col = feature.replace('_standardized', '')
                if base_col in df.columns:
                    # Calcular estandarización
                    mean_val = df[base_col].mean()
                    std_val = df[base_col].std()
                    df[feature] = (df[base_col] - mean_val) / std_val if std_val != 0 else 0
                else:
                    df[feature] = 0
                    
            else:
                df[feature] = 0
    
    return df

def obtener_features_modelo(ruta_json='models/info_flask.json'):
    """Lee las características del modelo desde el archivo JSON"""
    import json
    try:
        with open(ruta_json, 'r') as f:
            info = json.load(f)
        return info.get('features_para_modelo', [])
    except FileNotFoundError:
        print(f"❌ Archivo {ruta_json} no encontrado")
        return []
    
def preparar_datos_para_modelo_sin_duplicados(ruta='data/car_price_cleaned.csv'):
    """
    Versión simplificada que prepara datos para el modelo Flask.
    """
    print("🔄 Preparando datos para el modelo (versión simplificada)...")
    
    # 1. Cargar y limpiar datos
    df = cargar_datos(ruta)
    df_clean = limpiar_datos(df)
    
    # 2. Crear DataFrame con columnas básicas
    columnas_base = ['Year', 'Engine Size', 'Mileage', 'Brand', 'Fuel Type', 'Transmission', 'Condition']
    df_base = df_clean[columnas_base].copy()
    
    # 3. Crear las 13 características manualmente
    df_final = pd.DataFrame(index=df_base.index)
    
    # Variables numéricas
    for col in ['Year', 'Engine Size', 'Mileage']:
        df_final[col] = df_base[col].astype(float)
    
    # Codificar Brand
    unique_brands = sorted(df_base['Brand'].dropna().unique())
    brand_mapping = {brand: idx for idx, brand in enumerate(unique_brands)}
    df_final['Brand_encoded'] = df_base['Brand'].map(brand_mapping).fillna(0).astype(int)
    
    # One-Hot para Fuel Type
    for fuel in ['Diesel', 'Electric', 'Hybrid', 'Petrol']:
        df_final[f'Fuel_Type_{fuel}'] = (df_base['Fuel Type'] == fuel).astype(int)
    
    # One-Hot para Transmission
    for trans in ['Automatic', 'Manual']:
        df_final[f'Transmission_{trans}'] = (df_base['Transmission'] == trans).astype(int)
    
    # Estandarización
    scalers = {}
    for col in ['Year', 'Engine Size', 'Mileage']:
        mean_val = df_final[col].mean()
        std_val = df_final[col].std()
        df_final[f'{col}_standardized'] = (df_final[col] - mean_val) / (std_val if std_val != 0 else 1)
        scalers[col] = {'mean': mean_val, 'std': std_val}
    
    # Variable objetivo (usar strings directamente)
    df_final['Condition_encoded'] = df_base['Condition']
    
    # Verificar que tenemos 13 características + 1 objetivo
    print(f"✅ Datos preparados: {df_final.shape}")
    print(f"   Columnas: {df_final.columns.tolist()}")
    
    return df_final, {'New': 'New', 'Like New': 'Like New', 'Used': 'Used'}, brand_mapping, scalers