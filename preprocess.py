import pandas as pd
import numpy as np
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