import plotly.express as px
import plotly.graph_objects as go
import plotly
import json
import plotly.io as pio
import base64
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from io import BytesIO

def crear_matriz_confusion(y_true, y_pred, labels):
    """Crea una matriz de confusión y devuelve imagen base64"""
    from sklearn.metrics import confusion_matrix
    
    try:
        # Creamos matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Creamos la figura
        fig = px.imshow(cm, 
                        text_auto=True,
                        labels=dict(x="Predicho", y="Real", color="Cantidad"),
                        x=labels, 
                        y=labels,
                        title="Matriz de Confusión",
                        color_continuous_scale='Blues')
        
        fig.update_layout(width=600, height=500)
        
        # Convertimos a imagen base64
        img_bytes = pio.to_image(fig, format='png', width=600, height=500)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"❌ Error creando matriz de confusión: {e}")
        return None

def crear_grafica_distribucion(df, columna, target_col=None):
    """Crea un histograma y devuelve imagen base64"""
    try:
        if columna not in df.columns:
            print(f"⚠️  Columna {columna} no encontrada")
            return None
        
        if target_col and target_col in df.columns:
            fig = px.histogram(df.dropna(subset=[target_col]), 
                              x=columna, 
                              color=target_col, 
                              barmode='group',
                              title=f"Distribución de {columna} por {target_col}")
        else:
            fig = px.histogram(df, x=columna, title=f"Distribución de {columna}")
        
        fig.update_layout(width=800, height=500)
        
        # Convertimos a imagen base64
        img_bytes = pio.to_image(fig, format='png', width=800, height=500)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"❌ Error creando gráfica de distribución: {e}")
        return None

def crear_grafica_rendimiento_por_clase(y_true, y_pred, labels):
    """Crea gráfica de rendimiento por clase y devuelve imagen base64"""
    try:
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Precisión',
            x=labels,
            y=precision,
            text=[f'{p:.3f}' for p in precision],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Recall',
            x=labels,
            y=recall,
            text=[f'{r:.3f}' for r in recall],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=labels,
            y=f1,
            text=[f'{f:.3f}' for f in f1],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Rendimiento por Clase",
            xaxis_title="Clase",
            yaxis_title="Valor",
            barmode='group',
            width=800,
            height=500
        )
        
        # Convertimos a imagen base64
        img_bytes = pio.to_image(fig, format='png', width=800, height=500)
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"❌ Error creando gráfica de rendimiento: {e}")
        return None

def crear_grafica_metricas_comparacion(metricas_dict):
    """Crea gráfico de barras para comparar métricas"""
    fig = go.Figure()
    
    for modelo, metricas in metricas_dict.items():
        fig.add_trace(go.Bar(
            name=modelo,
            x=list(metricas.keys()),
            y=list(metricas.values()),
            text=[f'{v:.3f}' for v in metricas.values()],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Comparación de Métricas",
        xaxis_title="Métrica",
        yaxis_title="Valor",
        barmode='group'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def crear_grafica_rendimiento_por_clase(y_true, y_pred, labels):
    """Crea gráfico de rendimiento por clase"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), average=None
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Precisión',
        x=labels,
        y=precision,
        text=[f'{p:.3f}' for p in precision],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Recall',
        x=labels,
        y=recall,
        text=[f'{r:.3f}' for r in recall],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='F1-Score',
        x=labels,
        y=f1,
        text=[f'{f:.3f}' for f in f1],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Rendimiento por Clase",
        xaxis_title="Clase",
        yaxis_title="Valor",
        barmode='group'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def crear_grafica_distribucion(df, columna, target_col=None):
    """Crea un histograma o gráfico de barras"""
    try:
        # Verificamos que la columna exista en el DataFrame
        if columna not in df.columns:
            print(f"⚠️  Columna {columna} no encontrada en el DataFrame")
            return None
        
        # Si se proporciona target_col, verificar que exista
        if target_col and target_col in df.columns:
            # Filtramos filas donde target_col no sea nulo
            df_filtrado = df.dropna(subset=[target_col])
            if len(df_filtrado) == 0:
                print(f"⚠️  No hay datos válidos para {target_col}")
                return None
            fig = px.histogram(df_filtrado, x=columna, color=target_col, barmode='group')
        else:
            # Si no hay target_col, usar solo la columna
            fig = px.histogram(df, x=columna)
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"❌ Error creando gráfica de distribución: {e}")
        return None

def crear_matriz_confusion(y_true, y_pred, labels):
    """Crea una matriz de confusión interactiva"""
    # Convertimos a arrays de numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filtramos solo las etiquetas que están en la lista de labels
    mask = np.isin(y_true, labels) & np.isin(y_pred, labels)
    y_true_filtrado = y_true[mask]
    y_pred_filtrado = y_pred[mask]
    
    if len(y_true_filtrado) == 0:
        print("⚠️  No hay datos válidos para la matriz de confusión")
        # Creamos una matriz de ceros
        cm = np.zeros((len(labels), len(labels)), dtype=int)
    else:
        cm = confusion_matrix(y_true_filtrado, y_pred_filtrado, labels=labels)
    
    fig = px.imshow(cm, text_auto=True, 
                    labels=dict(x="Predicho", y="Real", color="Cantidad"),
                    x=labels, y=labels,
                    title="Matriz de Confusión")
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def crear_grafica_importancia(modelo, nombres_features, top_n=10):
    """Crea gráfico de importancia de características"""
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
        indices = np.argsort(importances)[-top_n:]  # Top N
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importances[indices],
            y=[nombres_features[i] for i in indices],
            orientation='h'
        ))
        fig.update_layout(title=f"Importancia de Características (Top {top_n})")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return None

def crear_grafica_comparacion_metricas(metrics_dict):
    """Crea gráfico de comparación de métricas entre modelos"""
    modelos = list(metrics_dict.keys())
    metricas = list(metrics_dict[modelos[0]].keys())
    
    fig = go.Figure()
    
    for i, modelo in enumerate(modelos):
        valores = [metrics_dict[modelo][metrica] for metrica in metricas]
        fig.add_trace(go.Bar(
            name=modelo,
            x=metricas,
            y=valores,
            text=[f'{v:.3f}' for v in valores],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Comparación de Métricas por Modelo",
        barmode='group',
        xaxis_title="Métrica",
        yaxis_title="Valor"
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def crear_grafica_dispersion(df, x_col, y_col, color_col=None):
    """Crea gráfico de dispersión"""
    if color_col and color_col in df.columns:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
    else:
        fig = px.scatter(df, x=x_col, y=y_col)
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def crear_grafica_correlacion(df, columnas):
    """Crea matriz de correlación"""
    corr_matrix = df[columnas].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        title="Matriz de Correlación"
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
