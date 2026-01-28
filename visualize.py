import plotly.express as px
import plotly.graph_objects as go
import plotly
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def crear_grafica_distribucion(df, columna, target_col=None):
    """Crea un histograma o gráfico de barras"""
    if target_col and target_col in df.columns:
        fig = px.histogram(df, x=columna, color=target_col, barmode='group')
    else:
        fig = px.histogram(df, x=columna)
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def crear_matriz_confusion(y_true, y_pred, labels):
    """Crea una matriz de confusión interactiva"""
    cm = confusion_matrix(y_true, y_pred)
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