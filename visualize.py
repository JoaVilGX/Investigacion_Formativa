import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import base64
import pandas as pd
import numpy as np
import io
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from io import BytesIO

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def crear_matriz_confusion(y_true, y_pred, labels):
    """Crea matriz de confusión con matplotlib"""
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Crear figura con matplotlib
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Matriz de Confusión')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        
        # Convertir a base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"❌ Error creando matriz de confusión: {e}")
        return None

def crear_grafica_distribucion(df, columna, target_col=None):
    """Crea un histograma con matplotlib"""
    try:
        if columna not in df.columns:
            print(f"⚠️  Columna {columna} no encontrada")
            return None
        
        plt.figure(figsize=(10, 6))
        
        if target_col and target_col in df.columns:
            # Filtramos filas donde target_col no sea nulo
            df_filtrado = df.dropna(subset=[target_col])
            if len(df_filtrado) == 0:
                print(f"⚠️  No hay datos válidos para {target_col}")
                return None
            
            # Obtener las categorías únicas de la columna objetivo
            categorias = df_filtrado[target_col].unique()
            
            for categoria in categorias:
                subset = df_filtrado[df_filtrado[target_col] == categoria]
                plt.hist(subset[columna], alpha=0.5, label=str(categoria), bins=30)
            
            plt.legend(title=target_col)
            plt.title(f"Distribución de {columna} por {target_col}")
        else:
            plt.hist(df[columna].dropna(), bins=30)
            plt.title(f"Distribución de {columna}")
        
        plt.xlabel(columna)
        plt.ylabel('Frecuencia')
        
        # Convertir a base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"❌ Error creando gráfica de distribución: {e}")
        return None

def crear_grafica_rendimiento_por_clase(y_true, y_pred, labels):
    """Crea gráfica de rendimiento por clase con matplotlib"""
    try:
        # Convertir a arrays de numpy si no lo son
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Verificar que hay datos
        if len(y_true) == 0 or len(y_pred) == 0:
            print("⚠️  No hay datos para crear gráfica de rendimiento")
            return None
        
        # Identificar las etiquetas únicas presentes en los datos
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        # Filtrar las etiquetas que están en la lista proporcionada
        labels_present = [label for label in labels if label in unique_labels]
        
        if len(labels_present) == 0:
            print("⚠️  No hay etiquetas comunes entre los datos y la lista proporcionada")
            return None
        
        # Si las etiquetas son strings, mapearlas a números para el cálculo
        if isinstance(unique_labels[0], str):
            # Crear un mapeo de etiqueta a índice
            label_to_index = {label: idx for idx, label in enumerate(labels_present)}
            # Convertir y_true y y_pred a índices
            y_true_indices = np.array([label_to_index[label] for label in y_true if label in label_to_index])
            y_pred_indices = np.array([label_to_index[label] for label in y_pred if label in label_to_index])
        else:
            # Si ya son números, usarlos directamente
            y_true_indices = y_true
            y_pred_indices = y_pred
        
        # Calcular métricas solo para las etiquetas presentes
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_indices, y_pred_indices, labels=range(len(labels_present)), average=None, zero_division=0
        )
        
        # Si solo hay una clase, convertir los arrays a 2D
        if len(labels_present) == 1:
            precision = np.array([precision])
            recall = np.array([recall])
            f1 = np.array([f1])
        
        # Crear gráfico de barras
        x = np.arange(len(labels_present))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(x - width, precision, width, label='Precisión', color='#4CAF50')
        ax.bar(x, recall, width, label='Recall', color='#2196F3')
        ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800')
        
        ax.set_xlabel('Clase')
        ax.set_ylabel('Valor')
        ax.set_title('Rendimiento por Clase')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_present)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir los valores en las barras
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax.text(i - width, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
            ax.text(i, r + 0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width, f + 0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Ajustar el límite del eje Y para que haya espacio para los textos
        ax.set_ylim([0, max(max(precision), max(recall), max(f1)) + 0.1])
        
        # Convertir a base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"❌ Error creando gráfica de rendimiento: {e}")
        import traceback
        traceback.print_exc()
        return None

def crear_grafica_importancia(modelo, nombres_features, top_n=10):
    """Crea gráfico de importancia de características con matplotlib"""
    try:
        if hasattr(modelo, 'feature_importances_'):
            importances = modelo.feature_importances_
            indices = np.argsort(importances)[-top_n:]  # Top N
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Crear barras horizontales
            y_pos = np.arange(len(indices))
            ax.barh(y_pos, importances[indices], color='steelblue', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([nombres_features[i] for i in indices])
            ax.set_xlabel('Importancia', fontsize=12)
            ax.set_title(f'Importancia de Características (Top {top_n})', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Convertir a base64
            img_buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
        return None
        
    except Exception as e:
        print(f"❌ Error creando gráfica de importancia: {e}")
        return None
