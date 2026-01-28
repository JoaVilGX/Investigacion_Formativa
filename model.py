import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def cargar_modelo(ruta='models/modelo_entrenado.pkl'):
    """Carga el modelo entrenado."""
    return joblib.load(ruta)

def entrenar_modelo(X, y, test_size=0.2, random_state=42):
    """Entrena un modelo de clasificación y devuelve el modelo y métricas."""
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Modelo (ajusta según tu notebook)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
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
        best_name = "Random Forest"
        best_acc = acc_rf
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
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

def predecir(modelo, datos):
    """Realiza predicciones con el modelo."""
    return modelo.predict(datos)

def obtener_metricas(y_true, y_pred):
    """Calcula y retorna métricas de evaluación."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

def cargar_escalador(ruta='models/scaler.pkl'):
    """Carga el escalador entrenado desde un archivo .pkl"""
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