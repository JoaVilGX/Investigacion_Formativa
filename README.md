# 🚗 Car Price Condition Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

Un sistema predictivo web basado en Machine Learning para determinar la condición de vehículos usados. Desarrollado con Flask, scikit-learn y una interfaz web interactiva.

## 📋 Tabla de Contenidos
- [Descripción](#descripción)
- [Características Principales](#características-principales)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación y Configuración](#instalación-y-configuración)
- [Uso del Sistema](#uso-del-sistema)
- [API Endpoints](#api-endpoints)
- [Modelo de Machine Learning](#modelo-de-machine-learning)
- [Dataset](#dataset)
- [Despliegue en GitHub Codespaces](#despliegue-en-github-codespaces)
- [Problemas Conocidos](#problemas-conocidos)
- [Autor](#autor)

## 📖 Descripción

Este proyecto implementa un sistema completo de Machine Learning para predecir la condición de vehículos usados ("New", "Like New", "Used") basándose en características como año, kilometraje, tamaño del motor, marca, tipo de combustible y transmisión.

El sistema incluye:
- **Backend API**: Flask con endpoints RESTful
- **Modelo de ML**: Regresión Logística entrenada
- **Frontend Web**: Interfaz interactiva con formularios dinámicos
- **Sistema de configuración**: Muestreo aleatorio y división de datos
- **Evaluación completa**: Métricas y análisis del modelo

## ✨ Características Principales

### 🔧 **Configuración Flexible**
- Configuración dinámica de muestras con semilla aleatoria
- Ajuste de porcentaje de datos a utilizar (5% - 100%)
- Selección de división entrenamiento/prueba (70/30 o 80/20)

### 📊 **Evaluación del Modelo**
- Cálculo de métricas: Accuracy, Precision, Recall, F1-Score
- Generación de gráficas (matriz de confusión, distribución, importancia)
- Análisis de rendimiento por clase

### 🔮 **Sistema de Predicciones**
- Formulario interactivo para predicciones manuales
- Devuelve predicción con porcentaje de confianza
- Muestra probabilidades por cada clase
- Preprocesamiento automático de datos de entrada

### 🌐 **Arquitectura Completa**
- API REST con endpoints documentados
- Frontend responsive con vanilla JavaScript
- Pipeline completo de ML: preprocesamiento → entrenamiento → predicción

## 🛠️ Tecnologías Utilizadas

### **Backend**
- **Python 3.9+**: Lenguaje principal
- **Flask 2.3.3**: Framework web
- **scikit-learn 1.3.0**: Machine Learning
- **pandas 2.0.3**: Manipulación de datos
- **joblib 1.3.2**: Serialización de modelos
- **Flask-CORS**: Manejo de CORS

### **Frontend**
- **HTML5/CSS3**: Estructura y estilos
- **JavaScript (ES6)**: Interactividad
- **Plotly.js**: Visualización de gráficas
- **CSS Personalizado**: Diseño responsive

### **Herramientas de Desarrollo**
- **GitHub Codespaces**: Entorno de desarrollo
- **JSON**: Configuración y comunicación
- **CSV**: Dataset
- **PKL**: Modelos serializados

## 📁 Estructura del Proyecto

```
car-price-predictor/
├── app.py                          # Aplicación principal Flask
├── requirements.txt               # Dependencias Python
├── analizar_modelo.py            # Análisis del modelo entrenado
├── diagnostico_columnas.py       # Diagnóstico de características
├── entrenar_modelo.py            # Script de entrenamiento
├── verificar_features.py         # Verificación de características
│
├── data/                         # Datasets
│   ├── car_price_cleaned.csv      # Dataset limpio
│   └── car_price_prediction_with_missing.csv
│
├── models/                       # Modelos y configuraciones
│   ├── modelo_entrenado.pkl      # Modelo serializado
│   ├── scaler.pkl               # Escalador guardado
│   └── info_flask.json          # Configuración del modelo
│
├── static/                       # Archivos estáticos
│   ├── css/
│   │   └── style.css            # Estilos CSS
│   └── js/
│       ├── app.js               # Lógica principal frontend
│       └── scripts.js           # Scripts adicionales
│
├── templates/                    # Plantillas HTML
│   └── index.html               # Página principal
│
├── devcontainer/                 # Configuración Codespaces
│   └── devcontainer.json
│
└── módulos_python/              # Módulos personalizados
    ├── preprocess.py            # Preprocesamiento de datos
    ├── model.py                 # Funciones del modelo ML
    └── visualize.py             # Generación de gráficas
```

## ⚙️ Instalación y Configuración

### **Requisitos Previos**
- Python 3.9 o superior
- pip (gestor de paquetes Python)
- Git
- Navegador web moderno

### **Instalación Local**

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/car-price-predictor.git
cd car-price-predictor
```

2. **Crear entorno virtual (recomendado):**
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Entrenar el modelo (opcional):**
```bash
python entrenar_modelo.py
```

5. **Ejecutar la aplicación:**
```bash
python app.py
```

6. **Acceder a la aplicación:**
Abre tu navegador y visita: `http://localhost:5000`

### **Instalación en GitHub Codespaces**
El proyecto está configurado para ejecutarse automáticamente en GitHub Codespaces:
1. Haz click en "Code" → "Codespaces"
2. Crea un nuevo codespace
3. La aplicación se iniciará automáticamente en el puerto 5000
4. Se abrirá automáticamente en tu navegador

## 🚀 Uso del Sistema

### **1. Configuración Inicial**
1. Accede a la aplicación web
2. Configura los parámetros de muestra:
   - **Semilla aleatoria**: Controla la reproducibilidad
   - **Porcentaje a usar**: Selecciona qué porcentaje del dataset utilizar
   - **División entrenamiento/prueba**: Elige 80/20 o 70/30
3. Haz click en "Aplicar Configuración"

### **2. Evaluación del Modelo**
1. Haz click en "Generar Gráficas y Métricas"
2. El sistema mostrará:
   - Tabla con métricas de evaluación
   - Gráficas de análisis (si están disponibles)

### **3. Realizar Predicciones**
1. Completa el formulario con los datos del vehículo:
   - Año, Tamaño del motor, Kilometraje
   - Marca, Tipo de combustible, Transmisión
2. Haz click en "Predecir Condición"
3. Verás:
   - Condición predicha (New/Like New/Used)
   - Porcentaje de confianza
   - Probabilidades por cada clase

## 🔌 API Endpoints

### **GET /**
- **Descripción**: Página principal con información del dataset
- **Respuesta**: HTML con interfaz web

### **POST /configurar**
- **Descripción**: Configura la muestra de datos
- **Body**: `{semilla: int, porcentaje: float, split: float}`
- **Respuesta**: JSON con confirmación y tamaños de muestra

### **GET /evaluar**
- **Descripción**: Evalúa el modelo y genera métricas
- **Respuesta**: JSON con métricas y gráficas

### **POST /predecir_manual**
- **Descripción**: Realiza una predicción con datos manuales
- **Body**: `{Year: float, Engine Size: float, Mileage: float, Brand: str, Fuel Type: str, Transmission: str}`
- **Respuesta**: JSON con predicción y probabilidades

### **GET /info_dataset**
- **Descripción**: Obtiene información del dataset
- **Respuesta**: JSON con estadísticas y características

### **GET /health**
- **Descripción**: Verifica el estado del servidor
- **Respuesta**: JSON con estado del servicio

## 🤖 Modelo de Machine Learning

### **Algoritmo Utilizado**
- **Modelo**: Regresión Logística Multiclase
- **Precisión**: 33.78% (problema multiclase desbalanceado)
- **Clases**: ['Used', 'Like New', 'New']

### **Características del Modelo**
El modelo utiliza 13 características procesadas:
1. **Numéricas originales**: Year, Engine Size, Mileage
2. **Codificadas**: Brand_encoded
3. **One-hot encoding**: 
   - Fuel_Type: Diesel, Electric, Hybrid, Petrol
   - Transmission: Automatic, Manual
4. **Estandarizadas**: Year_standardized, Engine Size_standardized, Mileage_standardized

### **Pipeline de Procesamiento**
1. **Carga**: Dataset car_price_cleaned.csv
2. **Limpieza**: Eliminación/imputación de valores faltantes
3. **Transformación**: 
   - Codificación de variables categóricas
   - One-hot encoding
   - Estandarización Z-score
4. **Entrenamiento**: Regresión Logística con validación cruzada implícita
5. **Serialización**: Modelo guardado en formato PKL

## 📊 Dataset

### **Origen**
Dataset sintético de vehículos con 2,500 registros.

### **Características**
- **Filas**: 2,500
- **Columnas**: 8 características + variable objetivo
- **Balance**: Distribución desbalanceada (predominio de "Used")

### **Columnas Disponibles**
1. **Car ID**: Identificador único
2. **Year**: Año del vehículo (2000-2023)
3. **Engine Size**: Tamaño del motor en litros
4. **Mileage**: Kilometraje
5. **Brand**: Marca del vehículo
6. **Fuel Type**: Tipo de combustible (Petrol, Diesel, Electric, Hybrid)
7. **Transmission**: Tipo de transmisión (Automatic, Manual)
8. **Condition**: Variable objetivo (New, Like New, Used)

### **Preprocesamiento**
- Eliminación de filas completamente vacías
- Imputación de valores faltantes con mediana/moda
- Conversión de tipos de datos
- Eliminación de duplicados

## 🌐 Despliegue en GitHub Codespaces

### **Ventajas**
- ✅ Entorno preconfigurado
- ✅ Sin necesidad de instalación local
- ✅ Accesible desde cualquier navegador
- ✅ Recursos escalables

### **Configuración Automática**
El archivo `devcontainer/devcontainer.json` configura:
- **Puerto 5000** para la aplicación Flask
- **Instalación automática** de dependencias
- **Apertura automática** del navegador

### **Acceso**
1. Visita el repositorio en GitHub
2. Haz click en "Code" → "Codespaces"
3. Crea un nuevo codespace
4. Espera a que se instalen las dependencias
5. ¡La aplicación se abrirá automáticamente!

## ⚠️ Problemas Conocidos

### **Problema con Gráficas en Codespaces**
**Descripción**: Las imágenes base64 generadas por Plotly no se renderizan correctamente en GitHub Codespaces debido a limitaciones técnicas del entorno remoto.

**Evidencia de Funcionamiento**:
- ✅ Los datos de las gráficas se generan correctamente
- ✅ En consola aparece: "Matriz de confusión generada con 97 muestras"
- ✅ El backend procesa y devuelve las imágenes
- ✅ El problema es específico del renderizado en el navegador remoto

**Solución Temporal**: Las métricas numéricas están disponibles y el sistema de predicciones funciona completamente.

### **Características Desbalanceadas**
- El dataset tiene distribución desbalanceada (más vehículos "Used")
- Esto afecta la precisión del modelo para las clases minoritarias

### **Limitaciones del Modelo**
- Precisión del 33.78% debido a la complejidad multiclase
- Se recomienda explorar otros algoritmos para mejorar resultados

### **Mejoras Sugeridas**
- Implementar más algoritmos de ML para comparación
- Añadir más métricas de evaluación
- Mejorar el balance del dataset
- Implementar sistema de logging
- Añadir tests unitarios
- Mejorar interfaz de usuario


## 👨‍💻 Autores

**Nombre**: Joaquin Villacreses Moreno
- GitHub: [@JoaVilGX](https://github.com/JoaVilGX)


- Proyecto: Investigacion_Formativa / Car Price Condition Predictor
- Curso: Segundo Semestre "C"
- Universidad: Universidad Nacional de Chimborazo

---

<div align="center">
  <p><strong>¡Gracias por visitar el proyecto! 🚀</strong></p>
  <p>Si encuentras útil este proyecto, ¡dale una estrella en GitHub!</p>
</div>