# Clasificación de Cáncer de Mama (Breast Cancer Wisconsin)

Este repositorio presenta un estudio técnico y riguroso sobre la clasificación de tumores de mama utilizando algoritmos de Machine Learning. El enfoque se centra en la precisión diagnóstica, la interpretabilidad del modelo y la minimización de falsos negativos mediante un flujo de trabajo de ciencia de datos estandarizado.

## 1. El Problema
El objetivo es predecir si un tumor de mama es **Maligno** o **Benigno** basándose en características computarizadas de una imagen digital de una aspiración con aguja fina (FNA) de una masa mamaria.

*   **Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set.
*   **Tipo de Problema:** Clasificación Binaria Supervisada.
*   **Métrica Clave:** En este contexto médico, el **Recall (Sensibilidad)** es crítico, ya que queremos minimizar los falsos negativos (no detectar un cáncer real).

## 2. Notebook
El corazón de este proyecto es el notebook `notebooks/BreastCancerClsf.ipynb`. Este no es solo un script de ejecución, sino una guía interactiva que cubre:

1.  **Filosofía y Configuración:** Estética profesional y reproducibilidad.
2.  **Inspección Estadística:** Detección de anomalías y necesidades de escalado.
3.  **EDA:** Visualización de distribuciones y separabilidad de clases (Pairplots).
4.  **Preparacion de los datos:** Preprocesamiento robusto con Pipelines.
5.  **Modelado:** Comparación de Regresión Logística, Random Forest y Gradient Boosting.
6.  **Evaluacion:** Análisis de rendimiento mediante multiples metricas.

## 3. Uso del Repositorio

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/angmzz/BreastCancerML.git
cd Breast-Cancer-Classification
```

### Paso 2: Crear un Entorno Virtual
```bash
# En Windows
python -m venv venv

# En macOS/Linux
python3 -m venv venv
```

### Paso 3: Activar el Entorno Virtual
```bash
# PowerShell
.\venv\Scripts\Activate

# CMD
.\venv\Scripts\activate.bat

# macOS/Linux
source venv/bin/activate
```

### Paso 4: Instalar Requerimientos
```bash
pip install -r requirements.txt
```

### Paso 5: Ejecutar Jupyter Lab
```bash
jupyter lab
```