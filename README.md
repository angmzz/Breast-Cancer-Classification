# Clasificación de Cáncer de Mama (Dataset Breast Cancer Wisconsin)

Este repositorio contiene la implementación de un flujo de trabajo de Machine Learning para diagnosticar tumores de mama. Para ello, se utiliza el célebre Breast Cancer Wisconsin (Diagnostic) Dataset, disponible en la librería Scikit-Learn. Este conjunto de datos fue generado originalmente por el Dr. William H. Wolberg en la Universidad de Wisconsin (1992) a partir de imágenes digitalizadas de aspiraciones con aguja fina.

El proyecto se centra en resolver este problema de clasificación binaria priorizando la sensibilidad (Recall), dado que en un contexto médico es crítico minimizar los falsos negativos. El código está estructurado para ser reproducible y sigue buenas prácticas de Ciencia de Datos, desde la limpieza hasta la evaluación de modelos.

## El Problema
El diagnóstico temprano del cáncer de mama es determinante para la supervivencia del paciente. Los datos utilizados en este proyecto provienen de estudios de Aspiración con Aguja Fina (FNA), una técnica minimamente invasiva que permite observar las características celulares de una masa mamaria.

El desafío principal radica en que la interpretación visual de estas muestras puede ser subjetiva. El objetivo de este modelo no es reemplazar el diagnóstico médico, sino actuar como una herramienta de soporte a la decisión clínica que permita:

Clasificar con alta precisión si una muestra es Maligna o Benigna basándose en la morfología celular digitalizada.

Priorizar la Sensibilidad (Recall) sobre la Precisión pura. En este contexto médico, el costo de un Falso Negativo (decirle a un paciente con cáncer que está sano) es inaceptable, mucho mayor que el de un Falso Positivo (una falsa alarma que se descarta con más pruebas).

## Uso del Repositorio

### Clonar el Repositorio
```bash
git clone https://github.com/angmzz/BreastCancerML.git
cd Breast-Cancer-Classification
```

### Crear un Entorno Virtual
```bash
# En Windows
python -m venv venv

# En macOS/Linux
python3 -m venv venv
```

### Activar el Entorno Virtual
```bash
# PowerShell
.\venv\Scripts\Activate

# CMD
.\venv\Scripts\activate.bat

# macOS/Linux
source venv/bin/activate
```

### Instalar Requerimientos
```bash
pip install -r requirements.txt
```

### Ejecutar Jupyter Lab
```bash
jupyter lab
```