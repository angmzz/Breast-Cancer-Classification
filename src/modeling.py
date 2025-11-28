from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Devuelve un diccionario de modelos instanciados para comparar.
    
    Args:
        random_state: Semilla para reproducibilidad.
        
    Return:
        Dict: Diccionario de nombres de modelos e instancias.
    """
    return {
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=random_state),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=random_state)
    }

def train_and_evaluate(models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, y_test: pd.Series, cv_folds: int = 5) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Entrena modelos, los evalúa usando CV y conjunto de prueba, y devuelve métricas y resultados de predicción.
    
    Args:
        models: Diccionario de modelos.
        X_train: Características de entrenamiento.
        y_train: Objetivo de entrenamiento.
        X_test: Características de prueba.
        y_test: Objetivo de prueba.
        cv_folds: Número de pliegues para validación cruzada.
        
    Return:
        Tuple: (DataFrame de métricas, Diccionario de resultados para graficar)
    """
    metrics_list = []
    results = {}
    
    from processing import create_pipeline
    
    for name, model in models.items():
        # Crear Pipeline
        pipeline = create_pipeline(model)
        
        # Validación Cruzada (en conjunto de entrenamiento)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='accuracy')
        mean_cv_accuracy = np.mean(cv_scores)
        
        # Entrenar en todo el conjunto de entrenamiento
        pipeline.fit(X_train, y_train)
        
        # Predecir en conjunto de prueba
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.decision_function(X_test)
        
        # Calcular Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Agregar métricas al DataFrame
        metrics_list.append({
            'Model': name,
            'CV Accuracy': mean_cv_accuracy,
            'Test Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        # Agregar resultados al diccionario
        results[name] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
    return pd.DataFrame(metrics_list), results

def perform_grid_search(model: Any, param_grid: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Realiza una búsqueda de cuadrícula (Grid Search) para optimizar hiperparámetros.
    
    Args:
        model: El modelo o pipeline a optimizar.
        param_grid: Diccionario con los nombres de los parámetros y listas de valores a probar.
        X_train: Características de entrenamiento.
        y_train: Objetivo de entrenamiento.
        cv: Número de pliegues para validación cruzada.
        
    Returns:
        Tuple: (Mejor Modelo, Mejores Parámetros, Objeto GridSearch completo)
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search
