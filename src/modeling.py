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

def perform_optuna_study(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50, cv: int = 5) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Realiza un estudio de Optuna para optimizar hiperparámetros.
    
    Args:
        model_name: Nombre del modelo ('Logistic Regression', 'Random Forest', etc.).
        X_train: Características de entrenamiento.
        y_train: Objetivo de entrenamiento.
        n_trials: Número de intentos (trials).
        cv: Número de pliegues para validación cruzada.
        
    Returns:
        Tuple: (Mejor Modelo, Mejores Parámetros, Estudio de Optuna)
    """
    import optuna
    from processing import create_pipeline
    
    def objective(trial):
        # Definir espacio de búsqueda según el modelo
        if model_name == 'Logistic Regression':
            params = {
                'C': trial.suggest_float('C', 1e-4, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'liblinear'
            }
            model = LogisticRegression(random_state=42, **params)
        else:
            raise ValueError(f"Modelo '{model_name}' no soportado para optimización con Optuna.")

        # Crear Pipeline
        pipeline = create_pipeline(model)
        
        # Validación Cruzada
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Re-entrenar el mejor modelo
    best_params = study.best_params
    
    # Reconstruir el modelo con los mejores parámetros (lógica duplicada necesaria para instanciar)
    if model_name == 'Logistic Regression':
        best_params['solver'] = 'liblinear' # Asegurar solver
        best_model = LogisticRegression(random_state=42, **best_params)
    elif model_name == 'Random Forest':
        best_model = RandomForestClassifier(random_state=42, **best_params)
    elif model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(random_state=42, **best_params)
    elif 'SVM' in model_name:
        if 'Linear' in model_name: best_params['kernel'] = 'linear'
        elif 'RBF' in model_name: best_params['kernel'] = 'rbf'
        best_model = SVC(probability=True, random_state=42, **best_params)
        
    # Ajustar el mejor modelo con todos los datos de entrenamiento (usando pipeline)
    from processing import create_pipeline
    best_pipeline = create_pipeline(best_model)
    best_pipeline.fit(X_train, y_train)
    
    return best_pipeline, best_params, study
