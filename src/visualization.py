import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
from typing import List, Dict, Optional, Any

# --- Configuración y Estilo ---
COLORS = {
    "primary": "#2962FF",  
    "secondary": "#FF0055", 
    "accent": "#00C853",   
    "yellow": "#FFD600",    
    "purple": "#AA00FF",    
    "background": "#FFFFFF",
    "grid": "#E0E0E0",
    "text": "#000000"
}

def set_style():
    """Establece el estilo estético personalizado para todos los gráficos."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['figure.facecolor'] = COLORS['background']
    plt.rcParams['axes.facecolor'] = COLORS['background']
    plt.rcParams['axes.edgecolor'] = COLORS['text']
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.color'] = COLORS['grid']
    plt.rcParams['xtick.color'] = COLORS['text']
    plt.rcParams['ytick.color'] = COLORS['text']
    plt.rcParams['text.color'] = COLORS['text']
    plt.rcParams['axes.labelcolor'] = COLORS['text']
    plt.rcParams['axes.titlecolor'] = COLORS['text']
    
    # Configurar paleta con colores vibrantes
    sns.set_palette([COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['yellow'], COLORS['purple']])

def plot_distributions(df: pd.DataFrame, columns: List[str], target: str):
    """
    Grafica la distribución de características agrupadas por objetivo usando KDE y Boxplots.
    
    Args:
        df: DataFrame de entrada.
        columns: Lista de nombres de columnas para graficar.
        target: Nombre de la columna objetivo para agrupar.
    """
    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols, 2, figsize=(14, 4 * n_cols))
    
    for i, col in enumerate(columns):
        # KDE Plot
        sns.kdeplot(data=df, x=col, hue=target, fill=True, ax=axes[i, 0], 
                    palette=[COLORS['primary'], COLORS['accent']], alpha=0.6)
        axes[i, 0].set_title(f'Distribución de {col}', fontweight='bold')
        axes[i, 0].set_xlabel('')
        
        # Box Plot
        sns.boxplot(data=df, x=target, y=col, hue=target, legend=False, ax=axes[i, 1],
                    palette=[COLORS['primary'], COLORS['accent']])
        axes[i, 1].set_title(f'{col} por Target', fontweight='bold')
        axes[i, 1].set_xlabel('')
        
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Grafica un mapa de calor de correlación limpio.
    
    Args:
        df: DataFrame de entrada.
    """
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=False)
    
    plt.title('Matriz de Correlación de Características', fontsize=16, fontweight='bold')
    plt.show()

def plot_model_comparison(results: Dict[str, Dict]):
    """
    Grafica Matrices de Confusión y Curvas ROC para múltiples modelos.
    
    Args:
        results: Diccionario donde las claves son nombres de modelos y los valores contienen 
                 'y_true', 'y_pred', 'y_prob'.
    """
    models = list(results.keys())
    n_models = len(models)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, n_models)
    
    # Matrices de Confusión
    for i, model_name in enumerate(models):
        ax = fig.add_subplot(gs[0, i])
        y_true = results[model_name]['y_true']
        y_pred = results[model_name]['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', cbar=False, ax=ax)
        ax.set_title(f'Matriz de Confusión: {model_name}', fontweight='bold')
        ax.set_xlabel('Predecido')
        ax.set_ylabel('Real')

    # Curvas ROC (Combinadas)
    ax_roc = fig.add_subplot(gs[1, :])
    for model_name in models:
        y_true = results[model_name]['y_true']
        y_prob = results[model_name]['y_prob']
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax_roc.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('Tasa de Falso Positivo')
    ax_roc.set_ylabel('Tasa de Verdadero Positivo')
    ax_roc.set_title('Curva ROC', fontweight='bold')
    ax_roc.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()

def plot_selected_pairplot(df: pd.DataFrame, columns: List[str], target: str):
    """
    Grafica un pairplot enfocado solo en las columnas seleccionadas para evitar gráficos ilegibles.
    
    Args:
        df: DataFrame de entrada.
        columns: Lista de nombres de columnas para incluir en el pairplot.
        target: Nombre de la columna objetivo para el color (hue).
    """
    # Filtrar el DataFrame para incluir solo columnas seleccionadas y el target
    subset_df = df[columns + [target]]
    
    sns.pairplot(subset_df, hue=target, palette=[COLORS['primary'], COLORS['accent']], 
                 diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
    
    plt.suptitle(f'Pairplot de Características Seleccionadas', y=1.02, fontsize=16, fontweight='bold')
    plt.show()

def plot_learning_curves(models: Dict[str, Any], X: pd.DataFrame, y: pd.Series, cv: int = 5):
    """
    Calcula y grafica las curvas de aprendizaje para diagnosticar Bias vs Variance.
    
    Args:
        models: Diccionario de modelos instanciados.
        X: Características.
        y: Objetivo.
        cv: Número de pliegues para validación cruzada.
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    
    if n_models == 1:
        axes = [axes]
    
    for i, (name, model) in enumerate(models.items()):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        ax = axes[i]
        ax.set_title(f'Curva de Aprendizaje: {name}', fontweight='bold')
        ax.set_xlabel("Ejemplos de Entrenamiento")
        ax.set_ylabel("Accuracy")
        
        # Plot training scores
        ax.plot(train_sizes, train_scores_mean, 'o-', color=COLORS['primary'], label="Entrenamiento")
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color=COLORS['primary'])
        
        # Plot validation scores
        ax.plot(train_sizes, test_scores_mean, 'o-', color=COLORS['accent'], label="Validación Cruzada")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color=COLORS['accent'])
        
        ax.grid(True)
        ax.legend(loc="best")
    
    plt.tight_layout()
    plt.show()


