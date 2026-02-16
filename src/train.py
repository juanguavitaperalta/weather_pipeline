import os
import logging

# Crear carpeta logs si no existe
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    handlers=[
        logging.FileHandler("logs/train.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Workaround típico para conflicto OpenMP en Windows (MKL/oneDNN)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Opcional: menos ruido en logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Opcional: apagar oneDNN si quieres reproducibilidad total
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
print("TensorFlow OK:", tf.__version__)




from sklearn.linear_model import Lasso, Ridge, ElasticNet
import pandas as pd
import logging
from pathlib import Path
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib
import json
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def root_mean_squared_error(y_true, y_pred):
    """Calcula RMSE para compatibilidad con versiones antiguas de sklearn"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def leer_archivo_csv(ruta_archivo: str) -> pd.DataFrame:
    try: 
        df = pd.read_csv(ruta_archivo)
        logger.info(f"El archivo {ruta_archivo} fue leído correctamente")
        print(df.head())
        print(df.isna().sum())
        # df.dtypes para temas de tipos de datos
    except FileNotFoundError:
        logger.error(f"El archivo {ruta_archivo} no fue encontrado.")
        return pd.DataFrame()  
    return df

def plot_ridge_bias_variance(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, ruta_salida: str = "reports/figures/curvas aprendizaje/ridge_bias_variance.png"):
    """
    Grafica lambda vs varianza, bias^2 y test squared error para Ridge.
    """
    alphas = np.logspace(-8, 8, 40)
    n_repeats = 10
    preds = np.zeros((n_repeats, len(alphas), len(y_test)))
    for rep in range(n_repeats):
        for i, a in enumerate(alphas):
            ridge = Ridge(alpha=a, fit_intercept=True, max_iter=20000)
            ridge.fit(x_train, y_train)
            preds[rep, i, :] = ridge.predict(x_test)
    # Calcular bias^2, varianza y error cuadrático medio
    mean_preds = preds.mean(axis=0)
    bias2 = np.mean((mean_preds - y_test.values) ** 2, axis=1)
    variance = np.mean(np.var(preds, axis=0), axis=1)
    test_error = np.mean((preds - y_test.values) ** 2, axis=(0,2))
    # Graficar
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, bias2, label="Bias$^2$")
    plt.plot(alphas, variance, label="Varianza")
    plt.plot(alphas, test_error, label="Test Squared Error")
    plt.xscale("log")
    plt.xlabel("Lambda (alpha)")
    plt.ylabel("Error")
    plt.title("Bias$^2$, Varianza y Test Squared Error vs Lambda (Ridge)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Curva bias^2, varianza y test squared error guardada en: {ruta_salida}")


def plot_ridge_coefs_vs_lambda(x_train: pd.DataFrame, y_train: pd.Series, ruta_salida: str = "reports/figures/curvas aprendizaje/ridge_coefs_vs_lambda.png"):
    """
    Grafica la evolución de los coeficientes de Ridge en función de lambda (alpha).
    """
    alphas = np.logspace(-8, 8, 100)
    coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=True, max_iter=20000)
        ridge.fit(x_train, y_train)
        coefs.append(ridge.coef_)
    coefs = np.array(coefs)
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[1]):
        plt.plot(alphas, coefs[:, i], label=f"Coef {i+1}")
    plt.xscale("log")
    plt.xlabel("Lambda (alpha)")
    plt.ylabel("Coeficiente")
    plt.title("Curva de coeficientes vs Lambda para Ridge")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Curva de coeficientes vs lambda guardada en: {ruta_salida}")


def dividir_train_test(df:pd.DataFrame, objective_col:str) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    df["time"]= pd.to_datetime(df["time"], errors='coerce')
    df = df.dropna(subset=["time"])
    df = df.sort_values(by="time").reset_index(drop=True)

    y = df[objective_col]
    x = df.drop(columns=[objective_col,'time'])

    len_train = int(0.8 * len(df))
    x_train = x.iloc[:len_train]
    x_test = x.iloc[len_train:]    
    y_train = y.iloc[:len_train]
    y_test = y.iloc[len_train:] 

    return x_train, x_test, y_train, y_test

def dividir_train_test_lstm(df:pd.DataFrame, objective_col:str) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values(by="time").reset_index(drop=True)

    y = df[objective_col]
    x = df.drop(columns=[objective_col, "time"])

    n = len(df)
    test_size = 0.20
    val_frac_within_train = 0.15
    trainval_cut = int(np.floor((1 - test_size) * n))
    val_cut = int(np.floor((1 - val_frac_within_train) * trainval_cut))

    x_train = x.iloc[:val_cut]
    y_train = y.iloc[:val_cut]
    x_val = x.iloc[val_cut:trainval_cut]
    y_val = y.iloc[val_cut:trainval_cut]
    x_test = x.iloc[trainval_cut:]
    y_test = y.iloc[trainval_cut:]

    return x_train, y_train, x_val, y_val, x_test, y_test
   
   

def modelos_lineales(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
       
    tscv = TimeSeriesSplit(n_splits=5)
    alphas = np.logspace(-8, 8, 20)

    models = {
        "Lasso": (
            Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(max_iter=20000))]), {"lasso__alpha": alphas}
            ),       
        "Ridge": (
            Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())]), {"ridge__alpha": alphas}
            ),
        "ElasticNet": (
            Pipeline([('scaler', StandardScaler()), ('elasticnet', ElasticNet(max_iter=20000))]),
            {"elasticnet__alpha": np.logspace(-6, 6, 13), "elasticnet__l1_ratio": [0.2, 0.5, 0.8]})
    }

    scoring = {'RMSE': 'neg_root_mean_squared_error', 'MAE': 'neg_mean_absolute_error'}

    results = []
    best_models = {}

    for model_name, (model, param_grid) in models.items():
        logger.info(f"Entrenando modelo: {model_name}")
        gscv = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring=scoring,refit='RMSE', n_jobs=-1)
        gscv.fit(x_train, y_train)
        
        best_model = gscv.best_estimator_   
        logger.info(f"Mejor modelo de CV: {model_name} con parámetros {gscv.best_params_}")
        logger.info(f"Mejor RMSE de CV: {-gscv.best_score_}")
        
        y_pred = best_model.predict(x_test)
        test_rmse = root_mean_squared_error(y_test, y_pred)
        logger.info(f"RMSE en el conjunto de prueba para {model_name}: {test_rmse}")
        test_mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"MAE en el conjunto de prueba para {model_name}: {test_mae}")
        best_i = gscv.best_index_
        logger.info(f"Resultados detallados del mejor modelo en el conjunto de validación cruzada:")
        cv_rmse = -gscv.cv_results_['mean_test_RMSE'][best_i]
        cv_mae = -gscv.cv_results_['mean_test_MAE'][best_i]
        logger.info(f"RMSE de validación cruzada: {cv_rmse}")   
        logger.info(f"MAE de validación cruzada: {cv_mae}") 

        best_models[model_name] = best_model

        results.append({
            "model": model_name,
            "cv_RMSE": cv_rmse,
            "cv_MAE": cv_mae,
            "test_RMSE": test_rmse,
            "test_MAE": test_mae,
            "best_params": gscv.best_params_
        })


    dataframe_results = pd.DataFrame(results)
    # Graficar curva de coeficientes vs lambda para Ridge
    if "Ridge" in best_models:
        plot_ridge_coefs_vs_lambda(x_train, y_train)
        plot_ridge_bias_variance(x_train, y_train, x_test, y_test)
    return dataframe_results, best_models

def columnas_estacionalidad(df: pd.DataFrame) -> pd.DataFrame:
    df2=df.copy()
    df2["time"] = pd.to_datetime(df2["time"], errors="coerce")  
    df2["dayofweek"] = df2["time"].dt.dayofweek
    df2["month"] = df2["time"].dt.month
    df2["hour"] = df2["time"].dt.hour

    df2["sin_comp"] = np.sin(2 * np.pi * df2["hour"] / 24)
    df2["cos_comp"] = np.cos(2 * np.pi * df2["hour"] / 24)
    return df2

def last_block_split(X, y, val_size=0.15):
    n = len(X)
    cut = int(np.floor((1 - val_size) * n))
    X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
    X_va, y_va = X.iloc[cut:], y.iloc[cut:]
    return X_tr, y_tr, X_va, y_va
                        

def entrenar_xgboost(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    base_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=5000,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
    "max_depth": [3,4,5,6],
    "min_child_weight": [1,3,5,10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.5, 1.0, 2.0],
    "reg_alpha": [0.0, 1e-3, 1e-2, 1e-1],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}

    search = RandomizedSearchCV(
        estimator=base_model, 
        param_distributions=param_dist, 
        n_iter=40,
        scoring=scoring,
        refit="RMSE",
        cv=tscv, 
        verbose=2, 
        random_state=42, 
        n_jobs=-1
    )

    logger.info("Iniciando búsqueda aleatoria de hiperparámetros para XGBoost.")
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    logger.info(f"Mejor modelo XGBoost con parámetros: {search.best_params_}")
    y_pred = best_model.predict(x_test)
    test_rmse = root_mean_squared_error(y_test, y_pred) 
    test_mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"RMSE en el conjunto de prueba para XGBoost: {test_rmse}")
    logger.info(f"MAE en el conjunto de prueba para XGBoost: {test_mae}")
    best_i = search.best_index_
    cv_rmse = -search.cv_results_["mean_test_RMSE"][best_i]
    cv_mae = -search.cv_results_["mean_test_MAE"][best_i]

    logger.info(f"RMSE de validación cruzada para XGBoost: {cv_rmse}")
    logger.info(f"MAE de validación cruzada para XGBoost: {cv_mae}")
    best_params = search.best_params_

    results = pd.DataFrame([{
        "model": "XGBoost",
        "cv_RMSE": cv_rmse,
        "cv_MAE": cv_mae,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "best_params": best_params
    }])

    best_models = {"XGBoost": best_model}

    return results, best_models

def comparar_modelos(resultados: pd.DataFrame) -> pd.DataFrame:
    return resultados.sort_values(by="test_RMSE").reset_index(drop=True)


def guardar_modelo(modelo, ruta_modelo: str = "models/modelo_final.joblib") -> str:
    """
    Guarda el modelo entrenado usando joblib.
    
    Args:
        modelo: Modelo entrenado a guardar
        ruta_modelo: Ruta donde guardar el modelo
    
    Returns:
        Ruta donde se guardó el modelo
    """
    Path(ruta_modelo).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta_modelo)
    logger.info(f"Modelo guardado en: {ruta_modelo}")
    return ruta_modelo


def guardar_metadatos(metadatos: dict, ruta_metadatos: str) -> str:
    """
    Guarda los metadatos del modelo en formato JSON.
    
    Args:
        metadatos: Diccionario con los metadatos del modelo
        ruta_metadatos: Ruta donde guardar los metadatos
    
    Returns:
        Ruta donde se guardaron los metadatos
    """
    Path(ruta_metadatos).parent.mkdir(parents=True, exist_ok=True)
    
    # Convertir tipos numpy a tipos nativos de Python para JSON
    def convertir_tipos(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convertir_tipos(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convertir_tipos(item) for item in obj]
        return obj
    
    metadatos_convertidos = convertir_tipos(metadatos)
    
    with open(ruta_metadatos, 'w', encoding='utf-8') as f:
        json.dump(metadatos_convertidos, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Metadatos guardados en: {ruta_metadatos}")
    return ruta_metadatos


def cargar_modelo(ruta_modelo: str = "models/modelo_final.joblib"):
    """
    Carga un modelo previamente guardado con joblib.
    
    Args:
        ruta_modelo: Ruta del modelo a cargar
    
    Returns:
        Modelo cargado
    """
    modelo = joblib.load(ruta_modelo)
    logger.info(f"Modelo cargado desde: {ruta_modelo}")
    return modelo

def get_last_tss_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Devuelve el ÚLTIMO split de TimeSeriesSplit:
    - train_idx: todo el pasado (dentro del 80%)
    - val_idx: el bloque final (dentro del 80%)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    last_train_idx, last_val_idx = None, None
    for train_idx, val_idx in tscv.split(X):
        last_train_idx, last_val_idx = train_idx, val_idx
    return last_train_idx, last_val_idx

def xgb_learning_curve_purista(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, n_splits: int = 5,
    n_estimators: int = 1000, output_path: str = "reports/figures/curvas aprendizaje/xgb_n_estimators_curve.png",
    x_test: pd.DataFrame = None, y_test: pd.Series = None) -> dict:
    """
    Entrena 1 vez con best_params sobre el sub-train del último fold
    y evalúa en el valid interno del último fold. Extrae RMSE por iteración.
    """
    # 1) Split purista: valid interno dentro del TRAIN (no tocar TEST)
    tr_idx, val_idx = get_last_tss_split(x_train, y_train, n_splits=n_splits)
    X_tr, y_tr = x_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_val, y_val = x_train.iloc[val_idx], y_train.iloc[val_idx]

    # 2) Modelo final para la curva (best_params + muchos árboles)
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=n_estimators,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **best_params,  # <-- aquí entran max_depth, learning_rate, etc.
    )

    # 3) Fit con eval_set para que XGBoost guarde RMSE por iteración
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False
    )

    # 4) Extraer historia de RMSE por boosting round
    evals = model.evals_result()
    rmse_tr = evals["validation_0"]["rmse"]
    rmse_val = evals["validation_1"]["rmse"]
    rounds = np.arange(1, len(rmse_tr) + 1)

    # 5) Calcular test RMSE si se proporcionan datos de test
    test_rmse = None
    if x_test is not None and y_test is not None:
        y_pred_test = model.predict(x_test)
        test_rmse = root_mean_squared_error(y_test, y_pred_test)

    # 6) Plot n_estimators vs RMSE (train vs valid)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, rmse_tr, label="Train RMSE")
    plt.plot(rounds, rmse_val, label="Valid (inner) RMSE")
    
    # Agregar línea horizontal de test RMSE si está disponible
    if test_rmse is not None:
        plt.axhline(y=test_rmse, color='green', linestyle='--', linewidth=2, 
                    label=f"Test RMSE: {test_rmse:.4f}")
    
    plt.xlabel("n_estimators (boosting rounds)")
    plt.ylabel("RMSE")
    plt.title("XGBoost: Curva n_estimators vs RMSE (purista, valid interno del último fold)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 7) Punto óptimo (mínimo RMSE en valid interno)
    best_round = int(np.argmin(rmse_val) + 1)
    best_val_rmse = float(np.min(rmse_val))

    return {
        "best_round": best_round,
        "best_inner_val_rmse": best_val_rmse,
        "rmse_train_curve": rmse_tr,
        "rmse_val_curve": rmse_val,
        "test_rmse": test_rmse,
        "model_curve": model,  # modelo entrenado para la curva
        "output_path": output_path,
    }


def plot_curvas_aprendizaje(modelo, modelo_nombre: str, x_train: pd.DataFrame, y_train: pd.Series,
                            ruta_salida: str = "reports/figures/curva_aprendizaje.png") -> None:
    logger.info(f"Generando curva de aprendizaje para: {modelo_nombre}")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Tamaños de entrenamiento a evaluar
    train_sizes = np.linspace(0.2, 0.9, 8)
    
    # Calcular curvas de aprendizaje
    train_sizes_abs, train_scores, cv_scores = learning_curve(estimator=modelo,X=x_train,y=y_train,
        train_sizes=train_sizes, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1,
        shuffle=False)    
    # Convertir scores negativos a RMSE positivo
    train_rmse_mean = -train_scores.mean(axis=1)
    train_rmse_std = train_scores.std(axis=1)
    cv_rmse_mean = -cv_scores.mean(axis=1)
    cv_rmse_std = cv_scores.std(axis=1)
    
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    
    # Curva de entrenamiento
    plt.plot(train_sizes_abs, train_rmse_mean, 'o-', color='blue', 
             label='Training RMSE', linewidth=2, markersize=6)
    plt.fill_between(train_sizes_abs, 
                     train_rmse_mean - train_rmse_std,
                     train_rmse_mean + train_rmse_std, 
                     alpha=0.15, color='blue')
    
    # Curva de validación cruzada
    plt.plot(train_sizes_abs, cv_rmse_mean, 'o-', color='red', 
             label='Cross-Validation RMSE', linewidth=2, markersize=6)
    plt.fill_between(train_sizes_abs, 
                     cv_rmse_mean - cv_rmse_std,
                     cv_rmse_mean + cv_rmse_std, 
                     alpha=0.15, color='red')
    
    plt.xlabel('Tamaño del conjunto de entrenamiento', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(f'Curva de Aprendizaje - {modelo_nombre}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Crear directorio si no existe
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Curva de aprendizaje guardada en: {ruta_salida}")
    
    # Mostrar métricas finales
    logger.info(f"Training RMSE final: {train_rmse_mean[-1]:.4f} (+/- {train_rmse_std[-1]:.4f})")
    logger.info(f"CV RMSE final: {cv_rmse_mean[-1]:.4f} (+/- {cv_rmse_std[-1]:.4f})")  

def xgb_feature_importance_df(booster, feature_names):
    dfs = []
    for importance_type in ["gain", "cover", "weight"]:
        scores = booster.get_score(importance_type=importance_type)
        df = pd.DataFrame(
            scores.items(),
            columns=["feature", importance_type]
        )
        dfs.append(df)

    # Merge
    from functools import reduce
    df_imp = reduce(
        lambda left, right: pd.merge(left, right, on="feature", how="outer"),
        dfs
    ).fillna(0)

    # Normalizar gain (opcional pero recomendado)
    df_imp["gain_norm"] = df_imp["gain"] / df_imp["gain"].sum()

    return df_imp.sort_values("gain", ascending=False)


def plot_lstm_learning_curves(history, output_path: str = "reports/figures/curvas aprendizaje/lstm_learning_curves.png"):
    """
    Grafica las curvas de aprendizaje del modelo LSTM.
    
    Args:
        history: Historia del entrenamiento devuelta por model.fit()
        output_path: Ruta donde guardar la gráfica
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Configurar subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Curvas de Aprendizaje LSTM', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[0, 0].set_title('Loss vs Epochs', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE
    axes[0, 1].plot(history.history['rmse'], label='Train RMSE', color='blue', linewidth=2)
    axes[0, 1].plot(history.history['val_rmse'], label='Val RMSE', color='red', linewidth=2)
    axes[0, 1].set_title('RMSE vs Epochs', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE
    axes[1, 0].plot(history.history['mae'], label='Train MAE', color='blue', linewidth=2)
    axes[1, 0].plot(history.history['val_mae'], label='Val MAE', color='red', linewidth=2)
    axes[1, 0].set_title('MAE vs Epochs', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Convergence Analysis
    axes[1, 1].plot(history.history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axes[1, 1].axvline(x=len(history.history['val_loss'])-1, color='green', linestyle='--', 
                       label=f'Early Stop (Epoch {len(history.history["val_loss"])})')
    axes[1, 1].set_title('Convergencia (Validation Loss)', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Curvas de aprendizaje LSTM guardadas en: {output_path}")
    
    # También crear un resumen de métricas
    final_metrics = {
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_train_rmse': history.history['rmse'][-1],
        'final_val_rmse': history.history['val_rmse'][-1],
        'final_train_mae': history.history['mae'][-1],
        'final_val_mae': history.history['val_mae'][-1],
        'epochs_trained': len(history.history['loss']),
        'best_val_loss': min(history.history['val_loss']),
        'best_epoch': history.history['val_loss'].index(min(history.history['val_loss'])) + 1
    }
    
    return final_metrics


def analisis_shap(modelo, x_train: pd.DataFrame, x_test: pd.DataFrame, 
                  output_dir: str = "reports/figures/shap") -> dict:
    """
    Aplica análisis SHAP paso a paso para explicar el modelo XGBoost.
    
    Args:
        modelo: Modelo XGBoost entrenado
        x_train: Datos de entrenamiento
        x_test: Datos de prueba
        output_dir: Directorio para guardar las figuras
    
    Returns:
        Diccionario con SHAP values y métricas de importancia
    """
    try:
        import shap
    except ImportError as e:
        logger.error(f"Error al importar SHAP: {e}")
        return {}
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PASO 1: Crear el SHAP Explainer
    # =========================================================================
    logger.info("SHAP Paso 1: Creando TreeExplainer para XGBoost...")
    # TreeExplainer es óptimo para modelos basados en árboles (XGBoost, LightGBM, etc.)
    # Calcula valores SHAP exactos de manera eficiente
    explainer = shap.TreeExplainer(modelo)
    expected_val = explainer.expected_value
    if hasattr(expected_val, '__len__'):
        expected_val = expected_val[0] if len(expected_val) == 1 else expected_val
    logger.info(f"Explainer creado. Expected value (baseline): {float(expected_val):.4f}")
    
    # =========================================================================
    # PASO 2: Calcular SHAP Values
    # =========================================================================
    logger.info("SHAP Paso 2: Calculando SHAP values para conjunto de test...")
    # shap_values[i, j] = contribución del feature j a la predicción de la muestra i
    shap_values_test = explainer.shap_values(x_test)
    logger.info(f"SHAP values calculados. Shape: {shap_values_test.shape}")
    
    # También calcular para train (útil para análisis más completo)
    logger.info("Calculando SHAP values para conjunto de train (puede tomar más tiempo)...")
    shap_values_train = explainer.shap_values(x_train)
    
    # =========================================================================
    # PASO 3: Summary Plot - Importancia Global de Features
    # =========================================================================
    logger.info("SHAP Paso 3: Generando Summary Plot (importancia global)...")
    
    # 3a) Summary plot tipo "dot" - muestra distribución de impacto por feature
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_test, x_test, show=False, plot_size=(12, 8))
    plt.title("SHAP Summary Plot - Impacto de Features en Predicciones", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_dot.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Summary plot (dot) guardado en: {output_dir}/shap_summary_dot.png")
    
    # 3b) Summary plot tipo "bar" - importancia media absoluta
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_test, x_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (|SHAP| medio)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Summary plot (bar) guardado en: {output_dir}/shap_summary_bar.png")
    
    # =========================================================================
    # PASO 4: Dependence Plots - Relación Feature vs SHAP Value
    # =========================================================================
    logger.info("SHAP Paso 4: Generando Dependence Plots para top features...")
    
    # Calcular importancia media para identificar top features
    shap_importance = np.abs(shap_values_test).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': x_test.columns,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    
    # Generar dependence plots para los 4 features más importantes
    top_features = feature_importance_df['feature'].head(4).tolist()
    
    for i, feature in enumerate(top_features):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, 
            shap_values_test, 
            x_test,
            show=False
        )
        plt.title(f"SHAP Dependence Plot: {feature}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_dependence_{feature}.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Dependence plot para '{feature}' guardado")
    
    # =========================================================================
    # PASO 5: Waterfall Plot - Explicación de Predicciones Individuales
    # =========================================================================
    logger.info("SHAP Paso 5: Generando Waterfall Plots para predicciones individuales...")
    
    # Crear objeto Explanation para visualizaciones modernas
    explanation_test = shap.Explanation(
        values=shap_values_test,
        base_values=np.full(len(x_test), explainer.expected_value),
        data=x_test.values,
        feature_names=x_test.columns.tolist()
    )
    
    # Waterfall para las primeras 3 predicciones del test set
    for idx in range(min(3, len(x_test))):
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation_test[idx], show=False)
        plt.title(f"SHAP Waterfall - Predicción {idx+1}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_waterfall_pred_{idx+1}.png", dpi=150, bbox_inches='tight')
        plt.close()
    logger.info(f"Waterfall plots guardados para primeras 3 predicciones")
    
    # =========================================================================
    # PASO 6: Force Plot - Visualización Compacta
    # =========================================================================
    logger.info("SHAP Paso 6: Generando Force Plot (visualización compacta)...")
    
    # Force plot para una predicción individual (guardado como HTML)
    force_plot = shap.force_plot(
        explainer.expected_value, 
        shap_values_test[0], 
        x_test.iloc[0],
        matplotlib=False
    )
    shap.save_html(f"{output_dir}/shap_force_plot_single.html", force_plot)
    logger.info(f"Force plot individual guardado en: {output_dir}/shap_force_plot_single.html")
    
    # Force plot para múltiples predicciones
    force_plot_multi = shap.force_plot(
        explainer.expected_value, 
        shap_values_test[:50],  # Primeras 50 predicciones
        x_test.iloc[:50],
        matplotlib=False
    )
    shap.save_html(f"{output_dir}/shap_force_plot_multi.html", force_plot_multi)
    logger.info(f"Force plot múltiple guardado en: {output_dir}/shap_force_plot_multi.html")
    
    # =========================================================================
    # PASO 7: Beeswarm Plot - Alternativa moderna al Summary Plot
    # =========================================================================
    logger.info("SHAP Paso 7: Generando Beeswarm Plot...")
    
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(explanation_test, show=False)
    plt.title("SHAP Beeswarm Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Beeswarm plot guardado en: {output_dir}/shap_beeswarm.png")
    
    # =========================================================================
    # PASO 8: Guardar SHAP Values y Métricas
    # =========================================================================
    logger.info("SHAP Paso 8: Guardando SHAP values y métricas...")
    
    # Crear DataFrame con SHAP values
    shap_df = pd.DataFrame(
        shap_values_test, 
        columns=[f"shap_{col}" for col in x_test.columns]
    )
    shap_df.to_csv(f"{output_dir}/shap_values_test.csv", index=False)
    
    # Guardar importancia de features basada en SHAP
    feature_importance_df.to_csv(f"{output_dir}/shap_feature_importance.csv", index=False)
    
    logger.info("="*60)
    logger.info("ANÁLISIS SHAP COMPLETADO")
    logger.info("="*60)
    logger.info(f"\nTop 10 Features por SHAP Importance:")
    print(feature_importance_df.head(10).to_string(index=False))
    
    return {
        "explainer": explainer,
        "shap_values_test": shap_values_test,
        "shap_values_train": shap_values_train,
        "expected_value": explainer.expected_value,
        "feature_importance": feature_importance_df,
        "output_dir": output_dir
    }


def separar_datos_prediccion(df: pd.DataFrame, mes_prediccion: int = 6, ruta_salida: str = "data/predict_data/predict.xlsx") -> pd.DataFrame:
    """
    Separa los datos del mes de predicción y los guarda en un archivo Excel.
    Retorna el DataFrame sin el mes de predicción para entrenamiento.
    """
    df_copia = df.copy()
    df_copia["time"] = pd.to_datetime(df_copia["time"], errors="coerce")
    
    # Separar datos de junio (mes 6) para predicción
    df_predict = df_copia[df_copia["time"].dt.month == mes_prediccion].copy()
    df_train = df_copia[df_copia["time"].dt.month != mes_prediccion].copy()
    
    # Guardar datos de predicción en Excel
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_predict.to_excel(ruta_salida, index=False, engine='openpyxl')
    logger.info(f"Datos de predicción (mes {mes_prediccion}) guardados en {ruta_salida}")
    logger.info(f"Registros para predicción: {len(df_predict)}, Registros para entrenamiento: {len(df_train)}")
    
    return df_train

def separar_datos_prediccion_lstm(df: pd.DataFrame, mes_prediccion: int = 6, ruta_salida: str = "data/predict_data/predict.xlsx") -> pd.DataFrame:
    """
    Separa los datos del mes de predicción y los guarda en un archivo Excel.
    Retorna el DataFrame sin el mes de predicción para entrenamiento.
    """
    df_copia = df.copy()
    df_copia["time"] = pd.to_datetime(df_copia["time"], errors="coerce")

    # Separar datos de junio (mes 6) para predicción
    df_predict = df_copia[df_copia["time"].dt.month == mes_prediccion].copy()
    df_train = df_copia[df_copia["time"].dt.month != mes_prediccion].copy()

    # Incluir las 48 horas anteriores al primer día de junio en los datos de predicción
    if not df_predict.empty:
        primer_junio = df_predict["time"].min()
        if pd.notnull(primer_junio):
            limite = primer_junio - pd.Timedelta(hours=48)
            mask_48h = (df_train["time"] >= limite) & (df_train["time"] < primer_junio)
            df_48h = df_train[mask_48h].copy()
            # Añadir las 48h a los datos de predicción
            df_predict = pd.concat([df_48h, df_predict], ignore_index=True).sort_values("time").reset_index(drop=True)
            # Eliminar esas 48h del train
            df_train = df_train[~mask_48h]

    # Guardar datos de predicción en Excel
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_predict.to_excel(ruta_salida, index=False, engine='openpyxl')
    logger.info(f"Datos de predicción (mes {mes_prediccion} + 48h previas) guardados en {ruta_salida}")
    logger.info(f"Registros para predicción: {len(df_predict)}, Registros para entrenamiento: {len(df_train)}")

    return df_train

def lstm_window(df, columnas:list[str], target_col:str, window_size:int=48, horizon:int=3, time_col:str="time", dropna:bool=True)->tuple[np.ndarray, np.ndarray]:
    df_lstm = df.copy()
    if time_col in df_lstm.columns:
        df_lstm[time_col] = pd.to_datetime(df_lstm[time_col], errors="coerce")
        df_lstm = df_lstm.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    revision = columnas+[target_col]
    perdidas = [i for i in revision if i not in df_lstm.columns]
    if perdidas:
        raise ValueError(f"Las siguientes columnas no están en el DataFrame: {perdidas}")
    if dropna:
        df_lstm = df_lstm.dropna(subset=revision).reset_index(drop=True)

    x_values = df_lstm[columnas].to_numpy(dtype=np.float32)
    y_values = df_lstm[target_col].to_numpy(dtype=np.float32)
    t_values = df_lstm[time_col].to_numpy() if time_col in df_lstm.columns else np.arange(len(df_lstm))    

    n = len(df_lstm)
    ultimo_t = n-horizon-1

    x_list, y_list, t_list = [], [], []

    for i in range(window_size-1, ultimo_t+1):
        inicio = i - window_size + 1
        fin = i + 1  
        x_list.append(x_values[inicio:fin,:])
        y_list.append(y_values[i + horizon])
        t_list.append(t_values[i])

    x=np.stack(x_list) if x_list else np.empty((0, window_size, len(columnas)), dtype=np.float32)
    y=np.array(y_list, dtype=np.float32)
    t_index = np.array(t_list)

    return x,y,t_index

def scale_lstm(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, scaler = StandardScaler()) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    n_samples, window_size, n_features = x_train.shape
    x_train_reshaped = x_train.reshape(-1, n_features)
    x_val_reshaped = x_val.reshape(-1, n_features)
    x_test_reshaped = x_test.reshape(-1, n_features)

    scaler.fit(x_train_reshaped)
    x_train_scaled = scaler.transform(x_train_reshaped).reshape(n_samples, window_size, n_features)
    x_val_scaled = scaler.transform(x_val_reshaped).reshape(x_val.shape[0], window_size, n_features)
    x_test_scaled = scaler.transform(x_test_reshaped).reshape(x_test.shape[0], window_size, n_features)

    return x_train_scaled, x_val_scaled, x_test_scaled, scaler

def build_model(hparams: dict, input_shape: tuple) -> tuple:
    """
    Construye y compila un modelo LSTM con Conv1D y los hiperparámetros dados.
    
    Arquitectura:
        Input -> Conv1D -> LSTM #1 -> LSTM #2 -> LSTM #3 -> Dense (output)
    
    Args:
        hparams: Diccionario con hiperparámetros (filters, kernel_size, lstm1_units, 
                 lstm2_units, lstm3_units, dropout, recurrent_dropout, learning_rate, batch_size, etc.)
        input_shape: Tupla (window_size, n_features) para la forma de entrada
    
    Returns:
        tuple: (modelo compilado, batch_size)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, optimizers
    except ImportError as e:
        logger.error(f"Error al importar TensorFlow: {e}")
        logger.error("Para usar LSTM instala TensorFlow: pip install tensorflow")
        return None, None
    
    tf.keras.backend.clear_session()
    
    # Extraer hiperparámetros
    filters = hparams.get('filters', 32)
    kernel_size = hparams.get('kernel_size', 3)
    lstm1_units = hparams.get('lstm1_units', 64)
    lstm2_units = hparams.get('lstm2_units', 32)
    lstm3_units = hparams.get('lstm3_units', 16)  # Nueva capa LSTM
    dropout = hparams.get('dropout', 0.2)
    recurrent_dropout = hparams.get('recurrent_dropout', 0.1)
    learning_rate = hparams.get('learning_rate', 1e-3)
    batch_size = hparams.get('batch_size', 32)
    use_causal_padding = hparams.get('use_causal_padding', False)
    optimizer_name = hparams.get('optimizer', 'adam')
    
    # Construir modelo secuencial
    modelo = models.Sequential()
    modelo.add(layers.Input(shape=input_shape))
    
    # Capa Conv1D
    padding = 'causal' if use_causal_padding else 'same'
    modelo.add(layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        padding=padding
    ))
    
    # LSTM #1 (return_sequences=True para pasar a la siguiente LSTM)
    modelo.add(layers.LSTM(
        units=lstm1_units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    ))
    
    # LSTM #2 (return_sequences=True para pasar a LSTM #3)
    modelo.add(layers.LSTM(
        units=lstm2_units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    ))
    
    # LSTM #3 (return_sequences=False para salida final)
    modelo.add(layers.LSTM(
        units=lstm3_units,
        return_sequences=False,
        dropout=dropout
    ))
    
    # Capa Dense de salida (regresión)
    modelo.add(layers.Dense(1, activation='linear'))
    
    # Compilar modelo con el optimizador seleccionado
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'adamw':
        optimizer = optimizers.AdamW(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        logger.warning(f"Optimizador '{optimizer_name}' no reconocido. Usando Adam por defecto.")
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    modelo.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae")
        ],
    )
    
    return modelo, batch_size


def train_lstm(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
               n_trials, epochs, patience: int = 40, 
               verbose: int = 1, output_path: str = "models/lstm_final.h5",
               optuna_db: str = "sqlite:///models/optuna_lstm.db"):
    """
    Entrena un modelo LSTM usando Optuna para optimización de hiperparámetros.
    
    Args:
        x_train: Datos de entrenamiento (samples, window_size, features)
        y_train: Target de entrenamiento
        x_val: Datos de validación
        y_val: Target de validación
        n_trials: Número de trials para Optuna
        epochs: Épocas máximas por trial
        patience: Paciencia para early stopping
        verbose: Nivel de verbosidad
        output_path: Ruta para guardar el mejor modelo
        optuna_db: Ruta de la base de datos de Optuna
    
    Returns:
        tuple: (mejor_modelo, mejor_history, study)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import callbacks
        import optuna
        from optuna.integration import TFKerasPruningCallback
    except ImportError as e:
        logger.error(f"Error al importar dependencias: {e}")
        logger.error("Instala las dependencias: pip install tensorflow optuna")
        return None, None, None
    
    input_shape = (x_train.shape[1], x_train.shape[2])
    
    # Variable para guardar el mejor modelo y su historia
    best_model_info = {'model': None, 'history': None, 'val_loss': float('inf')}
    
    def objective(trial):
        """Función objetivo para Optuna"""
        tf.keras.backend.clear_session()
        
        # Muestrear hiperparámetros para arquitectura Conv1D + LSTM
        hparams = {
            'filters': trial.suggest_int('filters', 16, 64, step=16),
            'kernel_size': trial.suggest_int('kernel_size', 2, 5),
            'lstm1_units': trial.suggest_int('lstm1_units', 32, 128, step=32),
            'lstm2_units': trial.suggest_int('lstm2_units', 16, 64, step=16),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.0, 0.2),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'use_causal_padding': trial.suggest_categorical('use_causal_padding', [True, False]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop', 'sgd'])
        }
        
        # Construir modelo
        model, batch_size = build_model(hparams, input_shape)
        
        if model is None:
            return float('inf')
        
        # Callbacks
        # Nota: Pruning no está soportado en optimización multiobjetivo
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor="val_mae",
                patience=patience,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Entrenar modelo
        try:
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                callbacks=callbacks_list,
                verbose=0 if verbose == 0 else 1
            )
            
            # Obtener el mejor val_loss
            val_loss = min(history.history['val_loss'])
            
            # Guardar el mejor modelo globalmente
            if val_loss < best_model_info['val_loss']:
                best_model_info['model'] = model
                best_model_info['history'] = history
                best_model_info['val_loss'] = val_loss
                logger.info(f"Nuevo mejor modelo encontrado: val_loss = {val_loss:.6f}")
            
            return val_loss
            
        except Exception as e:
            logger.error(f"Error durante entrenamiento del trial {trial.number}: {e}")
            return float('inf')
    
    # Crear y ejecutar estudio Optuna
    logger.info("="*70)
    logger.info("INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA")
    logger.info("="*70)
    
    # Crear un nombre de estudio único por timestamp para no acumular trials pasados
    study_name = f"lstm_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name=study_name,
        storage=optuna_db,
        load_if_exists=False
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Resultados de la optimización
    logger.info("="*70)
    logger.info("OPTIMIZACIÓN COMPLETADA")
    logger.info("="*70)
    logger.info(f"Mejor trial: {study.best_trial.number}")
    logger.info(f"Mejor val_loss: {study.best_value:.6f}")
    logger.info(f"Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Guardar el mejor modelo
    if best_model_info['model'] is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        best_model_info['model'].save(output_path)
        logger.info(f"Mejor modelo guardado en: {output_path}")
        
        # Guardar mejores hiperparámetros en JSON
        best_params_path = output_path.replace('.h5', '_best_params.json').replace('.keras', '_best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_params': study.best_params,
                'best_value': study.best_value,
                'best_trial': study.best_trial.number,
                'n_trials': len(study.trials),
                'timestamp': datetime.now().isoformat()
            }, f, indent=4)
        logger.info(f"Mejores parámetros guardados en: {best_params_path}")
        
        # Guardar resumen del estudio
        study_summary_path = output_path.replace('.h5', '_optuna_study.json').replace('.keras', '_optuna_study.json')
        study_summary = {
            'best_trial_number': study.best_trial.number,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'trials_summary': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        with open(study_summary_path, 'w') as f:
            json.dump(study_summary, f, indent=4)
        logger.info(f"Resumen del estudio guardado en: {study_summary_path}")
    else:
        logger.warning("No se pudo entrenar ningún modelo exitosamente")
    
    return best_model_info['model'], best_model_info['history'], study


def train_lstm_multiobj(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                        n_trials: int, epochs: int, patience: int = 40, 
                        verbose: int = 1, output_path: str = "models/lstm_final.h5",
                        optuna_db: str = "sqlite:///models/optuna_lstm_multiobj.db",
                        use_mae_loss: bool = True):
    """
    Entrena LSTM con optimización MULTIOBJETIVO (MAE + RMSE).
    
    Args:
        x_train: Datos de entrenamiento normalizados (samples, window_size, features)
        y_train: Target de entrenamiento NORMALIZADO
        x_val: Datos de validación normalizados
        y_val: Target de validación NORMALIZADO
        n_trials: Número de trials para Optuna
        epochs: Épocas máximas por trial
        patience: Paciencia para early stopping
        verbose: Nivel de verbosidad
        output_path: Ruta para guardar el mejor modelo
        optuna_db: Ruta de la base de datos de Optuna
        use_mae_loss: Si True usa loss="mae", si False usa Huber loss
    
    Returns:
        tuple: (mejor_modelo, mejor_history, study, pareto_trials)
    
    Nota: 
    - Las métricas MAE y RMSE se calculan en escala REAL (°C) 
      desnormalizando internamente las predicciones.
    - Pruning no está disponible para optimización multiobjetivo en Optuna.
      Se usa solo EarlyStopping para evitar overfitting.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import callbacks
        import optuna
    except ImportError as e:
        logger.error(f"Error al importar dependencias: {e}")
        logger.error("Instala las dependencias: pip install tensorflow optuna")
        return None, None, None, []
    
    input_shape = (x_train.shape[1], x_train.shape[2])
    
    # Calcular estadísticas para desnormalización (y_train está normalizado)
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    logger.info(f"Estadísticas y: mean={mean_y:.4f}, std={std_y:.4f}")
    
    # Variable para guardar el mejor modelo (por MAE)
    best_model_info = {'model': None, 'history': None, 'mae': float('inf'), 'rmse': float('inf')}
    
    def objective(trial):
        """Función objetivo multiobjetivo: minimizar (MAE, RMSE) en validación"""
        tf.keras.backend.clear_session()
        
        # Hiperparámetros expandidos para competir con XGBoost
        hparams = {
            'filters': trial.suggest_int('filters', 32, 128, step=16),  # Más filtros
            'kernel_size': trial.suggest_int('kernel_size', 2, 7),  # Kernel más grande
            'lstm1_units': trial.suggest_int('lstm1_units', 64, 256, step=32),  # Más capacidad
            'lstm2_units': trial.suggest_int('lstm2_units', 32, 128, step=16),  # Más capacidad
            'lstm3_units': trial.suggest_int('lstm3_units', 16, 64, step=16),  # Tercera capa LSTM
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),  # Mayor regularización
            'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.0, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True),  # LR más bajo
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),  # Batch más grande
            'use_causal_padding': trial.suggest_categorical('use_causal_padding', [True, False]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop'])
        }
        
        # Construir modelo
        model, batch_size = build_model(hparams, input_shape)
        if model is None:
            return float('inf'), float('inf')
        
        # Re-compilar con loss alineada a MAE
        if use_mae_loss:
            loss_fn = "mae"
        else:
            loss_fn = tf.keras.losses.Huber(delta=1.0)  # Alternativa robusta
        
        # Obtener optimizer ya configurado
        opt = model.optimizer
        
        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                tf.keras.metrics.MeanAbsoluteError(name="mae")
            ]
        )
        
        # Callback para loguear por epoch
        class LogEpochCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logger.info(f"Trial {trial.number} - Epoch {epoch+1}: "
                            f"loss={logs.get('loss'):.4f}, val_loss={logs.get('val_loss'):.4f}, "
                            f"mae={logs.get('mae'):.4f}, val_mae={logs.get('val_mae'):.4f}, "
                            f"rmse={logs.get('rmse'):.4f}, val_rmse={logs.get('val_rmse'):.4f}")
        
        # Callbacks
        # Nota: Pruning no está soportado en optimización multiobjetivo
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor="val_mae",
                patience=patience,
                restore_best_weights=True,
                verbose=0
            ),
            LogEpochCallback()
        ]
        
        # Entrenar
        try:
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                callbacks=callbacks_list,
                verbose=0 if verbose == 0 else 1
            )
        except Exception as e:
            logger.error(f"Error en trial {trial.number}: {e}")
            return float('inf'), float('inf')
        
        # Evaluar en validación
        y_pred_norm = model.predict(x_val, verbose=0).reshape(-1)
        
        # Desnormalizar predicciones y targets a escala real (°C)
        y_pred_real = y_pred_norm * std_y + mean_y
        y_val_real = y_val * std_y + mean_y
        
        # Métricas finales en escala real (°C)
        val_mae = float(np.mean(np.abs(y_val_real - y_pred_real)))
        val_rmse = float(np.sqrt(np.mean((y_val_real - y_pred_real) ** 2)))
        
        # Guardar mejor modelo (por MAE)
        if val_mae < best_model_info['mae']:
            best_model_info['model'] = model
            best_model_info['history'] = history
            best_model_info['mae'] = val_mae
            best_model_info['rmse'] = val_rmse
            logger.info(f"Nuevo mejor modelo: Trial {trial.number}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}")
        
        # Logging
        train_mae = history.history['mae'][-1] if 'mae' in history.history else 0.0
        gap = abs(train_mae - val_mae)
        logger.info(f"Trial {trial.number}: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, gap={gap:.4f}")
        
        return val_mae, val_rmse  # Tupla multiobjetivo
    
    # Crear estudio multiobjetivo
    logger.info("="*70)
    logger.info("OPTIMIZACIÓN MULTIOBJETIVO CON OPTUNA (MAE + RMSE)")
    logger.info("="*70)
    
    study_name = f"lstm_multiobj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        directions=["minimize", "minimize"],  # Minimizar MAE y RMSE
        # Nota: Pruner no se utiliza en multi-objetivo (no soportado por Optuna)
        study_name=study_name,
        storage=optuna_db,
        load_if_exists=False
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Resultados
    logger.info("="*70)
    logger.info("OPTIMIZACIÓN COMPLETADA")
    logger.info("="*70)
    
    # Pareto Front
    pareto_trials = study.best_trials
    logger.info(f"Pareto Front contiene {len(pareto_trials)} soluciones óptimas:")
    for t in pareto_trials:
        mae, rmse = t.values
        logger.info(f"  Trial {t.number}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # Seleccionar mejor por MAE (competir con XGBoost)
    best_trial = min(pareto_trials, key=lambda t: t.values[0])
    logger.info(f"\nMejor por MAE: Trial {best_trial.number}")
    logger.info(f"  MAE: {best_trial.values[0]:.4f}")
    logger.info(f"  RMSE: {best_trial.values[1]:.4f}")
    logger.info("Mejores hiperparámetros:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Guardar modelo
    if best_model_info['model'] is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        best_model_info['model'].save(output_path)
        logger.info(f"Mejor modelo guardado en: {output_path}")
        
        # Guardar metadatos
        best_params_path = output_path.replace('.h5', '_best_params.json').replace('.keras', '_best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump({
                'optimization_type': 'multiobjetivo',
                'objectives': ['MAE', 'RMSE'],
                'best_trial': best_trial.number,
                'best_mae': best_trial.values[0],
                'best_rmse': best_trial.values[1],
                'best_params': best_trial.params,
                'n_pareto_solutions': len(pareto_trials),
                'n_trials': len(study.trials),
                'timestamp': datetime.now().isoformat()
            }, f, indent=4)
        logger.info(f"Metadatos guardados en: {best_params_path}")
        
        # Guardar Pareto Front
        pareto_path = output_path.replace('.h5', '_pareto.json').replace('.keras', '_pareto.json')
        pareto_data = {
            'pareto_trials': [
                {
                    'trial_number': t.number,
                    'mae': t.values[0],
                    'rmse': t.values[1],
                    'params': t.params
                }
                for t in pareto_trials
            ]
        }
        with open(pareto_path, 'w') as f:
            json.dump(pareto_data, f, indent=4)
        logger.info(f"Pareto Front guardado en: {pareto_path}")
        
        # Visualizar Pareto Front
        _plot_pareto_front(study, output_dir="reports/figures/optuna")
    else:
        logger.warning("No se entrenó ningún modelo exitosamente")
    
    return best_model_info['model'], best_model_info['history'], study, pareto_trials


def _plot_pareto_front(study, output_dir: str = "reports/figures/optuna"):
    """Visualiza el Pareto Front de un estudio multiobjetivo"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extraer datos
    all_trials = [t for t in study.trials if t.values[0] != float('inf')]
    pareto_trials = study.best_trials
    
    if len(all_trials) == 0:
        logger.warning("No hay trials válidos para graficar Pareto Front")
        return
    
    all_mae = [t.values[0] for t in all_trials]
    all_rmse = [t.values[1] for t in all_trials]
    pareto_mae = [t.values[0] for t in pareto_trials]
    pareto_rmse = [t.values[1] for t in pareto_trials]
    
    # Mejor por MAE
    best_mae_trial = min(pareto_trials, key=lambda t: t.values[0])
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.scatter(all_mae, all_rmse, alpha=0.5, s=50, label='Todos los trials')
    plt.scatter(pareto_mae, pareto_rmse, color='red', s=100, marker='*', 
                label='Pareto Front', zorder=5)
    plt.scatter([best_mae_trial.values[0]], [best_mae_trial.values[1]], 
                color='green', s=200, marker='X', label=f'Mejor MAE (Trial {best_mae_trial.number})', zorder=6)
    
    plt.xlabel('MAE (°C)', fontsize=12)
    plt.ylabel('RMSE (°C)', fontsize=12)
    plt.title('Pareto Front: MAE vs RMSE', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = f"{output_dir}/pareto_front_multiobj.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Pareto Front visualizado en: {output_path}")


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray, *, prefix: str = "test") -> Dict[str, float]:
    """
    Evalúa el modelo en test y retorna métricas.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        logger.error(f"Error al importar TensorFlow: {e}")
        return {}
    
    # Predicción
    y_pred = model.predict(x_test, verbose=0).reshape(-1)

    # Métricas numpy (compatibles con tu pipeline)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    # MAPE (cuidado con ceros)
    eps = 1e-8
    mape = float(np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + eps))) * 100.0)

    out = {
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae": mae,
        f"{prefix}_mape": mape,
        f"{prefix}_predictions": y_pred,
    }

    print(out)
    return out


def plot_optuna_study(study, output_dir: str = "reports/figures/optuna"):
    """
    Genera gráficos estáticos de análisis del estudio de Optuna.
    
    Args:
        study: Estudio de Optuna completado
        output_dir: Directorio donde guardar las figuras
    """
    try:
        import optuna
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_timeline,
            plot_pareto_front
        )
    except ImportError as e:
        logger.error(f"Error al importar Optuna: {e}")
        logger.error("Instala Optuna: pip install optuna")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Optimization History - Evolución del mejor valor objetivo
    logger.info("Generando Optimization History Plot...")
    try:
        fig = plot_optimization_history(study)
        fig.figure.savefig(f"{output_dir}/optimization_history.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Optimization History guardado en: {output_dir}/optimization_history.png")
    except Exception as e:
        logger.warning(f"No se pudo generar Optimization History: {e}")
    
    # 2. Timeline Plot - Visualiza trials con tiempo de ejecución
    logger.info("Generando Timeline Plot...")
    try:
        fig = plot_timeline(study)
        fig.figure.savefig(f"{output_dir}/timeline.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Timeline Plot guardado en: {output_dir}/timeline.png")
    except Exception as e:
        logger.warning(f"No se pudo generar Timeline Plot: {e}")
    
    # 3. Parameter Importances - Importancia de hiperparámetros
    logger.info("Generando Parameter Importances Plot...")
    try:
        fig = plot_param_importances(study)
        fig.figure.savefig(f"{output_dir}/param_importances.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Parameter Importances guardado en: {output_dir}/param_importances.png")
    except Exception as e:
        logger.warning(f"No se pudo generar Parameter Importances: {e}")
    
    # 4. Pareto Front - Solo si es multi-objetivo (si no lo es, se omite)
    logger.info("Generando Pareto Front Plot...")
    try:
        # Verificar si el estudio es multi-objetivo
        if len(study.directions) > 1:
            fig = plot_pareto_front(study)
            fig.figure.savefig(f"{output_dir}/pareto_front.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Pareto Front guardado en: {output_dir}/pareto_front.png")
        else:
            logger.info("Pareto Front no aplicable (estudio de objetivo único)")
    except Exception as e:
        logger.warning(f"No se pudo generar Pareto Front: {e}")
    
    # 5. Resumen estadístico manual con matplotlib
    logger.info("Generando resumen estadístico...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Resumen del Estudio Optuna', fontsize=16, fontweight='bold')
        
        # Subplot 1: Distribución de valores objetivo
        trials_values = [t.value for t in study.trials if t.value is not None]
        axes[0, 0].hist(trials_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(study.best_value, color='red', linestyle='--', linewidth=2, label=f'Best: {study.best_value:.4f}')
        axes[0, 0].set_title('Distribución de Valores Objetivo')
        axes[0, 0].set_xlabel('Valor Objetivo (Loss)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Subplot 2: Progreso de trials
        trial_numbers = [t.number for t in study.trials if t.value is not None]
        trial_values = [t.value for t in study.trials if t.value is not None]
        axes[0, 1].plot(trial_numbers, trial_values, 'o-', alpha=0.6, label='Trials')
        axes[0, 1].axhline(study.best_value, color='red', linestyle='--', linewidth=2, label=f'Best: {study.best_value:.4f}')
        axes[0, 1].set_title('Evolución de Trials')
        axes[0, 1].set_xlabel('Trial Number')
        axes[0, 1].set_ylabel('Valor Objetivo')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Subplot 3: Estado de trials
        trial_states = {}
        for t in study.trials:
            state = t.state.name
            trial_states[state] = trial_states.get(state, 0) + 1
        
        axes[1, 0].bar(trial_states.keys(), trial_states.values(), alpha=0.7, color=['green', 'red', 'orange', 'gray'])
        axes[1, 0].set_title('Estado de Trials')
        axes[1, 0].set_xlabel('Estado')
        axes[1, 0].set_ylabel('Cantidad')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Subplot 4: Información del mejor trial
        axes[1, 1].axis('off')
        best_info = f"""
        MEJOR TRIAL
        {'='*40}
        Trial Number: {study.best_trial.number}
        Valor Objetivo: {study.best_value:.6f}
        
        Mejores Hiperparámetros:
        {'-'*40}
        """
        for param, value in study.best_params.items():
            best_info += f"\n{param}: {value}"
        
        axes[1, 1].text(0.1, 0.5, best_info, fontsize=10, verticalalignment='center', 
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/study_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Resumen del estudio guardado en: {output_dir}/study_summary.png")
    except Exception as e:
        logger.warning(f"No se pudo generar resumen estadístico: {e}")
    
    logger.info(f"Todas las gráficas de Optuna guardadas en: {output_dir}/")


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                                output_path: str = "reports/figures/predicciones/lstm_pred_vs_actual.png",
                                title: str = "Gráfico de Predicción vs Valores Esperados"):
    """
    Genera un gráfico de dispersión de predicciones vs valores reales.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        output_path: Ruta donde guardar la figura
        title: Título del gráfico
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=30, color='blue', label='Predicciones')
    
    # Línea ideal (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Ideal')
    
    # Configuración
    plt.xlabel('Valores Esperados', fontsize=12)
    plt.ylabel('Valores Predichos', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico de predicción vs valores esperados guardado en: {output_path}")

def main(stage: str):
   
   logger.warning("lectura de archivo.")
   ruta_entrada  = "data/features/features.csv"
   ruta_datos_lstm = "data/features/clean.csv"
   df = leer_archivo_csv(ruta_entrada)

   df["time"] = pd.to_datetime(df["time"], errors="coerce")
   df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
   
   # Stage para solo separar datos sin entrenar
   if stage == "separar_datos":
       logger.info("Separando datos de junio para predicción...")
       df_train = separar_datos_prediccion(df, mes_prediccion=6, ruta_salida="data/predict_data/predict.xlsx")
       df_train_lstm = separar_datos_prediccion_lstm(df, mes_prediccion=6, ruta_salida=ruta_datos_lstm)
       logger.info("Separación completada. No se realizará entrenamiento.")
       
       logger.info("Generando columnas de estacionalidad.")
       df_train = columnas_estacionalidad(df_train)
       df_train_lstm = columnas_estacionalidad(df_train_lstm)
   
       logger.info("Dividiendo los datos en conjuntos de entrenamiento y prueba.")
       x_train, x_test, y_train, y_test = dividir_train_test(df_train, objective_col="temperature_2m_target")
       x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = dividir_train_test_lstm(df_train_lstm, objective_col="temperature_2m_target")

       # Guardar df_train_lstm como joblib para uso posterior
       import joblib
       from pathlib import Path
       Path("models").mkdir(parents=True, exist_ok=True)
       joblib.dump(df_train_lstm, "models/df_train_lstm.joblib")
       logger.info("df_train_lstm guardado en: models/df_train_lstm.joblib")

       return x_train, x_test, y_train, y_test, x_train2, y_train2, x_val2, y_val2, x_test2, y_test2, df_train_lstm
   
   
   
   if stage == "lineales":
       logger.info("Entrenando modelos lineales.")
       resultados, best_models = modelos_lineales(x_train, x_test, y_train, y_test)
       logger.info("Comparando modelos.")
       resultados_ordenados = comparar_modelos(resultados)
       print(resultados_ordenados)
       
       # Obtener el mejor modelo y generar curva de aprendizaje
       mejor_modelo_nombre = resultados_ordenados.iloc[0]["model"]
       mejor_modelo = best_models[mejor_modelo_nombre]
       logger.info(f"Mejor modelo seleccionado: {mejor_modelo_nombre}")
       ruta_curvas = f"reports/figures/curvas aprendizaje/curva_aprendizaje_{mejor_modelo_nombre.lower()}.png"

       plot_curvas_aprendizaje(modelo=mejor_modelo, modelo_nombre=mejor_modelo_nombre, x_train=x_train,
                               y_train=y_train, ruta_salida=ruta_curvas
       )
       
       # Guardar mejor modelo lineal
       ruta_modelo = f"models/{mejor_modelo_nombre.lower()}_final.joblib"
       guardar_modelo(mejor_modelo, ruta_modelo)
       
       # Crear y guardar metadatos del modelo lineal
       mejor_resultado = resultados_ordenados.iloc[0]
       metadatos = {
           "nombre_modelo": mejor_modelo_nombre,
           "fecha_entrenamiento": datetime.now().isoformat(),
           "version": "1.0.0",
           "target": "temperature_2m_target",
           "metricas": {
               "test_rmse": float(mejor_resultado["test_RMSE"]),
               "test_mae": float(mejor_resultado["test_MAE"]),
               "cv_rmse": float(mejor_resultado["cv_RMSE"]),
               "cv_mae": float(mejor_resultado["cv_MAE"])
           },
           "hiperparametros": mejor_resultado["best_params"],
           "features": list(x_train.columns),
           "n_features": len(x_train.columns),
           "n_muestras_train": len(x_train),
           "n_muestras_test": len(x_test),
           "ruta_modelo": ruta_modelo,
           "ruta_curva_aprendizaje": ruta_curvas
       }
       guardar_metadatos(metadatos, f"models/metadata/{mejor_modelo_nombre.lower()}_metadatos.json")
   
   elif stage == "xgboost": 
       resultados, best_models = entrenar_xgboost(x_train, x_test, y_train, y_test)
       best_params = resultados.iloc[0]["best_params"]
       curva = xgb_learning_curve_purista(x_train, y_train, best_params, n_splits=5, n_estimators=2000, 
                                          output_path="reports/figures/xgb_n_estimators_curve.png",
                                          x_test=x_test, y_test=y_test)
       best_round = curva["best_round"]
       logger.info(f"Mejor número de estimadores según curva purista: {best_round}")
       
       # Reentrenar modelo final con n_estimators óptimo
       logger.info(f"Reentrenando modelo final con n_estimators={best_round}")
       modelo_final = XGBRegressor(
           objective="reg:squarederror",
           eval_metric="rmse",
           n_estimators=best_round,
           tree_method="hist",
           random_state=42,
           n_jobs=-1,
           **best_params
       )
       modelo_final.fit(x_train, y_train)
       
       # Evaluar en conjunto de test
       y_pred = modelo_final.predict(x_test)
       test_rmse = root_mean_squared_error(y_test, y_pred)
       test_mae = mean_absolute_error(y_test, y_pred)
       logger.info(f"Modelo final - RMSE en test: {test_rmse:.4f}")
       logger.info(f"Modelo final - MAE en test: {test_mae:.4f}")

       df_importance = xgb_feature_importance_df(booster=modelo_final.get_booster(),feature_names=x_train.columns)
       print(df_importance.head(15))
       
       # Guardar modelo entrenado
       ruta_modelo = "models/xgboost_final.joblib"
       guardar_modelo(modelo_final, ruta_modelo)
       
       # Crear y guardar metadatos del modelo
       metadatos = {
           "nombre_modelo": "XGBoost Regressor",
           "fecha_entrenamiento": datetime.now().isoformat(),
           "version": "1.0.0",
           "target": "temperature_2m_target",
           "metricas": {
               "test_rmse": test_rmse,
               "test_mae": test_mae,
               "cv_rmse": float(resultados.iloc[0]["cv_RMSE"]),
               "cv_mae": float(resultados.iloc[0]["cv_MAE"])
           },
           "hiperparametros": best_params,
           "n_estimators_optimo": best_round,
           "features": list(x_train.columns),
           "n_features": len(x_train.columns),
           "n_muestras_train": len(x_train),
           "n_muestras_test": len(x_test),
           "ruta_modelo": ruta_modelo,
           "ruta_curva_aprendizaje": curva["output_path"]
       }
       guardar_metadatos(metadatos, "models/metadata/xgboost_metadatos.json")
       
       # Análisis SHAP
       logger.info("Iniciando análisis SHAP...")
       shap_results = analisis_shap(modelo_final, x_train, x_test, output_dir="reports/figures/shap")

   elif stage == "shap":
       # Cargar modelo guardado y ejecutar análisis SHAP
       logger.info("Cargando modelo XGBoost para análisis SHAP...")
       modelo_cargado = cargar_modelo("models/xgboost_final.joblib")
       
       # Ejecutar análisis SHAP paso a paso
       shap_results = analisis_shap(modelo_cargado, x_train, x_test, output_dir="reports/figures/shap")
   
   elif stage == "lstm":
       logger.info("Iniciando entrenamiento LSTM...")
       
       # Importar TensorFlow solo cuando sea necesario
       try:
           import tensorflow as tf
           from tensorflow.keras import layers, models, callbacks, optimizers
       except ImportError as e:
           logger.error(f"Error al importar TensorFlow: {e}")
           logger.error("Para usar LSTM necesitas instalar TensorFlow: pip install tensorflow")
           return
       
       # Verificar que df_train_lstm fue preparado en stage "separar_datos"
       import joblib
       try:
           logger.info("Cargando datos preparados para LSTM desde models/df_train_lstm.joblib ...")
           df_train_lstm = joblib.load("models/df_train_lstm.joblib")
           if df_train_lstm.empty or len(df_train_lstm) == 0:
               raise ValueError("El DataFrame df_train_lstm (joblib) está vacío")
           logger.info(f"Datos LSTM cargados desde joblib: {len(df_train_lstm)} registros")
       except (FileNotFoundError, ValueError) as e:
           logger.error(f"Error: {e}")
           logger.error("df_train_lstm está vacío o no fue encontrado en joblib.")
           logger.error("Debes ejecutar primero el stage 'separar_datos' para preparar los datos:")
           logger.error("  python src/train.py --stage separar_datos")
           return
       
       
       # División específica para LSTM (train/val/test)
       logger.info("Dividiendo los datos para LSTM (train/val/test).")
       x_train_df, y_train_df, x_val_df, y_val_df, x_test_df, y_test_df = dividir_train_test_lstm(df_train_lstm, objective_col="temperature_2m_target")
       
       # Crear ventanas deslizantes para LSTM
       logger.info("Creando ventanas deslizantes para LSTM...")
       features_lstm = [col for col in x_train_df.columns if col != "time"]
       
       # Reconstruir DataFrame completo para lstm_window
       df_full_train = pd.concat([x_train_df.reset_index(drop=True), y_train_df.reset_index(drop=True)], axis=1)
       df_full_val = pd.concat([x_val_df.reset_index(drop=True), y_val_df.reset_index(drop=True)], axis=1)
       df_full_test = pd.concat([x_test_df.reset_index(drop=True), y_test_df.reset_index(drop=True)], axis=1)
       
       # Crear ventanas para cada conjunto
       x_train_lstm, y_train_lstm, _ = lstm_window(df_full_train, features_lstm, "temperature_2m_target", window_size=48, horizon=3)
       x_val_lstm, y_val_lstm, _ = lstm_window(df_full_val, features_lstm, "temperature_2m_target", window_size=48, horizon=3)
       x_test_lstm, y_test_lstm, _ = lstm_window(df_full_test, features_lstm, "temperature_2m_target", window_size=48, horizon=3)
       
       logger.info(f"Formas de los datos LSTM:")
       logger.info(f"Train: x={x_train_lstm.shape}, y={y_train_lstm.shape}")
       logger.info(f"Val: x={x_val_lstm.shape}, y={y_val_lstm.shape}")
       logger.info(f"Test: x={x_test_lstm.shape}, y={y_test_lstm.shape}")
       
       # Escalado de datos
       logger.info("Escalando datos para LSTM...")
       x_train_scaled, x_val_scaled, x_test_scaled, scaler = scale_lstm(x_train_lstm, x_val_lstm, x_test_lstm)
       
       # Entrenamiento del modelo LSTM con Optuna
       logger.info("Entrenando modelo LSTM con optimización de hiperparámetros...")
       model_lstm, history, study = train_lstm(
           x_train_scaled, y_train_lstm,
           x_val_scaled, y_val_lstm,
           n_trials=30,  # Número de trials para Optuna
           epochs=500,
           patience=20,
           verbose=1,
           output_path="models/lstm_final.h5",
           optuna_db="sqlite:///models/optuna_lstm.db"
       )
       
       # Evaluación del modelo
       logger.info("Evaluando modelo LSTM en conjunto de test...")
       test_metrics = evaluate(model_lstm, x_test_scaled, y_test_lstm, prefix="test")
       
       # Generar gráficas de aprendizaje
       logger.info("Generando curvas de aprendizaje...")
       learning_curves_path = "reports/figures/curvas aprendizaje/lstm_sol/lstm_learning_curves.png"
       final_training_metrics = plot_lstm_learning_curves(history, learning_curves_path)
       
       # Generar gráficos de Optuna
       logger.info("Generando gráficos de análisis de Optuna...")
       plot_optuna_study(study, output_dir="reports/figures/optuna")
       
       # Generar gráfico de predicciones vs valores esperados
       logger.info("Generando gráfico de predicciones vs valores esperados...")
       y_pred_test = test_metrics['test_predictions']
       plot_predictions_vs_actual(
           y_test_lstm, 
           y_pred_test,
           output_path="reports/figures/predicciones/lstm_pred_vs_actual.png",
           title="LSTM: Predicción vs Valores Esperados"
       )
       
       logger.info("=" * 50)
       logger.info("RESULTADOS FINALES LSTM")
       logger.info("=" * 50)
       logger.info(f"RMSE en test: {test_metrics['test_rmse']:.4f}")
       logger.info(f"MAE en test: {test_metrics['test_mae']:.4f}")
       logger.info(f"Epochs entrenadas: {final_training_metrics['epochs_trained']}")
       logger.info(f"Mejor época: {final_training_metrics['best_epoch']} (val_loss: {final_training_metrics['best_val_loss']:.4f})")
       
       # Guardar metadatos del modelo LSTM
       metadatos_lstm = {
           "nombre_modelo": "LSTM",
           "fecha_entrenamiento": datetime.now().isoformat(),
           "version": "2.0.0",  # Actualizada a 2.0.0 con Optuna
           "target": "temperature_2m_target",
           "optimizacion": {
               "metodo": "Optuna",
               "n_trials": len(study.trials) if study else 0,
               "best_trial": study.best_trial.number if study else None,
               "best_value": float(study.best_value) if study else None
           },
           "arquitectura": {
               "window_size": 48,
               "horizon": 3,
               "best_params": study.best_params if study else {}
           },
           "metricas": {
               "test_rmse": float(test_metrics['test_rmse']),
               "test_mae": float(test_metrics['test_mae']),
               "final_train_rmse": float(final_training_metrics['final_train_rmse']),
               "final_val_rmse": float(final_training_metrics['final_val_rmse']),
               "final_train_mae": float(final_training_metrics['final_train_mae']),
               "final_val_mae": float(final_training_metrics['final_val_mae']),
               "best_val_loss": float(final_training_metrics['best_val_loss']),
               "best_epoch": int(final_training_metrics['best_epoch']),
               "epochs_trained": int(final_training_metrics['epochs_trained'])
           },
           "features": features_lstm,
           "n_features": len(features_lstm),
           "n_muestras_train": len(y_train_lstm),
           "n_muestras_val": len(y_val_lstm),
           "n_muestras_test": len(y_test_lstm),
           "ruta_modelo": "models/lstm_final.h5",
           "ruta_curvas_aprendizaje": learning_curves_path,
           "scaler_info": {
               "tipo": "StandardScaler",
               "parametros": {
                   "mean": scaler.mean_.tolist(),
                   "scale": scaler.scale_.tolist()
               }
           }
       }
       
       # Guardar metadatos
       guardar_metadatos(metadatos_lstm, "models/metadata/lstm_metadatos.json")
       
       # Guardar el scaler por separado
       joblib.dump(scaler, "models/lstm_scaler.joblib")
       logger.info("Scaler guardado en: models/lstm_scaler.joblib")
       
       logger.info("Entrenamiento LSTM completado exitosamente.")
   
   elif stage == "lstm_multiobj":
       logger.info("Iniciando entrenamiento LSTM MULTIOBJETIVO (MAE + RMSE)...")
       
       # Importar TensorFlow
       try:
           import tensorflow as tf
           from tensorflow.keras import layers, models, callbacks, optimizers
       except ImportError as e:
           logger.error(f"Error al importar TensorFlow: {e}")
           logger.error("Instala TensorFlow: pip install tensorflow")
           return
       
       # Cargar datos preparados
       import joblib
       try:
           logger.info("Cargando datos preparados para LSTM...")
           df_train_lstm = joblib.load("models/df_train_lstm.joblib")
           if df_train_lstm.empty:
               raise ValueError("df_train_lstm vacío")
           logger.info(f"Datos cargados: {len(df_train_lstm)} registros")
       except (FileNotFoundError, ValueError) as e:
           logger.error(f"Error: {e}")
           logger.error("Ejecuta primero: python src/train.py --stage separar_datos")
           return
       
       # División train/val/test
       logger.info("Dividiendo datos (train/val/test)...")
       x_train_df, y_train_df, x_val_df, y_val_df, x_test_df, y_test_df = dividir_train_test_lstm(
           df_train_lstm, objective_col="temperature_2m_target"
       )
       
       # Crear ventanas - Aumentar a 72h (3 días) para capturar más patrones
       logger.info("Creando ventanas deslizantes...")
       features_lstm = [col for col in x_train_df.columns if col != "time"]
       
       df_full_train = pd.concat([x_train_df.reset_index(drop=True), y_train_df.reset_index(drop=True)], axis=1)
       df_full_val = pd.concat([x_val_df.reset_index(drop=True), y_val_df.reset_index(drop=True)], axis=1)
       df_full_test = pd.concat([x_test_df.reset_index(drop=True), y_test_df.reset_index(drop=True)], axis=1)
       
       x_train_lstm, y_train_lstm, _ = lstm_window(df_full_train, features_lstm, "temperature_2m_target", window_size=72, horizon=3)
       x_val_lstm, y_val_lstm, _ = lstm_window(df_full_val, features_lstm, "temperature_2m_target", window_size=72, horizon=3)
       x_test_lstm, y_test_lstm, _ = lstm_window(df_full_test, features_lstm, "temperature_2m_target", window_size=72, horizon=3)
       
       logger.info(f"Train: x={x_train_lstm.shape}, y={y_train_lstm.shape}")
       logger.info(f"Val: x={x_val_lstm.shape}, y={y_val_lstm.shape}")
       logger.info(f"Test: x={x_test_lstm.shape}, y={y_test_lstm.shape}")
       
       # Escalado
       logger.info("Escalando datos...")
       x_train_scaled, x_val_scaled, x_test_scaled, scaler = scale_lstm(x_train_lstm, x_val_lstm, x_test_lstm)
       
       # Entrenamiento multiobjetivo
       logger.info("Entrenando con optimización multiobjetivo (MAE + RMSE)...")
       model_lstm, history, study, pareto_trials = train_lstm_multiobj(
           x_train_scaled, y_train_lstm,
           x_val_scaled, y_val_lstm,
           n_trials=100,  # Máxima exploración
           epochs=500,
           patience=40,
           verbose=1,
           output_path="models/lstm_multiobj_final.h5",
           optuna_db="sqlite:///models/optuna_lstm_multiobj.db",
           use_mae_loss=False  # Usar Huber loss (más robusto)
       )
       
       # Evaluación en test
       logger.info("Evaluando modelo en test...")
       test_metrics = evaluate(model_lstm, x_test_scaled, y_test_lstm, prefix="test")
       
       # Curvas de aprendizaje
       logger.info("Generando curvas de aprendizaje...")
       learning_curves_path = "reports/figures/curvas aprendizaje/lstm_multiobj/lstm_learning_curves.png"
       final_training_metrics = plot_lstm_learning_curves(history, learning_curves_path)
       
       # Gráficos Optuna
       logger.info("Generando gráficos de Optuna...")
       plot_optuna_study(study, output_dir="reports/figures/optuna_multiobj")
       
       # Predicciones vs actual
       logger.info("Generando gráfico de predicciones...")
       y_pred_test = test_metrics['test_predictions']
       plot_predictions_vs_actual(
           y_test_lstm,
           y_pred_test,
           output_path="reports/figures/predicciones/lstm_multiobj_pred_vs_actual.png",
           title="LSTM Multiobjetivo: Predicción vs Valores Esperados"
       )
       
       # Resultados finales
       logger.info("="*50)
       logger.info("RESULTADOS FINALES LSTM MULTIOBJETIVO")
       logger.info("="*50)
       logger.info(f"RMSE en test: {test_metrics['test_rmse']:.4f}")
       logger.info(f"MAE en test: {test_metrics['test_mae']:.4f}")
       logger.info(f"Soluciones Pareto: {len(pareto_trials)}")
       logger.info(f"Epochs entrenadas: {final_training_metrics['epochs_trained']}")
       
       # Guardar metadatos
       best_trial = min(pareto_trials, key=lambda t: t.values[0])  # Mejor por MAE
       metadatos_lstm = {
           "nombre_modelo": "LSTM Multiobjetivo",
           "fecha_entrenamiento": datetime.now().isoformat(),
           "version": "3.0.0",
           "target": "temperature_2m_target",
           "optimizacion": {
               "metodo": "Optuna Multiobjetivo",
               "objetivos": ["MAE", "RMSE"],
               "n_trials": len(study.trials),
               "n_pareto_solutions": len(pareto_trials),
               "best_trial": best_trial.number,
               "best_mae": float(best_trial.values[0]),
               "best_rmse": float(best_trial.values[1])
           },
           "arquitectura": {
               "window_size": 48,
               "horizon": 3,
               "loss_function": "mae",
               "best_params": best_trial.params
           },
           "metricas": {
               "test_rmse": float(test_metrics['test_rmse']),
               "test_mae": float(test_metrics['test_mae']),
               "final_train_rmse": float(final_training_metrics['final_train_rmse']),
               "final_val_rmse": float(final_training_metrics['final_val_rmse']),
               "final_train_mae": float(final_training_metrics['final_train_mae']),
               "final_val_mae": float(final_training_metrics['final_val_mae']),
               "best_val_loss": float(final_training_metrics['best_val_loss']),
               "best_epoch": int(final_training_metrics['best_epoch']),
               "epochs_trained": int(final_training_metrics['epochs_trained'])
           },
           "features": features_lstm,
           "n_features": len(features_lstm),
           "n_muestras_train": len(y_train_lstm),
           "n_muestras_val": len(y_val_lstm),
           "n_muestras_test": len(y_test_lstm),
           "ruta_modelo": "models/lstm_multiobj_final.h5",
           "ruta_curvas_aprendizaje": learning_curves_path,
           "scaler_info": {
               "tipo": "StandardScaler",
               "parametros": {
                   "mean": scaler.mean_.tolist(),
                   "scale": scaler.scale_.tolist()
               }
           }
       }
       
       guardar_metadatos(metadatos_lstm, "models/metadata/lstm_multiobj_metadatos.json")
       joblib.dump(scaler, "models/lstm_multiobj_scaler.joblib")
       logger.info("Scaler guardado en: models/lstm_multiobj_scaler.joblib")
       logger.info("Entrenamiento LSTM multiobjetivo completado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="lstm_multiobj",
                        choices=["separar_datos", "lineales", "xgboost", "shap", "lstm", "lstm_multiobj"])
    args = parser.parse_args()
    main(args.stage)