import pandas as pd
import numpy as np
import joblib
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from prophet import Prophet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cargar_modelo(ruta_modelo: str):
    """Carga el modelo entrenado desde archivo joblib."""
    modelo = joblib.load(ruta_modelo)
    logger.info(f"Modelo cargado desde: {ruta_modelo}")
    return modelo


def cargar_metadatos(ruta_metadatos: str) -> dict:
    """Carga los metadatos del modelo."""
    with open(ruta_metadatos, "r", encoding="utf-8") as f:
        metadatos = json.load(f)
    logger.info(f"Metadatos cargados desde: {ruta_metadatos}")
    return metadatos


def leer_datos_prediccion(ruta_archivo: str) -> pd.DataFrame:
    """Lee los datos de predicción desde Excel."""
    df = pd.read_excel(ruta_archivo, engine='openpyxl')
    logger.info(f"Datos de predicción cargados: {len(df)} registros desde {ruta_archivo}")
    return df


def columnas_estacionalidad(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Prepara las features necesarias para el modelo."""
    df_copia = df.copy()
    
    # Convertir time a datetime y crear columnas de estacionalidad
    if "time" in df_copia.columns:
        df_copia["time"] = pd.to_datetime(df_copia["time"], errors="coerce")
        df_copia["dayofweek"] = df_copia["time"].dt.dayofweek
        df_copia["month"] = df_copia["time"].dt.month
        df_copia["hour"] = df_copia["time"].dt.hour
        df_copia["sin_comp"] = np.sin(2 * np.pi * df_copia["hour"] / 24)
        df_copia["cos_comp"] = np.cos(2 * np.pi * df_copia["hour"] / 24)
    
    # Verificar que todas las features están presentes
    features_faltantes = [f for f in features if f not in df_copia.columns]
    if features_faltantes:
        logger.warning(f"Features faltantes en los datos: {features_faltantes}")
    
    # Seleccionar solo las features necesarias
    features_disponibles = [f for f in features if f in df_copia.columns]
    X = df_copia[features_disponibles]
    
    logger.info(f"Features preparadas: {len(features_disponibles)} columnas")
    return X, df_copia


def realizar_predicciones(modelo, X: pd.DataFrame) -> np.ndarray:
    """Realiza predicciones con el modelo."""
    predicciones = modelo.predict(X)
    logger.info(f"Predicciones realizadas: {len(predicciones)} valores")
    return predicciones


def evaluar_predicciones(y_real: pd.Series, y_pred: np.ndarray) -> dict:
    """Evalúa las predicciones contra valores reales si están disponibles."""
    metricas = {
        "rmse": root_mean_squared_error(y_real, y_pred),
        "mae": mean_absolute_error(y_real, y_pred),
        "n_predicciones": len(y_pred)
    }
    logger.info(f"Métricas de predicción - RMSE: {metricas['rmse']:.4f}, MAE: {metricas['mae']:.4f}")
    return metricas


def guardar_predicciones(df: pd.DataFrame, predicciones: np.ndarray, 
                         ruta_salida: str, target_col: str = "temperature_2m_target") -> None:
    """Guarda las predicciones en un archivo Excel."""
    df_resultado = df.copy()
    df_resultado["prediccion"] = predicciones
    
    # Calcular error si hay valores reales
    if target_col in df_resultado.columns:
        df_resultado["error"] = df_resultado[target_col] - df_resultado["prediccion"]
        df_resultado["error_absoluto"] = np.abs(df_resultado["error"])
    
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_excel(ruta_salida, index=False, engine='openpyxl')
    logger.info(f"Predicciones guardadas en: {ruta_salida}")


def graficar_predicciones(df: pd.DataFrame, predicciones: np.ndarray, 
                          target_col: str, output_dir: str = "reports/figures") -> None:
    """Genera gráficas de predicciones vs valores reales."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    time_col = df["time"] if "time" in df.columns else df.index
    y_real = df[target_col]
    
    # 1. Serie temporal: Real vs Predicción
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(time_col, y_real, label="Real", alpha=0.8, linewidth=1)
    ax1.plot(time_col, predicciones, label="Predicción", alpha=0.8, linewidth=1)
    ax1.set_title("Temperatura Real vs Predicción (Junio 2024)")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Temperatura (°C)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    fig1.savefig(output_dir / "prediccion_serie_temporal.png", dpi=200)
    logger.info(f"Gráfica de serie temporal guardada en {output_dir / 'prediccion_serie_temporal.png'}")
    plt.close(fig1)
    
    # 2. Scatter plot: Real vs Predicción
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(y_real, predicciones, alpha=0.5, s=10)
    min_val = min(y_real.min(), predicciones.min())
    max_val = max(y_real.max(), predicciones.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label="Línea ideal")
    ax2.set_title("Real vs Predicción")
    ax2.set_xlabel("Temperatura Real (°C)")
    ax2.set_ylabel("Temperatura Predicha (°C)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig2.savefig(output_dir / "prediccion_scatter.png", dpi=200)
    logger.info(f"Scatter plot guardado en {output_dir / 'prediccion_scatter.png'}")
    plt.close(fig2)
    
    # 3. Distribución de errores
    errores = y_real - predicciones
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.hist(errores, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_title(f"Distribución de Errores (Media: {errores.mean():.3f}, Std: {errores.std():.3f})")
    ax3.set_xlabel("Error (Real - Predicción) °C")
    ax3.set_ylabel("Frecuencia")
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    fig3.savefig(output_dir / "prediccion_errores_hist.png", dpi=200)
    logger.info(f"Histograma de errores guardado en {output_dir / 'prediccion_errores_hist.png'}")
    plt.close(fig3)
    
    # 4. Error a lo largo del tiempo
    fig4, ax4 = plt.subplots(figsize=(14, 5))
    ax4.plot(time_col, errores, alpha=0.7, linewidth=0.8)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1)
    ax4.fill_between(time_col, errores, 0, alpha=0.3)
    ax4.set_title("Error de Predicción en el Tiempo")
    ax4.set_xlabel("Fecha")
    ax4.set_ylabel("Error (°C)")
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    fig4.savefig(output_dir / "prediccion_errores_tiempo.png", dpi=200)
    logger.info(f"Gráfica de errores en tiempo guardada en {output_dir / 'prediccion_errores_tiempo.png'}")
    plt.close(fig4)


def entrenar_prophet(df_train: pd.DataFrame, target_col: str) -> Prophet:
    """Entrena un modelo Prophet con los datos de entrenamiento."""
    # Prophet requiere columnas 'ds' y 'y'
    df_prophet = df_train[["time", target_col]].copy()
    df_prophet.columns = ["ds", "y"]
    df_prophet = df_prophet.dropna()
    
    logger.info(f"Entrenando Prophet con {len(df_prophet)} registros...")
    
    # Configurar Prophet para datos horarios
    modelo_prophet = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,  # Solo tenemos 5 meses de datos
        changepoint_prior_scale=0.05
    )
    
    # Suprimir logs de Prophet
    import logging as log
    log.getLogger('prophet').setLevel(log.WARNING)
    log.getLogger('cmdstanpy').setLevel(log.WARNING)
    
    modelo_prophet.fit(df_prophet)
    logger.info("Modelo Prophet entrenado.")
    
    return modelo_prophet


def predecir_prophet(modelo_prophet: Prophet, df_predict: pd.DataFrame) -> np.ndarray:
    """Realiza predicciones con Prophet."""
    df_future = df_predict[["time"]].copy()
    df_future.columns = ["ds"]
    
    forecast = modelo_prophet.predict(df_future)
    predicciones = forecast["yhat"].values
    
    logger.info(f"Prophet: {len(predicciones)} predicciones realizadas.")
    return predicciones


def comparar_modelos(df: pd.DataFrame, pred_xgboost: np.ndarray, pred_prophet: np.ndarray,
                     target_col: str, output_dir: str = "reports/figures") -> dict:
    """Compara predicciones de XGBoost vs Prophet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_real = df[target_col].values
    time_col = df["time"]
    
    # Calcular métricas
    metricas = {
        "XGBoost": {
            "RMSE": root_mean_squared_error(y_real, pred_xgboost),
            "MAE": mean_absolute_error(y_real, pred_xgboost)
        },
        "Prophet": {
            "RMSE": root_mean_squared_error(y_real, pred_prophet),
            "MAE": mean_absolute_error(y_real, pred_prophet)
        }
    }
    
    logger.info("=" * 50)
    logger.info("COMPARACIÓN DE MODELOS")
    logger.info("=" * 50)
    logger.info(f"XGBoost  - RMSE: {metricas['XGBoost']['RMSE']:.4f}, MAE: {metricas['XGBoost']['MAE']:.4f}")
    logger.info(f"Prophet  - RMSE: {metricas['Prophet']['RMSE']:.4f}, MAE: {metricas['Prophet']['MAE']:.4f}")
    
    # Gráfica comparativa
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_col, y_real, label="Real", alpha=0.8, linewidth=1, color='black')
    ax.plot(time_col, pred_xgboost, label=f"XGBoost (RMSE: {metricas['XGBoost']['RMSE']:.2f})", alpha=0.7, linewidth=1)
    ax.plot(time_col, pred_prophet, label=f"Prophet (RMSE: {metricas['Prophet']['RMSE']:.2f})", alpha=0.7, linewidth=1)
    ax.set_title("Comparación: Real vs XGBoost vs Prophet")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Temperatura (°C)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "comparacion_xgboost_prophet.png", dpi=200)
    logger.info(f"Gráfica comparativa guardada en {output_dir / 'comparacion_xgboost_prophet.png'}")
    plt.close(fig)
    
    # Scatter plot comparativo
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(y_real, pred_xgboost, alpha=0.5, s=10)
    axes[0].plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
    axes[0].set_title(f"XGBoost (RMSE: {metricas['XGBoost']['RMSE']:.2f})")
    axes[0].set_xlabel("Real")
    axes[0].set_ylabel("Predicción")
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(y_real, pred_prophet, alpha=0.5, s=10, color='orange')
    axes[1].plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
    axes[1].set_title(f"Prophet (RMSE: {metricas['Prophet']['RMSE']:.2f})")
    axes[1].set_xlabel("Real")
    axes[1].set_ylabel("Predicción")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(output_dir / "comparacion_scatter.png", dpi=200)
    logger.info(f"Scatter comparativo guardado en {output_dir / 'comparacion_scatter.png'}")
    plt.close(fig2)
    
    # Calcular Skill Scores
    skill_rmse = 1 - (metricas["XGBoost"]["RMSE"] / metricas["Prophet"]["RMSE"])
    skill_mae = 1 - (metricas["XGBoost"]["MAE"] / metricas["Prophet"]["MAE"])
    
    # Crear DataFrame con métricas y skill scores
    df_metricas = pd.DataFrame([{
        "RMSE_xgboost": metricas["XGBoost"]["RMSE"],
        "MAE_xgboost": metricas["XGBoost"]["MAE"],
        "RMSE_prophet": metricas["Prophet"]["RMSE"],
        "MAE_prophet": metricas["Prophet"]["MAE"],
        "Skill_RMSE_vs_prophet": skill_rmse,
        "Skill_MAE_vs_prophet": skill_mae
    }])
    
    # Guardar métricas
    ruta_metricas = output_dir / "metricas_comparacion.csv"
    df_metricas.to_csv(ruta_metricas, index=False)
    logger.info(f"Métricas guardadas en {ruta_metricas}")
    
    # Mostrar métricas
    logger.info("-" * 50)
    logger.info("SKILL SCORES (XGBoost vs Prophet)")
    logger.info("-" * 50)
    logger.info(f"Skill RMSE: {skill_rmse:.4f} ({skill_rmse*100:.2f}% mejor)")
    logger.info(f"Skill MAE:  {skill_mae:.4f} ({skill_mae*100:.2f}% mejor)")
    
    print("\nDataFrame de métricas:")
    print(df_metricas.to_string(index=False))
    
    return metricas, df_metricas


def main():
    # Rutas de archivos
    ruta_modelo = "models/xgboost_final.joblib"
    ruta_metadatos = "models/metadata/xgboost_metadatos.json"
    ruta_datos_prediccion = "data/predict_data/predict.xlsx"
    ruta_salida = "data/predict_data/predicciones_junio.xlsx"
    
    # Cargar modelo y metadatos
    logger.info("=" * 60)
    logger.info("INICIANDO PROCESO DE PREDICCIÓN")
    logger.info("=" * 60)
    
    modelo = cargar_modelo(ruta_modelo)
    metadatos = cargar_metadatos(ruta_metadatos)
    features = metadatos["features"]
    target_col = metadatos["target"]
    
    logger.info(f"Modelo: {metadatos['nombre_modelo']}")
    logger.info(f"Target: {target_col}")
    logger.info(f"Features esperadas: {len(features)}")
    
    # Cargar y preparar datos
    df = leer_datos_prediccion(ruta_datos_prediccion)
    X, df_preparado = columnas_estacionalidad(df, features)
    
    # Realizar predicciones
    predicciones = realizar_predicciones(modelo, X)
    
    # Evaluar si hay valores reales disponibles
    if target_col in df_preparado.columns:
        y_real = df_preparado[target_col].dropna()
        if len(y_real) > 0:
            # Alinear predicciones con valores reales no nulos
            mask = df_preparado[target_col].notna()
            metricas = evaluar_predicciones(y_real, predicciones[mask])
            
            logger.info("-" * 40)
            logger.info("MÉTRICAS DE PREDICCIÓN (JUNIO)")
            logger.info("-" * 40)
            logger.info(f"RMSE: {metricas['rmse']:.4f}")
            logger.info(f"MAE: {metricas['mae']:.4f}")
            logger.info(f"N predicciones: {metricas['n_predicciones']}")
    
    # Guardar predicciones XGBoost
    guardar_predicciones(df_preparado, predicciones, ruta_salida, target_col)
    
    # Graficar predicciones XGBoost
    if target_col in df_preparado.columns:
        graficar_predicciones(df_preparado, predicciones, target_col, output_dir="reports/figures/predicciones")
    
    # === PROPHET ===
    logger.info("\n" + "=" * 60)
    logger.info("ENTRENANDO MODELO PROPHET PARA COMPARACIÓN")
    logger.info("=" * 60)
    
    # Cargar datos de entrenamiento (enero-mayo) para entrenar Prophet
    ruta_features = "data/features/features.csv"
    df_features = pd.read_csv(ruta_features)
    df_features["time"] = pd.to_datetime(df_features["time"])
    df_train_prophet = df_features[df_features["time"].dt.month != 6].copy()  # Excluir junio
    
    # Entrenar Prophet
    modelo_prophet = entrenar_prophet(df_train_prophet, target_col)
    
    # Predecir con Prophet en datos de junio
    predicciones_prophet = predecir_prophet(modelo_prophet, df_preparado)
    
    # Comparar modelos
    if target_col in df_preparado.columns:
        mask = df_preparado[target_col].notna()
        metricas_comparacion = comparar_modelos(
            df_preparado[mask], 
            predicciones[mask], 
            predicciones_prophet[mask],
            target_col,
            output_dir="reports/figures/predicciones"
        )
    
    # Guardar predicciones de ambos modelos
    df_comparacion = df_preparado.copy()
    df_comparacion["pred_xgboost"] = predicciones
    df_comparacion["pred_prophet"] = predicciones_prophet
    if target_col in df_comparacion.columns:
        df_comparacion["error_xgboost"] = df_comparacion[target_col] - df_comparacion["pred_xgboost"]
        df_comparacion["error_prophet"] = df_comparacion[target_col] - df_comparacion["pred_prophet"]
    df_comparacion.to_excel("data/predict_data/comparacion_modelos.xlsx", index=False, engine='openpyxl')
    logger.info("Comparación guardada en data/predict_data/comparacion_modelos.xlsx")
    
    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("PREDICCIÓN Y COMPARACIÓN COMPLETADAS")
    logger.info("=" * 60)
    logger.info(f"Archivo XGBoost: {ruta_salida}")
    logger.info(f"Archivo comparación: data/predict_data/comparacion_modelos.xlsx")
    
    # Mostrar primeras predicciones
    print("\nPrimeras 10 predicciones (XGBoost vs Prophet):")
    print(df_preparado[["time", target_col]].head(10).assign(
        pred_xgboost=predicciones[:10],
        pred_prophet=predicciones_prophet[:10]
    ))


if __name__ == "__main__":
    main()
