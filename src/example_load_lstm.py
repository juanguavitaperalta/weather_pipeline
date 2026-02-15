"""
Ejemplo de cómo cargar y usar el modelo LSTM entrenado con Optuna.

Este script demuestra:
1. Carga del modelo LSTM guardado
2. Carga del scaler
3. Carga de los mejores hiperparámetros
4. Realización de predicciones

Uso:
    python src/example_load_lstm.py
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_lstm_model(model_path: str = "models/lstm_final.h5"):
    """Carga el modelo LSTM guardado."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        logger.info(f"✓ Modelo cargado desde {model_path}")
        return model
    except ImportError:
        logger.error("TensorFlow no está instalado. Instala con: pip install tensorflow")
        return None
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        return None


def load_scaler(scaler_path: str = "models/lstm_scaler.joblib"):
    """Carga el scaler usado para normalizar los datos."""
    try:
        import joblib
        scaler = joblib.load(scaler_path)
        logger.info(f"✓ Scaler cargado desde {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Error al cargar el scaler: {e}")
        return None


def load_best_params(params_path: str = "models/lstm_final_best_params.json"):
    """Carga los mejores hiperparámetros encontrados por Optuna."""
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        logger.info(f"✓ Parámetros cargados desde {params_path}")
        return params
    except Exception as e:
        logger.error(f"Error al cargar los parámetros: {e}")
        return None


def load_metadata(metadata_path: str = "models/metadata/lstm_metadatos.json"):
    """Carga los metadatos del modelo."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"✓ Metadatos cargados desde {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"Error al cargar los metadatos: {e}")
        return None


def predict_with_lstm(model, scaler, x_data: np.ndarray):
    """
    Realiza predicciones con el modelo LSTM.
    
    Args:
        model: Modelo LSTM cargado
        scaler: Scaler para normalizar datos
        x_data: Datos de entrada con forma (n_samples, window_size, n_features)
    
    Returns:
        Predicciones desnormalizadas
    """
    # Normalizar datos
    n_samples, window_size, n_features = x_data.shape
    x_reshaped = x_data.reshape(-1, n_features)
    x_scaled = scaler.transform(x_reshaped).reshape(n_samples, window_size, n_features)
    
    # Predecir
    predictions = model.predict(x_scaled, verbose=0)
    
    return predictions.flatten()


def main():
    """Función principal de ejemplo."""
    logger.info("="*70)
    logger.info("EJEMPLO: CARGAR Y USAR MODELO LSTM")
    logger.info("="*70)
    
    # 1. Cargar modelo
    logger.info("\n📦 Cargando modelo...")
    model = load_lstm_model()
    
    if model is None:
        logger.error("No se pudo cargar el modelo. Asegúrate de haberlo entrenado primero.")
        return
    
    # 2. Cargar scaler
    logger.info("\n📦 Cargando scaler...")
    scaler = load_scaler()
    
    if scaler is None:
        logger.error("No se pudo cargar el scaler.")
        return
    
    # 3. Cargar mejores parámetros
    logger.info("\n📦 Cargando mejores hiperparámetros...")
    best_params = load_best_params()
    
    if best_params:
        logger.info("\n⚙️  Mejores hiperparámetros encontrados por Optuna:")
        for key, value in best_params.get('best_params', {}).items():
            logger.info(f"  {key}: {value}")
        logger.info(f"\nMejor val_loss: {best_params.get('best_value', 'N/A'):.6f}")
        logger.info(f"Trial ganador: #{best_params.get('best_trial', 'N/A')}")
    
    # 4. Cargar metadatos
    logger.info("\n📦 Cargando metadatos del modelo...")
    metadata = load_metadata()
    
    if metadata:
        logger.info("\n📊 Información del modelo:")
        logger.info(f"  Versión: {metadata.get('version', 'N/A')}")
        logger.info(f"  Fecha entrenamiento: {metadata.get('fecha_entrenamiento', 'N/A')}")
        logger.info(f"  Número de features: {metadata.get('n_features', 'N/A')}")
        
        metricas = metadata.get('metricas', {})
        logger.info("\n📈 Métricas en test:")
        logger.info(f"  RMSE: {metricas.get('test_rmse', 'N/A'):.4f}")
        logger.info(f"  MAE: {metricas.get('test_mae', 'N/A'):.4f}")
        
        arquitectura = metadata.get('arquitectura', {})
        logger.info("\n🏗️  Arquitectura:")
        logger.info(f"  Window size: {arquitectura.get('window_size', 'N/A')}")
        logger.info(f"  Horizon: {arquitectura.get('horizon', 'N/A')}")
    
    # 5. Resumen del modelo
    logger.info("\n🧠 Resumen del modelo:")
    model.summary()
    
    # 6. Ejemplo de predicción con datos sintéticos
    logger.info("\n🔮 Ejemplo de predicción con datos sintéticos...")
    
    # Crear datos sintéticos para demostración
    # Forma: (n_samples, window_size, n_features)
    n_samples = 5
    window_size = metadata.get('arquitectura', {}).get('window_size', 48)
    n_features = metadata.get('n_features', 15)
    
    # Generar datos aleatorios normalizados (simula datos reales)
    x_synthetic = np.random.randn(n_samples, window_size, n_features).astype(np.float32)
    
    logger.info(f"  Forma de entrada: {x_synthetic.shape}")
    logger.info(f"  (n_samples={n_samples}, window_size={window_size}, n_features={n_features})")
    
    # Realizar predicciones
    predictions = predict_with_lstm(model, scaler, x_synthetic)
    
    logger.info("\n  Predicciones:")
    for i, pred in enumerate(predictions, 1):
        logger.info(f"    Muestra {i}: {pred:.2f}°C")
    
    logger.info("\n" + "="*70)
    logger.info("EJEMPLO COMPLETADO")
    logger.info("="*70)
    logger.info("\n💡 Para usar el modelo con tus datos:")
    logger.info("  1. Prepara tus datos con la misma forma: (n_samples, 48, 15)")
    logger.info("  2. Asegúrate de que las features estén en el mismo orden que durante el entrenamiento")
    logger.info("  3. Usa predict_with_lstm(model, scaler, x_data) para predecir")
    logger.info("\n💡 Las features requeridas están en metadata['features']")


if __name__ == "__main__":
    main()
