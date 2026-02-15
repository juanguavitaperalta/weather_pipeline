# Guía de Uso: LSTM con Optimización Optuna

## Descripción General

Esta guía describe cómo usar el modelo LSTM refactorizado con optimización automática de hiperparámetros mediante Optuna.

## Arquitectura

### 1. Función `build_model(hparams, input_shape)`

Construye y compila un modelo LSTM con hiperparámetros configurables.

**Parámetros de entrada:**
```python
hparams = {
    'filters': int,               # Filtros Conv1D (ej: 16, 32, 48, 64)
    'kernel_size': int,           # Tamaño kernel Conv1D (ej: 2, 3, 4, 5)
    'lstm1_units': int,           # Unidades LSTM capa 1 (ej: 32, 64, 96, 128)
    'lstm2_units': int,           # Unidades LSTM capa 2 (ej: 16, 32, 48, 64)
    'dropout': float,             # Dropout (ej: 0.1, 0.2, 0.3, 0.4)
    'recurrent_dropout': float,   # Dropout recurrente (ej: 0.0, 0.1, 0.2)
    'learning_rate': float,       # Tasa de aprendizaje (ej: 1e-4, 1e-3, 1e-2)
    'batch_size': int,            # Tamaño del batch (ej: 16, 32, 64)
    'use_causal_padding': bool,   # Padding causal (True/False)
    'optimizer': str              # Optimizador ('adam', 'adamw', 'rmsprop', 'sgd')
}

input_shape = (window_size, n_features)  # ej: (48, 15)
```

**Retorna:**
- `modelo`: Modelo compilado listo para entrenar
- `batch_size`: Tamaño de batch para usar en el entrenamiento

**Estructura del modelo:**
```
Input(window_size, n_features)
    ↓
Conv1D(filters, kernel_size, padding='same'/'causal')
    ↓
LSTM #1(lstm1_units, return_sequences=True, dropout, recurrent_dropout)
    ↓
LSTM #2(lstm2_units, return_sequences=False, dropout)
    ↓
Dense(1, activation='linear')  # Output layer
```

### 2. Función `train_lstm(x_train, y_train, x_val, y_val, ...)`

Entrena un modelo LSTM usando Optuna para optimización bayesiana de hiperparámetros.

**Parámetros:**
```python
train_lstm(
    x_train=x_train_scaled,      # Array (samples, window_size, features)
    y_train=y_train_lstm,        # Array (samples,)
    x_val=x_val_scaled,          # Array (val_samples, window_size, features)
    y_val=y_val_lstm,            # Array (val_samples,)
    n_trials=30,                 # Número de combinaciones a probar
    epochs=500,                  # Épocas máximas por trial
    patience=20,                 # Paciencia para early stopping
    verbose=1,                   # Nivel de verbosidad (0, 1, 2)
    output_path="models/lstm_final.h5",           # Ruta del modelo
    optuna_db="sqlite:///models/optuna_lstm.db"   # Base de datos Optuna
)
```

**Retorna:**
- `mejor_modelo`: Modelo con mejor val_loss encontrado
- `mejor_history`: Historia de entrenamiento del mejor modelo
- `study`: Objeto Study de Optuna con todos los resultados

## Proceso de Optimización

### Función Objetivo

Para cada trial, Optuna:

1. **Muestrea hiperparámetros** del espacio de búsqueda:
   ```python
   filters = trial.suggest_int('filters', 16, 64, step=16)                    # 16, 32, 48, 64
   kernel_size = trial.suggest_int('kernel_size', 2, 5)                       # 2, 3, 4, 5
   lstm1_units = trial.suggest_int('lstm1_units', 32, 128, step=32)           # 32, 64, 96, 128
   lstm2_units = trial.suggest_int('lstm2_units', 16, 64, step=16)            # 16, 32, 48, 64
   dropout = trial.suggest_float('dropout', 0.1, 0.4)                         # Continuo [0.1, 0.4]
   recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, 0.2)     # Continuo [0.0, 0.2]
   learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True) # Escala log
   batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])         # 16, 32 o 64
   use_causal_padding = trial.suggest_categorical('use_causal_padding', [True, False])
   optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop', 'sgd'])
   ```

2. **Construye el modelo** con `build_model(hparams, input_shape)`

3. **Entrena el modelo** con callbacks:
   - `EarlyStopping`: Para cuando val_loss no mejora por `patience` épocas
   - `TFKerasPruningCallback`: Detiene trials no prometedores temprano

4. **Retorna val_loss** como métrica a minimizar

### Estrategia de Pruning

- **Pruner**: `MedianPruner`
  - `n_startup_trials=5`: Primeros 5 trials se completan sin pruning
  - `n_warmup_steps=10`: Primeras 10 épocas no se podan
- **Criterio**: Si val_loss de un trial es peor que la mediana de trials completados, se detiene

## Archivos Generados

### 1. Modelo Entrenado
**Ruta**: `models/lstm_final.h5`  
Modelo Keras con los mejores pesos encontrados.

**Cargar modelo:**
```python
import tensorflow as tf
model = tf.keras.models.load_model('models/lstm_final.h5')
```

### 2. Mejores Hiperparámetros
**Ruta**: `models/lstm_final_best_params.json`

**Estructura:**
```json
{
    "best_params": {
        "filters": 48,
        "kernel_size": 4,
        "lstm1_units": 96,
        "lstm2_units": 32,
        "dropout": 0.214,
        "recurrent_dropout": 0.00043,
        "learning_rate": 0.00225,
        "batch_size": 64,
        "use_causal_padding": false,
        "optimizer": "adam"
    },
    "best_value": 1.443,
    "best_trial": 1,
    "n_trials": 30,
    "timestamp": "2026-02-15T09:05:12"
}
```

### 3. Resumen del Estudio
**Ruta**: `models/lstm_final_optuna_study.json`

**Estructura:**
```json
{
    "best_trial_number": 17,
    "best_value": 0.8234,
    "best_params": {...},
    "n_trials": 30,
    "trials_summary": [
        {
            "number": 0,
            "value": 1.234,
            "params": {...},
            "state": "COMPLETE"
        },
        ...
    ]
}
```

### 4. Base de Datos Optuna
**Ruta**: `models/optuna_lstm.db` (SQLite)

Almacena todos los trials con detalles completos. Permite:
- Continuar optimización agregando más trials
- Visualizar resultados con Optuna Dashboard
- Analizar importancia de hiperparámetros

## Uso Básico

### Entrenar Nuevo Modelo

```bash
python src/train.py --stage lstm
```

Este comando:
1. Carga y prepara datos
2. Crea ventanas deslizantes (window_size=48, horizon=3)
3. Escala datos con StandardScaler
4. Ejecuta optimización Optuna (30 trials)
5. Guarda mejor modelo y metadatos

### Continuar Optimización

Si ya existe `optuna_lstm.db`, puedes agregar más trials modificando el código:

```python
# En la función train_lstm, el parámetro load_if_exists=True 
# permite continuar desde la base de datos existente
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(...),
    study_name="lstm_optimization",
    storage=optuna_db,
    load_if_exists=True  # <-- Clave para continuar
)
```

## Análisis de Resultados

### 1. Visualización con Optuna

```python
import optuna

# Cargar estudio
study = optuna.load_study(
    study_name="lstm_optimization",
    storage="sqlite:///models/optuna_lstm.db"
)

# Historia de optimización
optuna.visualization.plot_optimization_history(study).show()

# Importancia de hiperparámetros
optuna.visualization.plot_param_importances(study).show()

# Relaciones entre parámetros
optuna.visualization.plot_parallel_coordinate(study).show()

# Slice plot
optuna.visualization.plot_slice(study).show()
```

### 2. Análisis Manual

```python
import json

# Leer mejores parámetros
with open('models/lstm_final_best_params.json') as f:
    best = json.load(f)
    
print(f"Mejor val_loss: {best['best_value']}")
print(f"Encontrado en trial: {best['best_trial']}")
print(f"Hiperparámetros: {best['best_params']}")

# Leer todos los trials
with open('models/lstm_final_optuna_study.json') as f:
    study = json.load(f)
    
import pandas as pd
trials_df = pd.DataFrame(study['trials_summary'])
trials_df = trials_df[trials_df['state'] == 'COMPLETE']

# Top 5 trials
print(trials_df.nsmallest(5, 'value'))
```

## Personalización

### Modificar Espacio de Búsqueda

En la función `objective` dentro de `train_lstm`:

```python
def objective(trial):
    hparams = {
        # Ajustar rangos según necesidad
        'units': trial.suggest_int('units', 16, 256, step=16),  # Más opciones
        'dropout': trial.suggest_float('dropout', 0.0, 0.6),    # Rango ampliado
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'dense_units': trial.suggest_int('dense_units', 8, 128, step=8),
        'n_layers': trial.suggest_int('n_layers', 1, 3),        # Hasta 3 capas
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    }
    # ... resto del código
```

### Agregar Nuevos Hiperparámetros

**1. En `build_model`:**
```python
def build_model(hparams: dict, input_shape: tuple):
    # ... código existente ...
    
    # Nuevo: Activation function para Dense
    dense_activation = hparams.get('dense_activation', 'relu')
    modelo.add(layers.Dense(dense_units, activation=dense_activation))
```

**2. En `objective`:**
```python
def objective(trial):
    hparams = {
        # ... parámetros existentes ...
        'dense_activation': trial.suggest_categorical('dense_activation', ['relu', 'tanh', 'elu'])
    }
```

### Cambiar Métrica de Optimización

Por defecto se minimiza `val_loss`. Para usar otra métrica:

```python
def objective(trial):
    # ... entrenar modelo ...
    
    # Opción 1: Minimizar RMSE en validación
    val_rmse = min(history.history['val_rmse'])
    return val_rmse
    
    # Opción 2: Minimizar MAE en validación
    val_mae = min(history.history['val_mae'])
    return val_mae
```

## Consideraciones Importantes

### 1. Tiempo de Entrenamiento
- Cada trial puede tomar 5-15 minutos (dependiendo de epochs y patience)
- 30 trials ≈ 2.5-7.5 horas
- Usar `n_trials` menor para pruebas rápidas

### 2. Recursos Computacionales
- GPU recomendada para acelerar entrenamiento
- Memoria RAM: ~8GB mínimo
- Espacio en disco: ~500MB para base de datos y modelos

### 3. Early Stopping vs Pruning
- **Early Stopping**: Detiene entrenamiento del modelo actual si no mejora
- **Pruning**: Detiene el trial completo si no es prometedor vs otros
- Ambos trabajan juntos para eficiencia

### 4. Reproducibilidad
- TensorFlow usa semillas aleatorias internas
- Para reproducibilidad exacta, fijar todas las semillas:
  ```python
  import random
  import numpy as np
  import tensorflow as tf
  
  random.seed(42)
  np.random.seed(42)
  tf.random.set_seed(42)
  ```

## Solución de Problemas

### Error: "No module named 'optuna'"
```bash
pip install optuna
```

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Pruning muy agresivo (muchos trials PRUNED)
Ajustar parámetros del pruner:
```python
pruner=optuna.pruners.MedianPruner(
    n_startup_trials=10,   # Aumentar startup trials
    n_warmup_steps=20      # Aumentar warmup steps
)
```

### Val_loss siempre alto
- Verificar normalización de datos
- Aumentar `epochs` o disminuir `patience`
- Ampliar espacio de búsqueda de learning_rate

## Referencias

- [Documentación Optuna](https://optuna.readthedocs.io/)
- [Keras/TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Optuna + Keras Integration](https://optuna.readthedocs.io/en/stable/reference/integration.html#tensorflow-keras)

---

*Última actualización: 14 de febrero de 2026*
