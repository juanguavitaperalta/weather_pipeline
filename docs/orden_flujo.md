# Orden de Flujo del Pipeline

Este documento describe el orden de ejecución de los scripts del pipeline de datos meteorológicos.

## Flujo de Ejecución

```
1. ingest.py → 2. explore.py → 3. clean.py → 4. temporal_diagnostics.py → 4. features.py → 5. train.py → 6. predict.py
```

## Descripción de cada paso

### 1. Ingesta de datos (`ingest.py`)
- Descarga datos históricos de Open-Meteo API
- Configurable por fechas en `configs/config.yaml`
- **Entrada:** Configuración YAML
- **Salida:** `data/raw/open_meteo_hourly.csv`

```bash
python src/ingest.py
```

### 2. Exploración (`explore.py`)
- Análisis exploratorio inicial de los datos
- Estadísticas descriptivas y detección de valores faltantes
- **Entrada:** `data/raw/open_meteo_hourly.csv`
- **Salida:** Logs con estadísticas

```bash
python src/explore.py
```

### 3. Limpieza (`clean.py`)
- Tratamiento de datos faltantes
- Conversión de tipos de datos
- Eliminación de duplicados
- **Entrada:** `data/raw/open_meteo_hourly.csv`
- **Salida:** `data/processed/open_meteo_hourly_clean.csv`

```bash
python src/clean.py
```

### 4. Ingeniería de características (`features.py`)
- Creación de variables temporales (hora, día, mes)
- Generación de lags y variables rezagadas
- **Entrada:** `data/processed/open_meteo_hourly_clean.csv`
- **Salida:** `data/features/features.csv`

```bash
python src/features.py
```

### 5. Entrenamiento (`train.py`)
- Entrenamiento del modelo XGBoost
- Evaluación y métricas de rendimiento
- **Entrada:** `data/features/features.csv`
- **Salida:** `models/xgboost_final.joblib`, `models/metadata/xgboost_metadatos.json`

```bash
python src/train.py
```

### 6. Predicción (`predict.py`)
- Generación de predicciones con el modelo entrenado
- **Entrada:** Modelo entrenado + nuevos datos
- **Salida:** Predicciones

```bash
python src/predict.py
```

## Scripts auxiliares

### Diagnósticos temporales (`temporal_diagnostics.py`)
- Análisis de autocorrelación (ACF)
- Correlación cruzada entre variables
- **Salida:** `reports/analisis_temporal/`

```bash
python src/temporal_diagnostics.py
```

## Ejecución completa

Para ejecutar todo el pipeline en orden:

```bash
python src/ingest.py
python src/explore.py
python src/clean.py
python src/temporal_diagnostics.py
python src/features.py
python src/train.py
python src/predict.py
```
