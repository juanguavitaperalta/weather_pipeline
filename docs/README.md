# Weather Pipeline - Predicción de Temperatura

# Introducción

Este proyecto implementa un pipeline industrial de de Machine learning para forecasting de temperatura a corto plazo. En este Pipeline aplicamos los siguientes conceptos y algoritmos:

1. Análisis exploratorio: Generación de métricas de estadistica descriptiva para analizar el comportamiento de las variables del datase. Medidas necesarias para tomar decisiones adecuadas en la limpieza de datos.
2. Análisis temporal: Análisis del comportamiento de las variables en función del tiempo.
Analisis para variable objetivo: Temperatura
    - ACF, medimos la correlación lineal entre la serie temporal en una instante de tiempo y ela misma desplazada k periodos para determinar relaciones entre instantes.
    - PACF mide correlación directa de la serie entre diferentes instantes de tiempos. Intenta explicar si hay relaciones entre instantes de tiempo que no hayan sido identificados en iteraciones anteriores.
Analisis para variables independientes: Velocidad del viente y humedad relativa.
    - Correlación cruzada entre estas variables y la temperatura se utilizó para identificar que valores pasados de las variables independientes ayudan a explicar los valores futuros de la variable objetivo.
Deste análisis se genera un documento con el criterio de selección de lags para las variables del dataset.
3. Limpieza y preparación del dataset: La identificación de los lags permite generar nuevas varibles en el dataset mas variables que representan ciclos temporales, para representar el comportamiento ciclico de las variables climaticas.
4. Entrenamiento de modelos: Los siguientes modelos fueron tenidos en cuenta para realizar una comparación y selección del mejor modelo. Los siguientes modelos fueron evaluados en esta sección:
    - Lasso, Ridge y elastic net.
    - XGBoost.
5. Predicción y comparación de rendimiento vs con un modelo comercial profet.

## 📁 Estructura del Proyecto

```
weather_pipeline/
├── configs/
│   └── config.yaml           # Configuración de API y rutas
├── data/
│   ├── raw/                  # Datos crudos descargados
│   ├── processed/            # Datos limpios
│   ├── features/             # Features para entrenamiento
│   └── predict_data/         # Datos y resultados de predicción
├── docs/
│   ├── README.md             # Este archivo
│   ├── lag_selection.md      # Documentación de selección de lags
│   └── orden_flujo.md        # Orden de ejecución del pipeline
├── models/
│   ├── xgboost_final.joblib  # Modelo entrenado
│   └── metadata/             # Metadatos del modelo
├── reports/
│   ├── analisis_temporal/    # Gráficas ACF, PACF, cross-correlation
│   └── figures/              # Gráficas de predicción y SHAP
├── src/
│   ├── ingest.py             # Descarga de datos
│   ├── explore.py            # Análisis exploratorio
│   ├── clean.py              # Limpieza de datos
│   ├── features.py           # Ingeniería de características
│   ├── temporal_diagnostics.py # Diagnósticos de series temporales
│   ├── train.py              # Entrenamiento de modelos
│   ├── predict.py            # Predicción y comparación
│   └── utils.py              # Utilidades
└── tests/
    └── test_clean.py         # Tests unitarios
```

## 🚀 Instalación

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows)
.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

## 📊 Flujo de Ejecución

### 1. Ingesta de datos
```bash
python src/ingest.py
```
Descarga datos históricos de Open-Meteo según fechas configuradas en `config.yaml`.

### 2. Exploración
```bash
python src/explore.py
```
Genera estadísticas descriptivas y detecta valores faltantes.

### 3. Limpieza
```bash
python src/clean.py
```
Trata datos faltantes, convierte tipos y elimina duplicados.

### 4. Diagnósticos temporales
```bash
python src/temporal_diagnostics.py
```
Genera gráficas ACF/PACF, cross-correlation y tests de estacionariedad.

### 5. Ingeniería de características
```bash
python src/features.py
```
Crea lags y variable objetivo (`temperature_2m_target`).

### 6. Entrenamiento
```bash
# Solo separar datos (sin entrenar)
python src/train.py --stage separar_datos

# Entrenar modelos lineales
python src/train.py --stage lineales

# Entrenar XGBoost (default)
python src/train.py --stage xgboost

# Curvas de aprendizaje
python src/train.py --stage curvas_aprendizaje

# Análisis SHAP
python src/train.py --stage shap
```

### 7. Predicción
```bash
python src/predict.py
```
Genera predicciones con XGBoost y Prophet, compara métricas.

## ⚙️ Configuración

Editar `configs/config.yaml`:

```yaml
open_meteo:
  latitude: 4.7110
  longitude: -74.0721
  timezone: America/Bogota
  hourly:
    - temperature_2m
    - relative_humidity_2m
    - wind_speed_10m
  start_date: "2024-01-01"
  end_date: "2024-06-30"

paths:
  raw_dir: data/raw
```

## 📈 Métricas del Modelo

| Modelo | RMSE | MAE | Skill vs Prophet |
|--------|------|-----|------------------|
| **XGBoost** | 1.32°C | 1.01°C | +20.22% |
| Prophet | 1.66°C | 1.29°C | baseline |

## 🔧 Features Utilizadas

- **Variables meteorológicas:** `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`
- **Lags temporales:** 1, 2, 3, 6, 12, 24 horas
- **Estacionalidad:** `hour`, `dayofweek`, `month`, `sin_comp`, `cos_comp`

## 📂 Archivos de Salida

| Archivo | Descripción |
|---------|-------------|
| `models/xgboost_final.joblib` | Modelo XGBoost entrenado |
| `models/metadata/xgboost_metadatos.json` | Hiperparámetros y métricas |
| `data/predict_data/predicciones_junio.xlsx` | Predicciones XGBoost |
| `data/predict_data/comparacion_modelos.xlsx` | XGBoost vs Prophet |
| `reports/figures/predicciones/` | Gráficas de predicción |
| `reports/figures/shap/` | Análisis SHAP |

## 🧪 Tests

```bash
pytest tests/
```

## 📝 Notas

- **Horizonte de predicción:** 3 horas
- **Datos de entrenamiento:** Enero - Mayo 2024
- **Datos de predicción:** Junio 2024
- **API:** Open-Meteo (gratuita, sin API key)
