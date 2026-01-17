# 🎯 Predicciones y Resultados

Este documento presenta los resultados de predicción del modelo XGBoost y su comparación con Prophet.

---

## 📊 Métricas Finales

| Modelo | RMSE | MAE | Skill vs Prophet |
|--------|------|-----|------------------|
| **XGBoost** | 1.32°C | 1.01°C | **+20.22%** |
| Prophet | 1.66°C | 1.29°C | baseline |

---

## 📈 Comparación Visual

### XGBoost vs Prophet
<p align="center">
  <img src="../reports/figures/predicciones/comparacion_xgboost_prophet.png" width="800">
</p>

### Serie Temporal de Predicciones
<p align="center">
  <img src="../reports/figures/predicciones/prediccion_serie_temporal.png" width="800">
</p>

---

## 🔍 Análisis de Errores

### Scatter Plot: Predicción vs Real
<p align="center">
  <img src="../reports/figures/predicciones/prediccion_scatter.png" width="600">
</p>

### Comparación Scatter
<p align="center">
  <img src="../reports/figures/predicciones/comparacion_scatter.png" width="600">
</p>

### Distribución de Errores
<p align="center">
  <img src="../reports/figures/predicciones/prediccion_errores_hist.png" width="600">
</p>

### Errores en el Tiempo
<p align="center">
  <img src="../reports/figures/predicciones/prediccion_errores_tiempo.png" width="800">
</p>

---

## 📝 Conclusiones

1. **XGBoost supera a Prophet** por un margen significativo (+20.22% skill).
2. **Errores distribuidos normalmente:** No hay sesgo sistemático en las predicciones.
3. **Rendimiento consistente:** Los errores no muestran patrones temporales significativos.

---

## 📂 Archivos de Salida

| Archivo | Descripción |
|---------|-------------|
| `data/predict_data/predicciones_junio.xlsx` | Predicciones XGBoost |
| `data/predict_data/comparacion_modelos.xlsx` | Comparación XGBoost vs Prophet |
| `reports/figures/predicciones/metricas_comparacion.csv` | Métricas comparativas |

---

[← Volver al README principal](../README.md)
