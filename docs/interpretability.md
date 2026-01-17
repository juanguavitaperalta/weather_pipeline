# 🔍 Interpretabilidad del Modelo - SHAP

Este documento presenta el análisis de interpretabilidad del modelo XGBoost utilizando SHAP (SHapley Additive exPlanations).

---

## 📊 Importancia de Features

### Feature Importance Global
<p align="center">
  <img src="../reports/figures/shap/shap_summary_bar.png" width="700">
</p>

**Interpretación:** Las barras muestran la importancia promedio de cada feature en las predicciones del modelo.

---

## 🐝 SHAP Beeswarm Plot

<p align="center">
  <img src="../reports/figures/shap/shap_beeswarm.png" width="700">
</p>

**Interpretación:** 
- Cada punto representa una observación
- El color indica el valor de la feature (rojo = alto, azul = bajo)
- La posición horizontal indica el impacto en la predicción

---

## 📈 SHAP Summary Dot Plot

<p align="center">
  <img src="../reports/figures/shap/shap_summary_dot.png" width="700">
</p>

---

## 🔗 Gráficas de Dependencia

Las gráficas de dependencia muestran cómo el valor de una feature afecta la predicción.

### Temperatura Actual
<p align="center">
  <img src="../reports/figures/shap/shap_dependence_temperature_2m.png" width="600">
</p>

### Hora del Día
<p align="center">
  <img src="../reports/figures/shap/shap_dependence_hour.png" width="600">
</p>

### Humedad Relativa (Lag 12h)
<p align="center">
  <img src="../reports/figures/shap/shap_dependence_relative_humidity_2m_lag_12.png" width="600">
</p>

### Componente Seno (Estacionalidad)
<p align="center">
  <img src="../reports/figures/shap/shap_dependence_sin_comp.png" width="600">
</p>

### Componente Coseno (Estacionalidad)
<p align="center">
  <img src="../reports/figures/shap/shap_dependence_cos_comp.png" width="600">
</p>

### Día de la Semana
<p align="center">
  <img src="../reports/figures/shap/shap_dependence_dayofweek.png" width="600">
</p>

---

## 🌊 Waterfall Plots

Los waterfall plots muestran cómo cada feature contribuye a una predicción individual.

### Predicción 1
<p align="center">
  <img src="../reports/figures/shap/shap_waterfall_pred_1.png" width="700">
</p>

### Predicción 2
<p align="center">
  <img src="../reports/figures/shap/shap_waterfall_pred_2.png" width="700">
</p>

### Predicción 3
<p align="center">
  <img src="../reports/figures/shap/shap_waterfall_pred_3.png" width="700">
</p>

---

## 🎯 Force Plots Interactivos

Para visualizaciones interactivas, abrir los siguientes archivos HTML:

- [Force Plot Individual](../reports/figures/shap/shap_force_plot_single.html)
- [Force Plot Múltiple](../reports/figures/shap/shap_force_plot_multi.html)

---

## 📝 Conclusiones de Interpretabilidad

1. **Feature más importante:** La temperatura actual (`temperature_2m`) es el predictor más fuerte.
2. **Estacionalidad relevante:** Los componentes sin/cos capturan el ciclo diario.
3. **Humedad como predictor secundario:** Los lags de humedad aportan información complementaria.
4. **Modelo interpretable:** Las relaciones capturadas son físicamente coherentes.

---

## 📂 Archivos de Datos

| Archivo | Descripción |
|---------|-------------|
| `reports/figures/shap/shap_feature_importance.csv` | Importancia de features |
| `reports/figures/shap/shap_values_test.csv` | Valores SHAP para test set |

---

[← Volver al README principal](../README.md)
