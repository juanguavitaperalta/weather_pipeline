# 📈 Análisis Temporal

Este documento presenta el análisis de series temporales realizado para identificar patrones y determinar los lags óptimos para el modelo predictivo.

---

## 📉 Series Temporales

### Temperatura
<p align="center">
  <img src="../reports/analisis_temporal/series/serie_temperature_2m.png" width="800">
</p>

### Humedad Relativa
<p align="center">
  <img src="../reports/analisis_temporal/series/serie_relative_humidity_2m.png" width="800">
</p>

### Velocidad del Viento
<p align="center">
  <img src="../reports/analisis_temporal/series/serie_wind_speed_10m.png" width="800">
</p>

---

## 🔄 Autocorrelación (ACF & PACF)

El análisis de autocorrelación permite identificar la relación de la temperatura con sus valores pasados.

### Temperatura - ACF y PACF
<p align="center">
  <img src="../reports/analisis_temporal/acf/acf%20%26%20pacf_temperature_2m.png" width="800">
</p>

**Interpretación:**
- **ACF (Autocorrelation Function):** Mide la correlación lineal entre la serie temporal en un instante de tiempo y ella misma desplazada k periodos.
- **PACF (Partial Autocorrelation Function):** Mide la correlación directa entre diferentes instantes, controlando por los valores intermedios.

---

## 🔗 Correlación Cruzada

La correlación cruzada identifica qué valores pasados de las variables independientes ayudan a explicar los valores futuros de la temperatura.

### Temperatura vs Humedad Relativa
<p align="center">
  <img src="../reports/analisis_temporal/crosscorr/cross_corr_temperature_2m_relative_humidity_2m.png" width="700">
</p>

### Temperatura vs Velocidad del Viento
<p align="center">
  <img src="../reports/analisis_temporal/crosscorr/cross_corr_temperature_2m_wind_speed_10m.png" width="700">
</p>

---

## 📝 Conclusiones del Análisis Temporal

1. **Estacionalidad clara:** Se observa un patrón diario (24 horas) en la temperatura.
2. **Lags significativos:** Los lags 1, 2, 3, 6, 12 y 24 horas muestran correlación significativa.
3. **Correlación cruzada:** La humedad relativa tiene correlación negativa con la temperatura en lags de 6-12 horas.

Para más detalles sobre la selección de lags, ver [Selección de Lags](lag_selection.md).

---

[← Volver al README principal](../README.md)
