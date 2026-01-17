# 📈 Análisis Temporal

Este documento presenta el análisis de series temporales realizado para identificar patrones y determinar los lags óptimos para el modelo predictivo.

---

## 📉 Series Temporales

### Temperatura
<p align="center">
  <img src="../reports/analisis_temporal/series/serie_temperature_2m.png" width="800">
</p>

**Observaciones:**
- Se observa un patrón cíclico diario claro con máximos alrededor del mediodía (hasta 26.2°C) y mínimos en la madrugada (hasta 4.1°C).
- La temperatura media es de **14.46°C** con una desviación estándar de 4.49°C.
- La amplitud térmica diaria oscila aproximadamente entre 9°C y 22°C, típico del clima de Bogotá.
- No se evidencian tendencias marcadas a largo plazo en el periodo Enero-Junio 2024.
- La estacionalidad diaria (24 horas) es el componente dominante de la serie.

### Humedad Relativa
<p align="center">
  <img src="../reports/analisis_temporal/series/serie_relative_humidity_2m.png" width="800">
</p>

**Observaciones:**
- Comportamiento inversamente correlacionado con la temperatura: máximos en la noche/madrugada y mínimos al mediodía.
- La humedad media es de **79.86%** con alta variabilidad (σ = 20.86%).
- Rango de valores entre 18% (días secos al mediodía) y 100% (saturación frecuente).
- El 50% de los datos supera el 90% de humedad (mediana), indicando condiciones predominantemente húmedas.
- Saturación frecuente (100%) en horas nocturnas, característico del clima tropical de montaña de Bogotá.

### Velocidad del Viento
<p align="center">
  <img src="../reports/analisis_temporal/series/serie_wind_speed_10m.png" width="800">
</p>

**Observaciones:**
- Mayor variabilidad y comportamiento menos predecible que temperatura y humedad.
- Velocidad media de **5.10 km/h** con desviación estándar de 3.62 km/h.
- Rango entre calma total (0 km/h) y ráfagas de hasta 19.8 km/h.
- Patrón diario presente: vientos más intensos en horas de la tarde (16:00-18:00), coincidiendo con el enfriamiento.
- La distribución está sesgada a la derecha (mediana 3.9 km/h < media 5.1 km/h), indicando eventos ocasionales de vientos fuertes.

---

## 🔄 Autocorrelación (ACF & PACF)

El análisis de autocorrelación permite identificar la relación de la temperatura con sus valores pasados.

### Temperatura - ACF y PACF
<p align="center">
  <img src="../reports/analisis_temporal/acf/acf%20%26%20pacf_temperature_2m.png" width="800">
</p>

**Interpretación:**
- **ACF (Autocorrelation Function):** Mide la correlación lineal entre la serie temporal en un instante de tiempo y ella misma desplazada k periodos, para este caso horas. Para el gráfico de la variable objetivo, presenta un comportamiento ciclico, el cual es esperable por su comportamiento ciclico.

Si esta correlación es significativa, superará la bandas para determinar que son estadisticamente significativas. Estas bandas en la teoria, deberian ser estables bajo las siguientes condiciones:
1. Es una serie estacionaria.
2. Varianza aproximadamente constante.
3. Ruido es blanco, media igual a cero, varianza constantante, no hay memoria temporal.

Sin embargo, las bandas presentan un comportamiento creciente, debido a las siguientes razones:

1. Cuando se presenta un mayor k de retraso, existen menos observaciones disponibles y mayor varianza, contradiciendo el principio de ruido blanco (var k).

Debido a que al aumentar los lags, aumenta la incertidumbre, entonces debemos seleccionar lags pequeños que superen el intervalo de confianza y aquellos que representen el periodo de la señal de oscilación envolvente.

2. Al ser una variable cíclica, esta presenta estacionalidad y tiene una memoria temporal fuerte. 
3. Al ver la gráfica de la temperatura, la media es diferente de cero.

- **PACF (Partial Autocorrelation Function):** Mide la correlación directa entre diferentes instantes, controlando por los valores intermedios.

---

## 🔗 Correlación Cruzada

La correlación cruzada identifica qué valores pasados de las variables independientes ayudan a explicar los valores futuros de la temperatura.

En esta grafica se puede observar que los dos primeros lags (1 y 2), presentan alta dependencia.

Conclusión: Se seleccionaran los Lags 1, 2, 3 y 24 para la generación de información en el dataset para la variable de tremperatura.
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
## 
1. **Estacionalidad clara:** Se observa un patrón diario (24 horas) en la temperatura.
2. **Lags significativos (Temperatura):** Los lags 1, 2, 3 y 24 horas muestran correlación significativa.
3. **Correlación cruzada:** La humedad relativa tiene correlación negativa con la temperatura en lags de 6-12 horas.

Para más detalles sobre la selección de lags, ver [Selección de Lags](lag_selection.md).

---

[← Volver al README principal](../README.md)
