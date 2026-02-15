# 🎯 Guía: LSTM con Optimización Multiobjetivo (MAE + RMSE)

## 📋 Resumen Ejecutivo

Esta guía explica cómo usar optimización multiobjetivo en Optuna para entrenar LSTM que compite directamente con XGBoost en las métricas MAE y RMSE.

### **Ventajas vs Optimización de Objetivo Único:**
- ✅ Optimiza directamente MAE (métrica clave para competir con XGBoost: 0.76°C)
- ✅ Balancea MAE y RMSE simultáneamente (Pareto Front)
- ✅ Pruning inteligente corta trials malos temprano (ahorra tiempo)
- ✅ Loss alineada con métrica de evaluación (`loss="mae"`)

---

## 🎯 Objetivos Multiobjetivo Seleccionados

### **2 Objetivos Mínimos Viables:**

1. **Minimizar MAE en validación (escala real)**
   - Métrica directa comparable con XGBoost (MAE = 0.76°C)
   - Interpreta errores absolutos promedio en °C
   
2. **Minimizar RMSE en validación (escala real)**
   - Penaliza outliers (errores grandes)
   - Balance entre precisión general y robustez

### **Por qué estos 2:**
- **MAE**: Tu competidor XGBoost tiene MAE=0.76°C. Optimizar MAE directamente te permite competir en esta métrica clave
- **RMSE**: Complementa MAE al penalizar errores grandes (importante cuando picos de temperatura importan)
- **Escala real**: Evaluar en °C (no normalizado) hace métricas interpretables

---

## 🔧 Checklist Operativo

### **✅ Paso 1: Preparar Datos (si no lo hiciste)**

```bash
python src/train.py --stage separar_datos
```

Esto genera `models/df_train_lstm.joblib` con datos limpios.

---

### **✅ Paso 2: Entrenar con Multiobjetivo**

```bash
python src/train.py --stage lstm_multiobj
```

**¿Qué hace internamente?**
1. Carga datos preparados
2. Crea ventanas temporales (window_size=48, horizon=3)
3. Escala con StandardScaler
4. Optimiza con Optuna:
   - **Objetivos**: `(MAE, RMSE)` en validación
   - **Pruning**: MedianPruner corta trials malos después de 20 epochs
   - **Loss**: `"mae"` (alineada con métrica objetivo)
5. Guarda:
   - Mejor modelo: `models/lstm_multiobj_final.h5`
   - Pareto Front: `models/lstm_multiobj_final_pareto.json`
   - Metadatos: `models/metadata/lstm_multiobj_metadatos.json`
   - Gráficos: `reports/figures/optuna_multiobj/`

---

### **✅ Paso 3: Analizar Resultados del Pareto Front**

El Pareto Front contiene todas las soluciones **no-dominadas** (no hay otra solución mejor en ambos objetivos).

**Archivo generado:** `models/lstm_multiobj_final_pareto.json`

**Ejemplo:**
```json
{
    "pareto_trials": [
        {
            "trial_number": 5,
            "mae": 0.98,
            "rmse": 1.25
        },
        {
            "trial_number": 12,
            "mae": 1.02,
            "rmse": 1.20
        }
    ]
}
```

**Gráfico Pareto:** `reports/figures/optuna_multiobj/pareto_front_multiobj.png`

---

### **✅ Paso 4: Seleccionar Modelo Final**

**Estrategias:**

#### **Opción 1: Mínimo MAE (competir con XGBoost)**
```python
import json

with open('models/lstm_multiobj_final_pareto.json') as f:
    pareto = json.load(f)

best_mae = min(pareto['pareto_trials'], key=lambda t: t['mae'])
print(f"Mejor MAE: Trial {best_mae['trial_number']}, MAE={best_mae['mae']:.4f}")
```

✅ **Recomendado si tu objetivo es superar XGBoost en MAE.**

#### **Opción 2: Mejor balance MAE/RMSE**
```python
import numpy as np

# Distancia euclidiana al origen (0,0)
best_balanced = min(
    pareto['pareto_trials'],
    key=lambda t: np.sqrt(t['mae']**2 + t['rmse']**2)
)
print(f"Mejor balance: Trial {best_balanced['trial_number']}")
```

#### **Opción 3: MAE bajo umbral, luego mínimo RMSE**
```python
threshold = 1.0  # MAE objetivo

candidates = [t for t in pareto['pareto_trials'] if t['mae'] <= threshold]
if candidates:
    best = min(candidates, key=lambda t: t['rmse'])
    print(f"Mejor RMSE con MAE<{threshold}: Trial {best['trial_number']}")
```

---

### **✅ Paso 5: Comparar con Modelos Anteriores**

| Modelo      | RMSE   | MAE   | Método          |
|-------------|--------|-------|-----------------|
| Ridge       | 1.51°C | 1.19°C | GridSearch     |
| XGBoost     | 1.00°C | 0.76°C | GridSearch     |
| LSTM (v2)   | 1.29°C | 1.03°C | Optuna (val_loss) |
| **LSTM Multiobj** | **?** | **?** | **Optuna (MAE+RMSE)** |

**Meta:** MAE < 0.76°C (superar XGBoost)

---

## 🔬 Detalles Técnicos

### **1. Loss Function: MAE vs MSE**

**Configuración actual:** `loss="mae"`

**Por qué MAE:**
- ✅ Alineada con métrica de evaluación (MAE)
- ✅ Reduce discrepancia entre loss de entrenamiento y métrica final
- ✅ Ayuda a competir con XGBoost que optimiza objetivos absolutos

**Alternativa robusta:** Huber Loss
```python
loss=tf.keras.losses.Huber(delta=1.0)
```
- Usa si tienes outliers extremos en temperatura
- Balancea MAE (robustez) y MSE (suavidad)

### **2. Pruning con MedianPruner**

**Configuración:**
```python
pruner=optuna.pruners.MedianPruner(
    n_startup_trials=5,    # No podar primeros 5 trials
    n_warmup_steps=20,     # Esperar 20 epochs antes de evaluar poda
    interval_steps=5       # Evaluar poda cada 5 epochs
)
```

**Cómo funciona:**
- Reporta `val_mae` cada epoch
- Si `val_mae` es peor que la mediana de trials pasados → **Poda trial**
- Ahorra tiempo al cortar trials no prometedores temprano

### **3. Callback de Pruning Personalizado**

```python
class OptunaPruningCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get('val_mae')
        self.trial.report(current_value, epoch)
        
        if self.trial.should_prune():
            self.model.stop_training = True
            raise optuna.TrialPruned()
```

**Ventaja:** Integración nativa con Keras (no necesitas TFKerasPruningCallback modificado)

---

## 📊 Interpretación de Resultados

### **Gráficos Generados:**

1. **Pareto Front** (`pareto_front_multiobj.png`)
   - Scatter: todos los trials
   - Estrellas rojas: soluciones Pareto
   - X verde: mejor por MAE

2. **Optimization History** (si usas `plot_optuna_study`)
   - Evolución del mejor MAE a lo largo de trials

3. **Parameter Importances**
   - Identifica hiperparámetros más influyentes

### **Métricas en Logs:**

```
Trial 12: MAE=1.02, RMSE=1.25, gap=0.03
```

- **MAE**: Error absoluto medio en validación (°C)
- **RMSE**: Raíz del error cuadrático medio (°C)
- **gap**: `|train_mae - val_mae|` (indica overfitting si >0.2)

---

## 🚀 Siguientes Pasos

### **Si MAE > 0.76 (peor que XGBoost):**

1. **Aumentar trials:** `n_trials=50` o más
2. **Ajustar rangos de hiperparámetros:**
   - Probar `lstm1_units` más grandes: `(64, 256)`
   - Explorar `learning_rate` más bajos: `(1e-5, 1e-3)`
3. **Usar Huber Loss:** `loss=Huber(delta=0.5)`
4. **Aumentar `window_size`:** de 48 a 72 horas

### **Si MAE < 0.76 (mejor que XGBoost):**

1. ✅ **Actualizar tabla de comparación** en `docs/model_training.md`
2. ✅ **Evaluar en conjunto de test final**
3. ✅ **Guardar modelo para producción**
4. ✅ **Documentar mejores prácticas aprendidas**

---

## 📚 Recursos Adicionales

### **Documentación Optuna:**
- [Multi-Objective Optimization](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/005_multi_objective.html)
- [Pruning](https://optuna.readthedocs.io/en/stable/reference/pruners.html)

### **Archivos del Proyecto:**
- **Código fuente:** [src/train.py](../src/train.py) (función `train_lstm_multiobj`)
- **Metadatos:** `models/metadata/lstm_multiobj_metadatos.json`
- **Gráficos:** `reports/figures/optuna_multiobj/`

---

## ❓ Preguntas Frecuentes

### **¿Por qué no usar 3 objetivos (MAE, RMSE, gap)?**
- Pareto Front con 3+ objetivos es difícil de visualizar
- Gap se controla con early stopping (patience)
- Empezar con 2 objetivos simplifica decisión

### **¿Cuántos trials necesito?**
- Mínimo: 20-30 trials
- Recomendado: 50-100 trials para exploración robusta
- Con pruning, puedes permitirte más trials (descarta malos temprano)

### **¿Cómo elijo entre soluciones del Pareto Front?**
- Si priorizas competir en MAE: **mínimo MAE**
- Si quieres robustez: **mejor balance MAE/RMSE**
- Si tienes restricción operativa: **MAE < umbral, luego mínimo RMSE**

### **¿Qué pasa si el modelo se detiene temprano (pruning)?**
- Es **esperado** → pruning funciona correctamente
- Trials no prometedores se cortan → ahorra tiempo
- Observa logs: si >50% trials se podan, ajusta `n_warmup_steps`

---

## 📝 Ejemplo Completo de Ejecución

```bash
# 1. Preparar datos (una sola vez)
python src/train.py --stage separar_datos

# 2. Entrenar LSTM multiobjetivo
python src/train.py --stage lstm_multiobj

# 3. Revisar resultados
cat models/metadata/lstm_multiobj_metadatos.json

# 4. Visualizar Pareto Front
# Abrir: reports/figures/optuna_multiobj/pareto_front_multiobj.png
```

**Tiempo estimado:** 2-4 horas (30 trials, ~5-8 min por trial)

---

[← Volver al README principal](../README.md) | [Ver Guía LSTM Original](optuna_lstm_guide.md)
