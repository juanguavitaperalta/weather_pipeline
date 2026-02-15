# 🎯 Respuesta: Optimización Multiobjetivo LSTM

## ✅ Implementación Completada

He implementado optimización multiobjetivo (MAE + RMSE) con pruning para tu modelo LSTM.

---

## 🚀 Uso Rápido

### **1. Entrenar LSTM Multiobjetivo**

```bash
python src/train.py --stage lstm_multiobj
```

### **2. Analizar Resultados**

```bash
python scripts/analizar_pareto.py
```

### **3. Visualizar Pareto Front**

Abrir: `reports/figures/optuna_multiobj/pareto_front_multiobj.png`

---

## 📊 Objetivos Optimizados

✅ **Objetivo 1: Minimizar MAE** (competir con XGBoost: 0.76°C)  
✅ **Objetivo 2: Minimizar RMSE** (penalizar outliers)

**Ambos en escala real (°C)** → Interpretables directamente

---

## 🔬 Cambios Técnicos Implementados

### **1. Nueva función: `train_lstm_multiobj()`**

Ubicación: `src/train.py` (líneas ~1180-1380)

**Características:**
- **Objetivos:** `(val_mae, val_rmse)` en validación
- **Pruning:** MedianPruner (corta trials malos después de 20 epochs)
- **Loss:** `"mae"` (alineada con métrica objetivo)
- **Callback personalizado:** Reporta `val_mae` cada epoch

### **2. Función auxiliar: `_plot_pareto_front()`**

Visualiza todas las soluciones Pareto y marca el mejor trial por MAE.

### **3. Nuevo stage: `lstm_multiobj`**

Agregado a `main()` con argumento `--stage lstm_multiobj`

---

## 📁 Archivos Generados

```
models/
  ├── lstm_multiobj_final.h5              # Mejor modelo (por MAE)
  ├── lstm_multiobj_final_best_params.json # Hiperparámetros del mejor trial
  ├── lstm_multiobj_final_pareto.json      # Todas las soluciones Pareto
  └── metadata/
      └── lstm_multiobj_metadatos.json     # Metadatos completos

reports/
  ├── figures/
  │   ├── optuna_multiobj/
  │   │   ├── pareto_front_multiobj.png   # Visualización Pareto Front
  │   │   ├── optimization_history.png
  │   │   ├── param_importances.png
  │   │   └── timeline.png
  │   ├── curvas aprendizaje/
  │   │   └── lstm_multiobj/
  │   │       └── lstm_learning_curves.png
  │   └── predicciones/
  │       └── lstm_multiobj_pred_vs_actual.png
  └── pareto_summary.txt                   # Resumen del análisis
```

---

## 🎓 Respuesta a tus Preguntas

### **1️⃣ Diseño Multiobjetivo**

**Objetivos seleccionados:**
1. **MAE** (minimizar)
2. **RMSE** (minimizar)

**Por qué estos 2:**
- MAE: Tu competidor XGBoost tiene MAE=0.76°C → optimizar MAE directamente
- RMSE: Complementa MAE al penalizar errores grandes
- **No incluí gap** (train-val) como tercer objetivo porque:
  - Se controla con early stopping (patience=40)
  - Pareto Front 3D es difícil de visualizar/decidir

---

### **2️⃣ Pruning con Entrenamiento Keras**

**Implementación:**

```python
class OptunaPruningCallback(callbacks.Callback):
    def __init__(self, trial, monitor='val_mae'):
        self.trial = trial
        self.monitor = monitor
    
    def on_epoch_end(self, epoch, logs=None):
        # Reportar métrica de pruning
        current_value = logs.get(self.monitor)
        self.trial.report(current_value, epoch)
        
        # Verificar si debe podarse
        if self.trial.should_prune():
            self.model.stop_training = True
            raise optuna.TrialPruned()
```

**Configuración del pruner:**
```python
pruner=optuna.pruners.MedianPruner(
    n_startup_trials=5,    # No podar primeros 5 trials
    n_warmup_steps=20,     # Esperar 20 epochs antes de evaluar
    interval_steps=5       # Evaluar poda cada 5 epochs
)
```

**Función objective() devuelve:**
```python
return val_mae, val_rmse  # Tupla multiobjetivo
```

---

### **3️⃣ Alineación Loss vs Métrica**

**Cambio implementado:**

```python
model.compile(
    optimizer=optimizer,
    loss="mae",  # Alineado con MAE objetivo
    metrics=[
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        tf.keras.metrics.MeanAbsoluteError(name="mae")
    ]
)
```

**Por qué ayuda vs XGBoost:**
- XGBoost usa `reg:squarederror` por defecto (optimiza MSE)
- Al usar `loss="mae"`, LSTM optimiza **exactamente** la misma métrica que evalúas
- Reduce bias hacia minimizar errores grandes vs errores promedio

**Alternativa robusta (si tienes outliers):**
```python
loss=tf.keras.losses.Huber(delta=1.0)
```

Para activarla, usa:
```python
train_lstm_multiobj(..., use_mae_loss=False)  # Usará Huber
```

---

### **4️⃣ Checklist Operativo**

#### **Paso 1: Preparar datos** ✅
```bash
python src/train.py --stage separar_datos
```

#### **Paso 2: Entrenar multiobjetivo** ✅
```bash
python src/train.py --stage lstm_multiobj
```

#### **Paso 3: Analizar Pareto Front** ✅
```bash
python scripts/analizar_pareto.py
```

**Salida esperada:**
```
🏆 MEJOR SOLUCIÓN POR MAE:
   Trial: 12
   MAE:   0.98°C
   RMSE:  1.25°C

🎯 MEJORA VS XGBOOST:
   MAE:  ❌ -28.9% peor  (o ✅ +2.6% mejor si supera)
```

#### **Paso 4: Seleccionar modelo final** ✅

**Opción A: Mejor MAE** (recomendado para competir con XGBoost)
```python
import json
with open('models/lstm_multiobj_final_pareto.json') as f:
    pareto = json.load(f)

best_mae = min(pareto['pareto_trials'], key=lambda t: t['mae'])
print(f"Trial: {best_mae['trial_number']}, MAE: {best_mae['mae']}")
```

**Opción B: Mejor balance**
```python
best = min(pareto['pareto_trials'], 
           key=lambda t: np.sqrt(t['mae']**2 + t['rmse']**2))
```

#### **Paso 5: Actualizar documentación** ✅

Si supera XGBoost, actualizar tabla en `docs/model_training.md`:

```markdown
| Modelo      | RMSE   | MAE   |
|-------------|--------|-------|
| XGBoost     | 1.00°C | 0.76°C |
| **LSTM Multiobj** | **1.25°C** | **0.72°C** |  ← 🏆 Ganador
```

---

## 📚 Documentación Completa

- **Guía detallada:** [docs/LSTM_MULTIOBJ_GUIDE.md](docs/LSTM_MULTIOBJ_GUIDE.md)
- **Código fuente:** [src/train.py](src/train.py) (función `train_lstm_multiobj`)
- **Script de análisis:** [scripts/analizar_pareto.py](scripts/analizar_pareto.py)

---

## 🎯 Próximos Pasos

### **Si MAE > 0.76°C (peor que XGBoost):**

1. **Aumentar trials:** `n_trials=50` en `train_lstm_multiobj()`
2. **Ajustar hiperparámetros:**
   - Expandir `lstm1_units`: `(64, 256)` en vez de `(32, 128)`
   - Reducir `learning_rate`: `(1e-5, 1e-3)`
3. **Probar Huber loss:** `use_mae_loss=False`
4. **Aumentar `window_size`:** de 48 a 72 horas

### **Si MAE ≤ 0.76°C (mejor que XGBoost):**

1. ✅ Evaluar en conjunto de **test final**
2. ✅ Actualizar tabla de comparación
3. ✅ Documentar en `CHANGELOG.md`
4. ✅ Guardar modelo para producción

---

## 🔧 Configuración Actual

```python
# En train_lstm_multiobj()
n_trials = 30              # Aumentar a 50-100 si necesitas más exploración
epochs = 500               # Máximo por trial (con early stopping)
patience = 40              # Paciencia del early stopping
use_mae_loss = True        # False para Huber loss
```

**Pruner:**
```python
MedianPruner(
    n_startup_trials=5,    # No podar primeros 5 trials
    n_warmup_steps=20,     # Esperar 20 epochs
    interval_steps=5       # Evaluar poda cada 5 epochs
)
```

---

## ✅ Verificación

**Archivos creados:**
- ✅ `src/train.py` (modificado con `train_lstm_multiobj()`)
- ✅ `docs/LSTM_MULTIOBJ_GUIDE.md` (guía completa)
- ✅ `scripts/analizar_pareto.py` (análisis automático)

**Para verificar instalación:**
```bash
# Verificar que el stage existe
python src/train.py --help

# Debe mostrar: choices=['separar_datos', 'lineales', 'xgboost', 'shap', 'lstm', 'lstm_multiobj']
```

---

¿Listo para entrenar? Ejecuta:

```bash
python src/train.py --stage lstm_multiobj
```

Tiempo estimado: **2-4 horas** (30 trials × 5-8 min/trial)

---

[Ver Guía Completa →](docs/LSTM_MULTIOBJ_GUIDE.md)
