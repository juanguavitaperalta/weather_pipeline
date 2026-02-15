# Notas

Este archivo contiene notas relevantes sobre el proyecto weather_pipeline.

## Refactorización LSTM con Optuna (Febrero 2026)

### Cambios Implementados

**1. Nueva función `build_model`**
- Construye modelos LSTM de forma dinámica con hiperparámetros configurables
- Parámetros soportados:
  - `units`: Unidades en capas LSTM (32-128)
  - `dropout`: Tasa de dropout (0.1-0.5)
  - `learning_rate`: Tasa de aprendizaje (1e-4 a 1e-2)
  - `dense_units`: Unidades en capa densa (16-64)
  - `n_layers`: Número de capas LSTM (1-2)
  - `batch_size`: Tamaño del batch (16, 32, 64)
- Retorna: `(modelo_compilado, batch_size)`

**2. Refactorización de `train_lstm`**
- Ahora usa **Optuna** para búsqueda bayesiana de hiperparámetros
- Función `objective(trial)` interna que:
  - Muestrea hiperparámetros usando `trial.suggest_*`
  - Construye modelo con `build_model`
  - Entrena y evalúa en validación
  - Retorna `val_loss` para minimizar
- Configuración del estudio:
  ```python
  study = optuna.create_study(
      direction="minimize",
      pruner=optuna.pruners.MedianPruner()
  )
  study.optimize(objective, n_trials=30)
  ```

**3. Callbacks integrados**
- `EarlyStopping`: Detiene entrenamiento si no mejora val_loss
- `TFKerasPruningCallback`: Poda trials no prometedores de Optuna
- Restauración automática de los mejores pesos

**4. Guardado automático**
- **Mejor modelo**: `models/lstm_final.h5`
- **Mejores hiperparámetros**: `models/lstm_final_best_params.json`
  - Incluye: best_params, best_value, best_trial, n_trials, timestamp
- **Resumen del estudio**: `models/lstm_final_optuna_study.json`
  - Contiene todos los trials ejecutados con sus resultados
- **Base de datos Optuna**: `models/optuna_lstm.db` (SQLite)
  - Permite continuar optimización en futuras ejecuciones

**5. Metadatos actualizados**
- Versión actualizada a 2.0.0
- Nueva sección `optimizacion` con detalles de Optuna
- `best_params` almacenados en `arquitectura`

### Uso

Para entrenar el modelo LSTM con optimización:

```bash
python src/train.py --stage lstm
```

### Ventajas de la Nueva Implementación

✅ **Desacoplamiento**: Construcción del modelo separada del proceso de optimización  
✅ **Reproducibilidad**: Todos los trials y parámetros guardados automáticamente  
✅ **Eficiencia**: Pruning de Optuna descarta trials no prometedores temprano  
✅ **Flexibilidad**: Fácil agregar nuevos hiperparámetros a optimizar  
✅ **Trazabilidad**: Historial completo de optimización en base de datos SQLite  

### Próximos Pasos

- [ ] Analizar importancia de hiperparámetros con Optuna visualization
- [ ] Experimentar con diferentes espacios de búsqueda
- [ ] Comparar rendimiento vs modelo estático anterior
- [ ] Considerar optimización multi-objetivo (RMSE + tiempo de entrenamiento)

---

*Actualizado: 14 de febrero de 2026*
