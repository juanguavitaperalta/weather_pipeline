"""
Script para analizar resultados del Pareto Front de LSTM Multiobjetivo.

Uso:
    python scripts/analizar_pareto.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analizar_pareto():
    """Analiza y visualiza el Pareto Front del estudio LSTM multiobjetivo."""
    
    # Rutas
    pareto_path = "models/lstm_multiobj_final_pareto.json"
    metadata_path = "models/metadata/lstm_multiobj_metadatos.json"
    
    # Verificar archivos
    if not Path(pareto_path).exists():
        print(f"❌ Error: {pareto_path} no encontrado")
        print("Ejecuta primero: python src/train.py --stage lstm_multiobj")
        return
    
    # Cargar datos
    with open(pareto_path) as f:
        pareto_data = json.load(f)
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    pareto_trials = pareto_data['pareto_trials']
    
    # Estadísticas
    print("="*60)
    print("📊 ANÁLISIS DEL PARETO FRONT - LSTM MULTIOBJETIVO")
    print("="*60)
    print(f"\nNúmero de soluciones Pareto: {len(pareto_trials)}")
    print(f"Total de trials ejecutados: {metadata['optimizacion']['n_trials']}")
    print(f"Porcentaje en Pareto: {len(pareto_trials)/metadata['optimizacion']['n_trials']*100:.1f}%")
    
    # Mejor por MAE
    best_mae = min(pareto_trials, key=lambda t: t['mae'])
    print(f"\n🏆 MEJOR SOLUCIÓN POR MAE:")
    print(f"   Trial: {best_mae['trial_number']}")
    print(f"   MAE:   {best_mae['mae']:.4f}°C")
    print(f"   RMSE:  {best_mae['rmse']:.4f}°C")
    
    # Mejor por RMSE
    best_rmse = min(pareto_trials, key=lambda t: t['rmse'])
    print(f"\n🏆 MEJOR SOLUCIÓN POR RMSE:")
    print(f"   Trial: {best_rmse['trial_number']}")
    print(f"   MAE:   {best_rmse['mae']:.4f}°C")
    print(f"   RMSE:  {best_rmse['rmse']:.4f}°C")
    
    # Mejor balance (distancia euclidiana)
    best_balance = min(
        pareto_trials,
        key=lambda t: np.sqrt(t['mae']**2 + t['rmse']**2)
    )
    print(f"\n⚖️  MEJOR BALANCE (distancia euclidiana):")
    print(f"   Trial: {best_balance['trial_number']}")
    print(f"   MAE:   {best_balance['mae']:.4f}°C")
    print(f"   RMSE:  {best_balance['rmse']:.4f}°C")
    print(f"   Distancia: {np.sqrt(best_balance['mae']**2 + best_balance['rmse']**2):.4f}")
    
    # Comparación con otros modelos
    print(f"\n📈 COMPARACIÓN CON OTROS MODELOS:")
    comparisons = {
        "Ridge": {"mae": 1.19, "rmse": 1.51},
        "XGBoost": {"mae": 0.76, "rmse": 1.00},
        "LSTM v2": {"mae": 1.03, "rmse": 1.29}
    }
    
    for model, metrics in comparisons.items():
        print(f"\n   {model:12s} → MAE: {metrics['mae']:.2f}°C, RMSE: {metrics['rmse']:.2f}°C")
    
    print(f"\n   {'LSTM Multiobj':12s} → MAE: {best_mae['mae']:.2f}°C, RMSE: {best_mae['rmse']:.2f}°C")
    
    # Mejora vs XGBoost
    mejora_mae = ((comparisons['XGBoost']['mae'] - best_mae['mae']) / comparisons['XGBoost']['mae']) * 100
    mejora_rmse = ((comparisons['XGBoost']['rmse'] - best_mae['rmse']) / comparisons['XGBoost']['rmse']) * 100
    
    print(f"\n🎯 MEJORA VS XGBOOST:")
    if mejora_mae > 0:
        print(f"   MAE:  ✅ {mejora_mae:+.1f}% mejor")
    else:
        print(f"   MAE:  ❌ {mejora_mae:+.1f}% peor")
    
    if mejora_rmse > 0:
        print(f"   RMSE: ✅ {mejora_rmse:+.1f}% mejor")
    else:
        print(f"   RMSE: ❌ {mejora_rmse:+.1f}% peor")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIÓN:")
    if best_mae['mae'] < comparisons['XGBoost']['mae']:
        print("   ✅ LSTM Multiobj supera a XGBoost en MAE")
        print("   → Usar este modelo para producción")
        print("   → Actualizar tabla de comparación en docs/model_training.md")
    else:
        print("   ⚠️  LSTM Multiobj aún no supera a XGBoost en MAE")
        print("   → Sugerencias:")
        print("      • Aumentar n_trials (50-100)")
        print("      • Ajustar rangos de hiperparámetros")
        print("      • Probar Huber loss con delta=0.5")
        print("      • Aumentar window_size a 72h")
    
    # Guardar resumen
    summary_path = "reports/pareto_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RESUMEN PARETO FRONT - LSTM MULTIOBJETIVO\n")
        f.write("="*60 + "\n\n")
        f.write(f"Soluciones Pareto: {len(pareto_trials)}\n")
        f.write(f"Total trials: {metadata['optimizacion']['n_trials']}\n\n")
        f.write(f"Mejor MAE: {best_mae['mae']:.4f}°C (Trial {best_mae['trial_number']})\n")
        f.write(f"Mejor RMSE: {best_rmse['rmse']:.4f}°C (Trial {best_rmse['trial_number']})\n")
        f.write(f"Mejor balance: Trial {best_balance['trial_number']}\n\n")
        f.write(f"Mejora vs XGBoost (MAE): {mejora_mae:+.1f}%\n")
        f.write(f"Mejora vs XGBoost (RMSE): {mejora_rmse:+.1f}%\n")
    
    print(f"\n📄 Resumen guardado en: {summary_path}")
    print("="*60)


if __name__ == "__main__":
    analizar_pareto()
