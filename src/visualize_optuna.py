"""
Script para visualizar y analizar los resultados de optimización Optuna del modelo LSTM.

Uso:
    python src/visualize_optuna.py

Requisitos:
    pip install optuna plotly kaleido
"""

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)
import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_study(db_path: str = "models/optuna_lstm.db", study_name: str = "lstm_optimization"):
    """Carga el estudio de Optuna desde la base de datos SQLite."""
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}"
        )
        logger.info(f"Estudio cargado exitosamente: {len(study.trials)} trials encontrados")
        return study
    except KeyError:
        logger.error(f"No se encontró el estudio '{study_name}' en {db_path}")
        logger.info("Estudios disponibles:")
        storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
        for name in storage.get_all_study_summaries():
            logger.info(f"  - {name.study_name}")
        return None
    except Exception as e:
        logger.error(f"Error al cargar el estudio: {e}")
        return None


def print_study_summary(study):
    """Imprime un resumen del estudio de optimización."""
    logger.info("="*70)
    logger.info("RESUMEN DEL ESTUDIO DE OPTIMIZACIÓN")
    logger.info("="*70)
    
    logger.info(f"\n📊 Estadísticas Generales:")
    logger.info(f"  Total de trials: {len(study.trials)}")
    logger.info(f"  Trials completados: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    logger.info(f"  Trials podados: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    logger.info(f"  Trials fallidos: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    logger.info(f"\n🏆 Mejor Trial:")
    logger.info(f"  Trial número: {study.best_trial.number}")
    logger.info(f"  Mejor valor (val_loss): {study.best_value:.6f}")
    
    logger.info(f"\n⚙️  Mejores Hiperparámetros:")
    for param, value in study.best_params.items():
        logger.info(f"  {param}: {value}")
    
    # Top 5 trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)[:5]
    
    logger.info(f"\n🥇 Top 5 Trials:")
    for i, trial in enumerate(sorted_trials, 1):
        logger.info(f"\n  {i}. Trial #{trial.number} - val_loss: {trial.value:.6f}")
        for param, value in trial.params.items():
            logger.info(f"     {param}: {value}")


def create_visualizations(study, output_dir: str = "reports/figures/optuna"):
    """Crea visualizaciones de Optuna y las guarda."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n📈 Generando visualizaciones en {output_dir}...")
    
    try:
        # 1. Historia de optimización
        fig = plot_optimization_history(study)
        fig.write_html(str(output_path / "optimization_history.html"))
        logger.info("  ✓ Historia de optimización guardada")
        
        # 2. Importancia de parámetros
        fig = plot_param_importances(study)
        fig.write_html(str(output_path / "param_importances.html"))
        logger.info("  ✓ Importancia de parámetros guardada")
        
        # 3. Coordenadas paralelas
        fig = plot_parallel_coordinate(study)
        fig.write_html(str(output_path / "parallel_coordinate.html"))
        logger.info("  ✓ Coordenadas paralelas guardadas")
        
        # 4. Slice plots
        fig = plot_slice(study)
        fig.write_html(str(output_path / "slice_plot.html"))
        logger.info("  ✓ Slice plots guardados")
        
        # 5. Contour plots (para pares de parámetros importantes)
        params = list(study.best_params.keys())
        if len(params) >= 2:
            # Graficar las dos primeras combinaciones más importantes
            fig = plot_contour(study, params=[params[0], params[1]])
            fig.write_html(str(output_path / f"contour_{params[0]}_{params[1]}.html"))
            logger.info(f"  ✓ Contour plot para {params[0]} vs {params[1]} guardado")
        
        logger.info(f"\n✅ Todas las visualizaciones guardadas en {output_dir}/")
        
    except Exception as e:
        logger.error(f"Error al crear visualizaciones: {e}")


def create_trials_dataframe(study):
    """Crea un DataFrame con información de todos los trials."""
    trials_data = []
    
    for trial in study.trials:
        trial_info = {
            'trial_number': trial.number,
            'value': trial.value,
            'state': trial.state.name,
            'duration': trial.duration.total_seconds() if trial.duration else None,
        }
        # Agregar parámetros
        trial_info.update(trial.params)
        trials_data.append(trial_info)
    
    df = pd.DataFrame(trials_data)
    return df


def analyze_trials(study, output_path: str = "reports/figures/optuna/trials_analysis.csv"):
    """Analiza y guarda información detallada de los trials."""
    df = create_trials_dataframe(study)
    
    # Filtrar solo trials completados
    df_complete = df[df['state'] == 'COMPLETE'].copy()
    
    if len(df_complete) > 0:
        # Guardar CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_complete.to_csv(output_path, index=False)
        logger.info(f"\n💾 Análisis de trials guardado en {output_path}")
        
        # Mostrar estadísticas
        logger.info("\n📊 Estadísticas de Hiperparámetros (trials completados):")
        param_cols = [col for col in df_complete.columns if col not in ['trial_number', 'value', 'state', 'duration']]
        
        for param in param_cols:
            logger.info(f"\n  {param}:")
            if df_complete[param].dtype in ['int64', 'float64']:
                logger.info(f"    Media: {df_complete[param].mean():.4f}")
                logger.info(f"    Std: {df_complete[param].std():.4f}")
                logger.info(f"    Min: {df_complete[param].min()}")
                logger.info(f"    Max: {df_complete[param].max()}")
            else:
                logger.info(f"    Valores: {df_complete[param].value_counts().to_dict()}")
        
        # Correlación entre parámetros y valor objetivo
        logger.info("\n🔗 Correlación con val_loss:")
        numeric_params = df_complete[param_cols].select_dtypes(include=['int64', 'float64']).columns
        for param in numeric_params:
            corr = df_complete[[param, 'value']].corr().iloc[0, 1]
            logger.info(f"  {param}: {corr:.4f}")
    else:
        logger.warning("No hay trials completados para analizar")


def main():
    """Función principal del script."""
    logger.info("="*70)
    logger.info("VISUALIZACIÓN Y ANÁLISIS DE RESULTADOS OPTUNA")
    logger.info("="*70)
    
    # Cargar estudio
    study = load_study()
    
    if study is None:
        logger.error("No se pudo cargar el estudio. Asegúrate de haber entrenado el modelo LSTM primero.")
        return
    
    # Imprimir resumen
    print_study_summary(study)
    
    # Crear visualizaciones
    create_visualizations(study)
    
    # Analizar trials
    analyze_trials(study)
    
    logger.info("\n" + "="*70)
    logger.info("ANÁLISIS COMPLETADO")
    logger.info("="*70)
    logger.info("\n📁 Archivos generados:")
    logger.info("  - reports/figures/optuna/optimization_history.html")
    logger.info("  - reports/figures/optuna/param_importances.html")
    logger.info("  - reports/figures/optuna/parallel_coordinate.html")
    logger.info("  - reports/figures/optuna/slice_plot.html")
    logger.info("  - reports/figures/optuna/trials_analysis.csv")
    logger.info("\n💡 Abre los archivos HTML en tu navegador para visualización interactiva")


if __name__ == "__main__":
    main()
