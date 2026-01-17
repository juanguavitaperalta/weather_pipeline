import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,  plot_pacf
import pandas as pd
import logging
from pathlib import Path
from pandas.api.types import is_numeric_dtype
from statsmodels.tsa.stattools import adfuller, kpss, ccf
from arch.unitroot import PhillipsPerron
import os
import numpy as np



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def leer_archivo_csv(ruta_archivo: str) -> pd.DataFrame:
    try: 
        df = pd.read_csv(ruta_archivo)
        logger.info(f"El archivo {ruta_archivo} fue leído correctamente")
        print(df.head())
        print(df.isna().sum())
        # df.dtypes para temas de tipos de datos
    except FileNotFoundError:
        logger.error(f"El archivo {ruta_archivo} no fue encontrado.")
        return pd.DataFrame()  
    return df

def graficar_series(df: pd.DataFrame, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            fig1 = plt.figure(figsize=(14, 5))
            plt.plot(df.index, df[col], label=col)
            plt.title(f"Serie temporal: {col}")
            plt.xlabel("Tiempo")
            plt.ylabel("Valor")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            ruta_guardado = os.path.join(output_dir, f"serie_{col}.png")
            fig1.savefig(ruta_guardado,dpi=200,bbox_inches="tight")
            logger.info(f"Gráfica de la serie temporal {col} guardada en {ruta_guardado}.")
            plt.close(fig1)

def analisis_estacionariedad(df: pd.DataFrame, threshold: float) -> dict[str, dict]:
    """Realiza tests de estacionariedad ADF, KPSS y Phillips-Perron."""
    import warnings
    df_results = {}
    
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            adf_result = adfuller(df[col], regression='c', autolag='AIC')
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(df[col], regression='c', nlags="auto")
            
            pp_result = PhillipsPerron(df[col])

            df_results[col] = {
                "ADF Statistic": adf_result[0],
                "ADF p-value": adf_result[1],
                "ADF Conclusion": "Estacionaria" if adf_result[1] < threshold else "No estacionaria",
                "KPSS Statistic": kpss_result[0],
                "KPSS p-value": kpss_result[1],
                "KPSS Conclusion": "No estacionaria" if kpss_result[1] < threshold else "Estacionaria",
                "Phillips-Perron Statistic": pp_result.stat,
                "Phillips-Perron p-value": pp_result.pvalue,
                "Phillips-Perron Conclusion": "Estacionaria" if pp_result.pvalue < threshold else "No estacionaria"
            }
    return df_results

def graficas_estacionariedad(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for col in df.columns:
        if col == "temperature_2m":
            logger.info(f"n {col} es {len(df[col])}.")
            fig, axs = plt.subplots(2, 1, figsize=(14, 12))
            fig.suptitle(f"Análisis de estacionariedad: {col}", fontsize=16)

            # ADF
            plot_acf(df[col], ax=axs[0],lags=40)
            axs[0].set_title(f"ACF - {col}")

            # KPSS
            plot_pacf(df[col], ax=axs[1], lags=40)
            axs[1].set_title(f"PACF - {col}")

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            ruta_guardado = os.path.join(output_dir, f"acf & pacf_{col}.png")
            fig.savefig(ruta_guardado, dpi=200, bbox_inches="tight")
            logger.info(f"Gráfica de estacionariedad para {col} guardada en {ruta_guardado}.")
            plt.close(fig)

def cross_corr(df: pd.DataFrame, col1_objective: str, col2: str, lags: int, horizon: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    df_subset = df[[col1_objective, col2]].dropna()
    y = df_subset[col1_objective]
    y=y.shift(-horizon)
    
    x = df_subset[col2]

    y = y - y.mean()
    x = x - x.mean()

    
    logger.info(f"Calculando cross-correlation entre {col1_objective} y {col2} con {len(x)} puntos alineados.")
    corrs = []
    ns = []

    for k in range(lags + 1):
        xlag = df_subset[col2].shift(k)
        aligned = pd.concat([xlag, y], axis=1).dropna()
        n = len(aligned)
        ns.append(n)
        if n < 10:
            corrs.append(np.nan)
        else:
            corrs.append(aligned.iloc[:,0].corr(aligned.iloc[:,1]))

    out = pd.DataFrame(
        {
            'lag': range(lags + 1), 
            'correlation': corrs, 
            'n': ns
        }
    )

    n_eff = int(np.nanmin(out["n"]))
    conf = 1.96 / np.sqrt(n_eff) if n_eff > 0 else np.nan

    logger.info(f"Graficando cross-correlation entre {col1_objective} y {col2}.")
    plt.figure(figsize=(10, 5))
    plt.stem(out['lag'], out['correlation'])
    plt.axhline(0, color='black', lw=0.5)

    plt.axhline(conf, color='red', linestyle='--', lw=0.8)
    plt.axhline(-conf, color='red', linestyle='--', lw=0.8)

    plt.title(f"Cross-Correlation: {col1_objective} vs {col2}")
    plt.xlabel("Lags")
    plt.ylabel("Correlation")
    plt.grid()
    plt.tight_layout()
    ruta_guardado = os.path.join(output_dir, f"cross_corr_{col1_objective}_{col2}.png")
    plt.savefig(ruta_guardado, dpi=200, bbox_inches="tight")
    logger.info(f"Gráfica de correlación cruzada para {col1_objective} y {col2} guardada en {ruta_guardado}.")
    plt.close()

 


def main() -> None:
    ruta_entrada = "data/processed/open_meteo_hourly_clean.csv"
    ruta_imagenes = "reports/figures/analisis_temporal/series"
    ruta_acf = "reports/figures/analisis_temporal/acf"
    ruta_cross = "reports/figures/analisis_temporal/crosscorr"
    df_resultados = {}
    logger.info("cargando archivos originales...")
    df = leer_archivo_csv(ruta_entrada)

    if df.empty:
        logger.error("El DataFrame está vacío. No se puede proceder con la limpieza de datos.")
        return
    
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    graficar_series(df,ruta_imagenes)
    logger.info(f"ejecutando graficas estacionariedad")
    graficas_estacionariedad(df, ruta_acf)
    logger.info(f"ejecutando analisis de correlación cruzada")
    cross_corr(df, 'temperature_2m', 'relative_humidity_2m', lags=100, horizon = 3, output_dir=ruta_cross)
    cross_corr(df, 'temperature_2m', 'wind_speed_10m', lags=100, horizon = 3, output_dir=ruta_cross)

    df_resultados = analisis_estacionariedad(df,threshold=0.05)

    for col, resultados in df_resultados.items():
        logger.info(f"Resultados para la columna '{col}':")
        for test, result in resultados.items():
            logger.info(f"  {test}: {result}")


if __name__ == "__main__":
    main()