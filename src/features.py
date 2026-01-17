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

def ordenar(df: pd.DataFrame, col_fecha: str) -> pd.DataFrame:

    if df.empty:
        logger.warning("El DataFrame está vacío. No hay datos para tratar.")
        return df
    if col_fecha not in df.columns:
        logger.error(f"La columna {col_fecha} no existe en el DataFrame.")
        return df
    
    if col_fecha == "time":
        df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
        df = df.dropna(subset=[col_fecha])

    df_ordenado = df.sort_values(by=col_fecha).reset_index(drop=True)
    return df_ordenado

def crear_target(df: pd.DataFrame, col_objetivo:str, horizon:int)->pd.DataFrame:
    df_copia = df.copy()
    df_copia[f"{col_objetivo}_target"] = df_copia[col_objetivo].shift(-horizon)
    return df_copia

def crear_columnas_lags(df:pd.DataFrame, col_independiente:str, lags:list[int])->pd.DataFrame:
    if col_independiente not in df.columns:
        logger.error(f"La columna {col_independiente} no existe en el DataFrame.")
        return df
    
    df_copia=df.copy()
    for i in lags:
        df_copia[f"{col_independiente}_lag_{i}"] = df_copia[col_independiente].shift(i)
    return df_copia

def preparacion_final(df:pd.DataFrame)->pd.DataFrame:
    df_final = df.dropna().reset_index(drop=True)
    return df_final

def guardar_archivo_csv(df: pd.DataFrame, ruta_salida: str) -> None:
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    logger.info(f"Archivo guardado en {ruta_salida}.")

def main() -> None:
    ruta_entrada = "data/processed/open_meteo_hourly_clean.csv"
    ruta_salida  = "data/features/features.csv"
    
    logger.warning("lectura de archivo.")
    df = leer_archivo_csv(ruta_entrada)
    logger.warning("ordenar dataframe.")
    df = ordenar(df, col_fecha="time")
    logger.warning("crear target con horizonte determinado de t = 3 horas.")
    df = crear_target(df, col_objetivo="temperature_2m", horizon=3)
    logger.warning("crear columnas lag.")
    df = crear_columnas_lags(df, col_independiente="temperature_2m", lags=[1,2,3,24])
    df = crear_columnas_lags(df, col_independiente="wind_speed_10m", lags=[1,2,6,12,24])
    df = crear_columnas_lags(df, col_independiente="relative_humidity_2m", lags=[1,2,36,12,24])
    logger.warning("preparacion final del dataframe.")
    df = preparacion_final(df)
    logger.warning("guardar archivo csv.")
    guardar_archivo_csv(df, ruta_salida)

if __name__ == "__main__":
    main()