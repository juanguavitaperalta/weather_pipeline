import pandas as pd
import numpy as np
import logging
from pathlib import Path
from pandas.api.types import is_numeric_dtype


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("weather_pipeline")

def leer_archivo_csv(ruta_archivo: str) -> pd.DataFrame:
    try: 
        df = pd.read_csv(ruta_archivo)
        logger.info(f"El archivo {ruta_archivo} fue leído correctamente")
    except FileNotFoundError:
        logger.error(f"El archivo {ruta_archivo} no fue encontrado.")
        return pd.DataFrame()  
    return df


def tratamiento_datos_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    
    df2= df.copy()
    
    cols = list(df2.columns)
    logger.info("Tratando datos faltantes...")

    if df2.empty:
        logger.warning("El DataFrame está vacío. No hay datos para tratar.")
        return df2

    for col in cols:
        if col == "time":
            df2[col] = pd.to_datetime(df2[col], errors="coerce")

            df2 = (
                df2
                .dropna(subset=[col])
                .sort_values(by=col)
                .drop_duplicates(subset=[col], keep="first")
            )
            logger.info("Columna %s: convertida,ordenada y sin duplicados.", col)
            continue
        """""
        if is_numeric_dtype(df2[col]):
            df2[col] = df2[col].fillna(df2[col].mean())
            logger.info(f"Columna {col} (numérica): datos faltantes llenados con la media.")
        """
    return df2.reset_index(drop=True)

def guardar_archivo_csv(df: pd.DataFrame, ruta_salida: str) -> None:
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    logger.info(f"Archivo guardado en {ruta_salida}.")

def main() -> None:
    ruta_entrada = "data/raw/open_meteo_hourly.csv"
    ruta_salida  = "data/processed/open_meteo_hourly_clean.csv"

    logger.info("cargando archivos originales...")
    df = leer_archivo_csv(ruta_entrada)

    if df.empty:
        logger.error("El DataFrame está vacío. No se puede proceder con la limpieza de datos.")
        return

    logger.info("tratando datos faltantes...")
    df_limpio = tratamiento_datos_faltantes(df)

    logger.info("guardando archivo limpio...")
    guardar_archivo_csv(df_limpio, ruta_salida)

    logger.info(f"Raw data saved to {ruta_salida}")


if __name__ == "__main__":
    main()