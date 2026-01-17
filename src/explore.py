import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from pandas.api.types import is_numeric_dtype
import os

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


def analisis_variables_numericas(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    ruta_salida = Path(output_dir)
    ruta_salida.mkdir(parents=True, exist_ok=True)
        
    datos_estadisticos = {}

    df.info()

    for col in df.select_dtypes(include=['number']).columns:
        n = len(df[col])
        media = df[col].mean()
        mediana = df[col].median()
        desviacion_estandar = df[col].std()
        skew = df[col].skew()
        kurtosis = df[col].kurtosis()
        vacios = df[col].isnull().sum()
        datos_estadisticos[col] = {
            'n': n,
            'media': media,
            'mediana': mediana,
            'desviacion_estandar': desviacion_estandar,
            'skewness': skew,
            'kurtosis': kurtosis,
            'missing_values': vacios
        }
        logger.info(f"Análisis de la columna {col}: Media={media}, Mediana={mediana},"
                    f"Desviación Estándar={desviacion_estandar}, Skewness={skew}, Kurtosis={kurtosis},"
                    f"Missing Values={vacios}, n={n}")

       # ====== FIGURA 1: histograma + tabla ======
        fig1 = plt.figure(figsize=(14, 5))
        gs1 = fig1.add_gridspec(nrows=1, ncols=2, width_ratios=[2.2, 1.2])

        ax_hist = fig1.add_subplot(gs1[0, 0])
        ax_tbl  = fig1.add_subplot(gs1[0, 1])

        # Histograma
        ax_hist.hist(df[col].dropna(), bins=30, alpha=0.7)
        ax_hist.set_title(f"Histograma de {col}")
        ax_hist.set_xlabel(col)
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.grid(alpha=0.3)

        # Tabla
        ax_tbl.axis("off")
        tabla = [
            ["missing values", datos_estadisticos[col]["missing_values"]],
            ["media", f"{media:.4f}"],
            ["mediana", f"{mediana:.4f}"],
            ["std", f"{desviacion_estandar:.4f}"],
            ["skewness", f"{skew:.4f}"],
            ["kurtosis", f"{kurtosis:.4f}"],
            ["n", f"{n:.4f}"],
        ]
        t = ax_tbl.table(
            cellText=tabla,
            colLabels=["Métrica", "Valor"],
            loc="center"
        )
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        t.scale(1, 1.4)
        ax_tbl.set_title("Resumen estadístico", pad=10)

        fig1.suptitle(f"Análisis numérico: {col}", fontsize=14)
        fig1.tight_layout()

        fig1.savefig(
            os.path.join(output_dir, f"analisis_{col}_hist_tabla.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig1)

        # ====== FIGURA 2: boxplot ======
        fig2, ax_box = plt.subplots(figsize=(6, 6))

        ax_box.boxplot(
            df[col].dropna(),
            vert=True,
            notch=True,
            patch_artist=True,
            showmeans=True
        )
        ax_box.set_title(f"Boxplot de {col}")
        ax_box.set_ylabel(col)
        ax_box.grid(alpha=0.3)

        fig2.tight_layout()

        fig2.savefig(
            os.path.join(output_dir, f"analisis_{col}_boxplot.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig2)


    # DataFrame resumen con todas las columnas
    return pd.DataFrame(datos_estadisticos).T

def main() -> None:
    ruta_entrada = "data/raw/open_meteo_hourly.csv"
    ruta_salida  = "reports/figures/exploratorios"

    logger.info("cargando archivos originales...")
    df = leer_archivo_csv(ruta_entrada)

    if df.empty:
        logger.error("DataFrame vacío. Abortando EDA.")
        return

    logger.info("obteniendo datos estadisticos")
    resumen = analisis_variables_numericas(df, ruta_salida)

    if not resumen.empty:
        logger.info("Análisis completado con éxito.")
    else:
        logger.warning("No se generaron estadísticas.")

if __name__ == "__main__":
    main()