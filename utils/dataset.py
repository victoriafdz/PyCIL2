# Modulo para generar el CSV base con solo las 8 columnas base y la columna class_M
import os
from typing import Optional

import pandas as pd

# Rutas por defecto: TXT de entrada y carpeta de salida Results/Gyro_Conversion
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TXT_PATH = os.path.join(PROJECT_ROOT, 'data', 'gyro_tot_v20180801.txt')
GYRO_DIR = os.path.join(PROJECT_ROOT, 'Results', 'Gyro_Conversion')
os.makedirs(GYRO_DIR, exist_ok=True)
BASE_CSV_PATH = os.path.join(GYRO_DIR, 'gyro_tot_v20180801_export.csv')


# Funcion publica que genera el CSV base con las 8 columnas y class_M
def get_dataset(save_csv: bool = True, csv_path: Optional[str] = None, num_clases: int = 5) -> pd.DataFrame:
    # Leer el fichero TXT original con separador de tabulaciones
    data = pd.read_csv(TXT_PATH, sep='\t', header=0)

    # Eliminar filas con valores nulos para evitar inconsistencias
    df = data.dropna(axis=0).copy()

    # Aplicar filtro fisico si las columnas necesarias existen
    if {'class', 'M', 'Prot'}.issubset(df.columns):
        # Mantener solo clase 'MS' y valores de M y Prot dentro de rangos razonables
        df = df.loc[(df['class'] == 'MS') & (df['M'] < 2) & (df['M'] > 0.7) & (df['Prot'] < 50)].copy()

    # Definir las 8 columnas base en el orden deseado y filtrar solo esas columnas si existen
    base_cols = [c for c in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age'] if c in df.columns]
    df_base = df[base_cols].copy()

    # Eliminar filas con NA en las columnas base
    if base_cols:
        df_base = df_base.dropna(subset=base_cols).reset_index(drop=True)

    # Crear la columna class_M basada en cuantiles de M SI la columna M existe
    if 'M' in df_base.columns:
        # Calcular labels 0..num_clases-1 y luego invertir el orden para que mayor M tenga label mayor
        df_base['class_M'] = pd.qcut(df_base['M'], q=num_clases, labels=False)
        max_label = df_base['class_M'].max()
        df_base['class_M'] = max_label - df_base['class_M']

    # Guardar CSV si se solicita
    if save_csv:
        out_path = csv_path if csv_path is not None else BASE_CSV_PATH
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Exportar solo las columnas base y class_M
        export_df = df_base.copy()
        export_df.to_csv(out_path, index=False)

    # Devolver el DataFrame con las 8 columnas base y class_M
    return df_base

# Fin del modulo dataset
