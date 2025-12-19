

import pandas as pd
import os
from typing import Optional

"""
dataset_ampliado.py
Script para ampliar el dataset Gyro a partir del TXT original.
- Usa TODAS las columnas del TXT original.
- Replica las features hasta alcanzar exactamente 1024 columnas.
- Clasifica automáticamente en 5 clases equilibradas por masa (M).
- Ordena de mayor a menor masa e invierte etiquetas de clase.
- Guarda el CSV ampliado en la ruta indicada.
"""


# ======= CONFIGURACIÓN =======
TXT_PATH = "/home/victoria/PycharmProjects/PyCIL2/data/gyro_tot_v20180801.txt"
CSV_PATH = "/home/victoria/PycharmProjects/PyCIL2/data/gyro_tot_v20180801_export.csv"
TARGET_FEATURES = 1024  # Número total de columnas deseadas

def get_dataset(save_csv: bool = True, csv_path: Optional[str] = None, num_clases: int = 5):
    """
    Carga y limpia el dataset, replica TODAS las columnas hasta alcanzar 1024 columnas,
    y clasifica automáticamente por masa (M) en 5 clases equilibradas.
    """

    # ======= 1. Leer TXT =======
    data = pd.read_csv(TXT_PATH, sep="\t", header=0)

    # ======= 2. Limpiar valores nulos =======
    df = data.dropna(axis=0).copy()

    # ======= 3. Filtrar según condiciones físicas de girocronología =======
    # Solo si las columnas existen
    if {'class', 'M', 'Prot'}.issubset(df.columns):
        df = df.loc[
            (df['class'] == 'MS') &
            (df['M'] < 2) &
            (df['M'] > 0.7) &
            (df['Prot'] < 50)
        ].copy()

    # ======= 4. Seleccionar TODAS las columnas numéricas =======
    # (evita replicar columnas categóricas como 'class')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Si no hay suficientes columnas, replicamos
    current_features = len(numeric_cols)
    replicas_needed = (TARGET_FEATURES - current_features) // current_features

    replicated_blocks = []
    for i in range(replicas_needed):
        block = df[numeric_cols].copy()
        block.columns = [f"{col}_rep{i+1}" for col in block.columns]
        replicated_blocks.append(block)

    # Concatenartodo
    df = pd.concat([df] + replicated_blocks, axis=1)

    # ======= 5. Ordenar por masa =======
    if 'M' in df.columns:
        df = df.sort_values(by=['M'], ascending=False).reset_index(drop=True)

    # ======= 6. Clasificación en 5 clases equilibradas =======
    if 'M' in df.columns:
        df['class_M'] = pd.qcut(df['M'], q=num_clases, labels=False)
        max_label = df['class_M'].max()
        df['class_M'] = max_label - df['class_M']

    # ======= 7. Guardar CSV =======
    if save_csv:
        if csv_path is None:
            csv_path = CSV_PATH
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Exportamos el DataFrame completo
        export_df = df.copy()
        # Eliminar SIMBAD_ID si existe
        export_df = export_df.drop(columns=['SIMBAD_ID','class','type','cat1','cat2','mode'], errors='ignore')
        export_df.to_csv(csv_path, index=False)

        print(f"Generado CSV con {len(export_df)} filas y {export_df.shape[1]} columnas en {csv_path}")

    return df

# ======= MAIN =======
if __name__ == "__main__":
    get_dataset()