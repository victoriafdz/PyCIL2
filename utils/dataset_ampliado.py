"""
dataset_ampliado.py
Script para ampliar el dataset Gyro a partir del TXT original.
- Replica las features hasta alcanzar exactamente 1024 columnas.
- Clasifica automáticamente en 5 clases equilibradas por masa (M).
- Ordena de mayor a menor masa e invierte etiquetas de clase.
- Guarda el CSV ampliado en la ruta indicada.
"""

import pandas as pd
import os
from typing import Optional

# ======= CONFIGURACIÓN =======
TXT_PATH = "/home/victoria/PycharmProjects/PyCIL2/data/gyro_tot_v20180801.txt"
CSV_PATH = "/home/victoria/PycharmProjects/PyCIL2/data/gyro_tot_v20180801_export.csv"
FEATURES = ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age']  # Features principales
TARGET_FEATURES = 1024  # Número total de columnas deseadas

def get_dataset(save_csv: bool = True, csv_path: Optional[str] = None):
    """
    Carga y limpia el dataset, replica features hasta alcanzar 1024 columnas,
    y clasifica automáticamente por masa (M) en 5 clases equilibradas.
    """

    # ======= 1. Leer TXT =======
    data = pd.read_csv(TXT_PATH, sep="\t", header=0)

    # Seleccionar columnas relevantes
    df = data[['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age', 'eAge1', 'eAge2', 'class']].copy()

    # Calcular límites de edad (solo informativos)
    df['low_age'] = df['Age'] - df['eAge1']
    df['high_age'] = df['Age'] + df['eAge2']

    # ======= 2. Limpiar valores nulos =======
    df = df.dropna(axis=0)

    # ======= 3. Filtrar según condiciones físicas de girocronología =======
    df = df.loc[(df['class'] == 'MS') & (df['M'] < 2) & (df['M'] > 0.7) & (df['Prot'] < 50)].copy()

    # ======= 4. Replicar features para alcanzar 1024 columnas =======
    # Número de réplicas necesarias
    replicas_needed = (TARGET_FEATURES - len(FEATURES)) // len(FEATURES)  # = 127
    replicated = []
    for i in range(replicas_needed):
        block = df[FEATURES].copy()
        block.columns = [f"{col}_rep{i+1}" for col in block.columns]
        replicated.append(block)

    # Concatenar todas las réplicas de golpe para evitar fragmentación
    df = pd.concat([df] + replicated, axis=1)

    # ======= 5. Ordenar por masa =======
    df = df.sort_values(by=['M'], ascending=False).reset_index(drop=True)

    # ======= 6. Clasificación en 5 clases equilibradas =======
    df['class_M'] = pd.qcut(df['M'], q=5, labels=False)

    # Invertir etiquetas: 0 = mayor masa, 4 = menor masa
    max_label = df['class_M'].max()
    df['class_M'] = max_label - df['class_M']

    # ======= 7. Guardar CSV =======
    if save_csv:
        if csv_path is None:
            csv_path = CSV_PATH
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        export_df = df.drop(columns=['low_age', 'high_age', 'eAge1', 'eAge2', 'class'], errors='ignore')
        export_df.to_csv(csv_path, index=False)
        print(f"Generado CSV con {len(export_df)} filas y {export_df.shape[1]} columnas en {csv_path}")

    return df

# ======= MAIN =======
if __name__ == "__main__":
    get_dataset()
