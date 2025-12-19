import pandas as pd
import os
from typing import Optional

def get_dataset(save_csv: bool = True, csv_path: Optional[str] = None, num_clases: int = 5) -> pd.DataFrame:
    """Load and clean the star dataset, classify automatically by stellar mass (M) into 5 bins.

    Args:
        save_csv: if True, save the cleaned DataFrame to `csv_path` (or a default path).
        csv_path: destination CSV path. If None and `save_csv` is True, a default file
            next to the data file will be used.

    Returns:
        pandas.DataFrame: the cleaned and filtered dataset with class_M column.
    """
    # locate the input data file relative to this module
    data_file = os.path.join(os.path.dirname(__file__), 'gyro_tot_v20180801.txt')
    data = pd.read_csv(data_file, sep="\t", header=0)

    df = data[['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age', 'eAge1', 'eAge2', 'class']].copy()
    features = ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age']  # Define las 8 columnas de interés.

    # age limits, only for graphics
    df['low_age'] = df['Age'] - df['eAge1']
    df['high_age'] = df['Age'] + df['eAge2']

    # clean NA values
    df = df.dropna(axis=0)

    # filter the datasets because of the physics behind gyrochronology
    df = df.loc[(df['class'] == 'MS') & (df['M'] < 2) & (df['M'] > 0.7) & (df['Prot'] < 50)].copy()

    #replicar las 8 features 120 veces → 1016 columnas
    replicated = []  # Inicializa una lista para almacenar bloques replicados de columnas.
    for i in range(120):  # Itera 120 veces para crear 128 réplicas de las 8 features.
        block = df[features].copy()  # Copia las 8 columnas originales en un bloque temporal.
        block.columns = [f"{col}_rep{i + 1}" for col in block.columns]  # Renombra las columnas añadiendo sufijo repX.
        replicated.append(block)  # Añade el bloque renombrado a la lista de réplicas.

    df = pd.concat([df]+replicated, axis=1)  # Concatena todos los bloques replicados horizontalmente.

    # Ordenar el dataframe por masa (M), de mayor a menor
    df = df.sort_values(by=['M'], ascending=False)

    # Clasificación automática en función de M en 5 clases equilibradas
    df['class_M'] = pd.qcut(df['M'], q=num_clases , labels=False)

    # Invertimos el orden para que 0 = mayor masa y 4 = menor masa
    max_label = df['class_M'].max()
    df['class_M'] = max_label - df['class_M']

    #guardar a CSV
    if save_csv:
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), 'gyro_tot_v20180801_export.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        export_df = df.drop(columns=['low_age', 'high_age', 'eAge1', 'eAge2', 'class'], errors='ignore')
        export_df.to_csv(csv_path, index=False)

    return df

# Ejecutar
get_dataset()