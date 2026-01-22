import pandas as pd
import os
from typing import Optional

"""
dataset_ampliado.py
Script para ampliar el dataset Gyro a partir del TXT original.
- Replica las 8 columnas base hasta alcanzar 1024 columnas exactas.
- Clasifica automaticamente en num_clases por masa (M).
- Guarda el CSV ampliado en Results/Gyro_Conversion/gyro_tot_v20180801_export.csv por defecto.
"""

# ======= CONFIGURACION =======
TXT_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'gyro_tot_v20180801.txt')
DATA_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801_export.csv')
TARGET_FEATURES = 1024  # Numero total de columnas deseadas


def _replicate_features_exact(df: pd.DataFrame, features: list, target_features: int) -> pd.DataFrame:
    # Funcion que replica exactamente las columnas de la lista 'features' hasta target_features columnas
    existing = [f for f in features if f in df.columns]
    if len(existing) == 0:
        return pd.DataFrame(index=df.index)

    cols_order = []
    rep_num = 1
    while len(cols_order) < target_features:
        for f in existing:
            col_name = f if rep_num == 1 else f"{f}_rep{rep_num}"
            cols_order.append(col_name)
            if len(cols_order) >= target_features:
                break
        rep_num += 1

    data = {}
    for name in cols_order:
        base = name.rsplit('_rep', 1)[0] if '_rep' in name else name
        data[name] = df[base].values
    return pd.DataFrame(data)


def get_dataset(save_csv: bool = True, csv_path: Optional[str] = None, num_clases: int = 5):
    # Leer el TXT
    data = pd.read_csv(TXT_PATH, sep='\t', header=0)

    # Limpiar nulos
    df = data.dropna(axis=0).copy()

    # Filtrar condiciones fisicas si las columnas existen
    if {'class', 'M', 'Prot'}.issubset(df.columns):
        df = df.loc[(df['class'] == 'MS') & (df['M'] < 2) & (df['M'] > 0.7) & (df['Prot'] < 50)].copy()

    # Definir las 8 columnas base en orden preferente
    base_features = [f for f in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age'] if f in df.columns]

    # Replica solo las columnas base hasta TARGET_FEATURES
    df_repl = _replicate_features_exact(df, base_features, TARGET_FEATURES)

    # Calcular class_M a partir de la columna M original y anadir como ultima columna
    if 'M' in df.columns:
        class_series = pd.qcut(df['M'], q=num_clases, labels=False)
        max_label = int(class_series.max())
        class_series = max_label - class_series
        df_repl['class_M'] = class_series.values

    # Guardar CSV base en la carpeta Results/Gyro_Conversion
    if save_csv:
        out_path = csv_path if csv_path is not None else CSV_PATH
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_repl.to_csv(out_path, index=False)
        print(f"Generado CSV base (sin e calculados) en {out_path} con {df_repl.shape[0]} filas y {df_repl.shape[1]} columnas")

    return df_repl


# Mantener ejecucion directa posible
if __name__ == '__main__':
    get_dataset()