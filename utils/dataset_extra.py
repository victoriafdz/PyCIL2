# Modulo para crear dos CSVs: base y extra, en Results/Gyro_Conversion
# El CSV base contiene solo las 8 columnas base replicadas hasta 1024 columnas y class_M como columna 1025
# El CSV extra contiene las 8 columnas base mas las columnas calculadas (low/high) replicadas hasta 1024 y class_M como columna 1025
import os
from typing import Optional, List

import pandas as pd

# Rutas y constantes por defecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
BASE_CSV_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801_export.csv')
EXTRA_CSV_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801_export_extra.csv')
TXT_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801.txt')
TARGET_FEATURES = 1024


# Funcion auxiliar: replicar exactamente la lista de 'features' hasta target_features columnas
def _replicate_features_exact(df: pd.DataFrame, features: List[str], target_features: int) -> pd.DataFrame:
    # Filtrar las features que realmente existen en el DataFrame
    existing = [f for f in features if f in df.columns]
    # Si no hay ninguna feature existente, devolver DataFrame vacio con mismo indice
    if len(existing) == 0:
        return pd.DataFrame(index=df.index)

    cols_order = []
    rep_num = 1
    # Repetir el conjunto de columnas en bloques hasta alcanzar target_features
    while len(cols_order) < target_features:
        for f in existing:
            if rep_num == 1:
                col_name = f
            else:
                col_name = f"{f}_rep{rep_num}"
            cols_order.append(col_name)
            if len(cols_order) >= target_features:
                break
        rep_num += 1

    # Construir un dict con los datos para cada columna replicada
    data = {}
    for name in cols_order:
        # obtener el nombre base (sin sufijo _repX)
        if '_rep' in name:
            base = name.rsplit('_rep', 1)[0]
        else:
            base = name
        # asignar la serie original a la columna replicada
        data[name] = df[base].values

    # Crear DataFrame con las columnas replicadas en el orden generado
    replicated_df = pd.DataFrame(data)
    return replicated_df


# Funcion que anade columnas calculadas low/high si existen columnas de error e<name>1/e<name>2
def add_calculated_e_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Obtener lista de columnas
    cols = list(df.columns)
    # Recorrer columnas buscando patrones e<name>1
    for c in cols:
        if not isinstance(c, str):
            continue
        if c.startswith('e') and c.endswith('1'):
            base_name = c[1:-1]
            e1 = f"e{base_name}1"
            e2 = f"e{base_name}2"
            # Si existen ambas columnas de error y la columna base, crear low/high calculadas
            if e1 in df.columns and e2 in df.columns and base_name in df.columns:
                low_name = f"{base_name}_low_calc"
                high_name = f"{base_name}_high_calc"
                try:
                    df[low_name] = df[base_name] - df[e1]
                    df[high_name] = df[base_name] + df[e2]
                except Exception:
                    df[low_name] = pd.NA
                    df[high_name] = pd.NA
    return df


# Funcion principal: genera ambos CSVs (base y extra) y devuelve ambos DataFrames
def get_dataset_extra(save_csv: bool = True, csv_path_extra: Optional[str] = None, num_clases: int = 5):
    # 1) leer datos desde el TXT original
    data = pd.read_csv(TXT_PATH, sep='\t', header=0)

    # 2) eliminar filas con valores nulos
    df_clean = data.dropna(axis=0).copy()

    # 3) aplicar filtro fisico si las columnas existen
    if {'class', 'M', 'Prot'}.issubset(df_clean.columns):
        df_clean = df_clean.loc[(df_clean['class'] == 'MS') & (df_clean['M'] < 2) & (df_clean['M'] > 0.7) & (df_clean['Prot'] < 50)].copy()

    # definir las 8 columnas base en orden
    base_features = [f for f in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age'] if f in df_clean.columns]

    # 7) preparar CSV extra: anadir columnas calculadas al dataset limpio
    df_with_calc = add_calculated_e_cols(df_clean.copy())
    # detectar las columnas calculadas generadas (sufijo _low_calc y _high_calc)
    calc_cols = [c for c in df_with_calc.columns if c.endswith('_low_calc') or c.endswith('_high_calc')]
    # features a replicar en extra = base_features + calc_cols (preservando orden)
    features_extra = base_features + [c for c in calc_cols if c not in base_features]

    # 8) replicar las features_extra hasta TARGET_FEATURES
    df_extra_repl = _replicate_features_exact(df_with_calc, features_extra, TARGET_FEATURES)

    # 9) calcular class_M para extra usando M en df_with_calc y anadir como ultima columna
    if 'M' in df_with_calc.columns:
        class_series_extra = pd.qcut(df_with_calc['M'], q=num_clases, labels=False)
        max_label_extra = int(class_series_extra.max())
        class_series_extra = max_label_extra - class_series_extra
        df_extra_repl['class_M'] = class_series_extra.values

    # 10) guardar CSV extra en la ruta solicitada (por defecto EXTRA_CSV_PATH)
    out_extra = csv_path_extra if csv_path_extra is not None else EXTRA_CSV_PATH
    if save_csv:
        os.makedirs(os.path.dirname(out_extra), exist_ok=True)
        df_extra_repl.to_csv(out_extra, index=False)

    # 11) devolver solo el DataFrame extra
    return df_extra_repl

# fin del modulo
