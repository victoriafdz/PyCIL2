# Nuevo modulo dataset_moredata.py
# Este modulo genera un CSV con filas ampliadas a partir del TXT original
# Se basa en las estructuras de dataset, dataset_extra y el ejemplo de aumento por incertidumbres
# Comentarios en espanol sin tildes ni caracteres especiales

import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
# importar el modulo que replica features a 1024 columnas
from utils import dataset_ampliado as ds_ampliado

# Definir rutas y constantes coherentes con el resto del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TXT_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801.txt')
# Guardar el CSV en data (no en Results) segun lo pedido
OUT_CSV_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801_export_moredata.csv')

# Numero objetivo de columnas para replicacion (si se necesitase en el futuro)
TARGET_FEATURES = 1024

# -----------------------------------------------------------------------------
# Funcion auxiliar: busca nombres de columnas de error asociadas a una columna base
# Ejemplos tipicos: eTeff1, eTeff2 ; Teffe1, Teffe2 ; Teff_e1, Teff_e2
# Devuelve una tupla (err_low_col, err_high_col) o None si no encuentra
# -----------------------------------------------------------------------------
def _find_error_cols(df: pd.DataFrame, col: str) -> Optional[Tuple[str, str]]:
    # lista de patrones a comprobar en el DataFrame
    candidates = [
        (f'e{col}1', f'e{col}2'),
        (f'{col}e1', f'{col}e2'),
        (f'{col}_e1', f'{col}_e2'),
        (f'{col}_err1', f'{col}_err2'),
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    return None

# -----------------------------------------------------------------------------
# Funcion que anade columnas calculadas a partir de columnas de error
# Para cada pareja e<name>1/e<name>2 y si existe la columna base, se anaden
# <name>_low_calc = base - e1 ; <name>_high_calc = base + e2
# -----------------------------------------------------------------------------
def add_calculated_e_cols(df: pd.DataFrame) -> pd.DataFrame:
    # copiar columnas actuales en lista para iterar sin modificar mientras iteramos
    cols = list(df.columns)
    for c in cols:
        # solo procesar nombres tipo string
        if not isinstance(c, str):
            continue
        # detectar patrones e<name>1
        if c.startswith('e') and c.endswith('1'):
            base_name = c[1:-1]
            e1 = f'e{base_name}1'
            e2 = f'e{base_name}2'
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

# -----------------------------------------------------------------------------
# Funcion de muestreo por incertidumbres (version para arrays numpy)
# X_input debe contener para cada feature 3 valores: valor, err_low, err_high
# y_input debe contener para target: valor, err_low, err_high
# Devuelve X_aug (solo valores) y y_aug (vector)
# -----------------------------------------------------------------------------
def data_augmentation_with_uncertainties_np(X_input: np.ndarray, y_input: np.ndarray, n_samples: int):
    # extraer solo valores (cada 3 columnas)
    X = X_input[:, 0::3]
    # dimensiones y errores
    m, n = X_input.shape
    if n == 0:
        return np.empty((0, 0)), np.empty((0,))
    num_features = int(n // 3)
    Xe = np.empty((m, num_features * 2), float)
    jj = 0
    kk = 0
    for ii in range(num_features):
        Xe[:, kk] = X_input[:, 1 + jj]
        kk += 1
        Xe[:, kk] = X_input[:, 2 + jj]
        kk += 1
        jj += 3
    # target y errores
    y = y_input[:, 0::3]
    ye = y_input[:, 1::]
    # si n_samples es 0 devolver datos originales
    if n_samples == 0:
        return X, np.ravel(y)
    # generar muestras con semilla fija para reproducibilidad
    from numpy.random import default_rng
    rng = default_rng(1)
    X_aug_list = []
    y_aug_list = []
    # iterar por cada fila
    for (s_x, s_xe, s_y, s_ye) in zip(X, Xe, y, ye):
        # anadir fila original
        X_aug_list.append(s_x.copy())
        # asegurar que guardamos un float y no una array (evitar ragged arrays)
        try:
            y_aug_list.append(float(s_y))
        except Exception:
            # en caso de que s_y sea un array unidimensional
            y_aug_list.append(float(np.ravel(s_y)[0]))
        # generar n_samples muestras uniformes por cada caracteristica
        # y para el target
        y_low = s_y - s_ye[0]
        y_high = s_y + s_ye[1]
        y_new = rng.uniform(y_low, y_high, size=(n_samples,))
        for ns in range(n_samples):
            x_new = np.empty((num_features,), float)
            ee = 0
            for ff in range(num_features):
                v = s_x[ff]
                e_low = s_xe[ee + 0]
                e_high = s_xe[ee + 1]
                # generar muestra uniforme dentro del intervalo
                x_new[ff] = float(rng.uniform(v - e_low, v + e_high))
                ee += 2
            X_aug_list.append(x_new)
            y_aug_list.append(y_new[ns])
    X_aug = np.vstack(X_aug_list)
    y_aug = np.array(y_aug_list)
    return X_aug, y_aug

# -----------------------------------------------------------------------------
# Funcion principal: genera el CSV con filas ampliadas y lo guarda
# params: save_csv True/False, csv_path ruta de salida opcional,
# n_samples_per_row numero de muestras a generar por cada fila original,
# num_clases para crear class_M
# -----------------------------------------------------------------------------
def get_dataset_moredata(save_csv: bool = True, csv_path: Optional[str] = None, n_samples_per_row: int = 10, num_clases: int = 2) -> pd.DataFrame:
    # leer el TXT original con pandas
    data = pd.read_csv(TXT_PATH, sep='\t', header=0)
    # copia para manipular
    df = data.copy()
    # aplicar filtro fisico si existen las columnas necesarias
    if {'class', 'M', 'Prot'}.issubset(df.columns):
        df = df.loc[(df['class'] == 'MS') & (df['M'] < 2) & (df['M'] > 0.7) & (df['Prot'] < 50)].copy()
    # definir columnas base en orden
    base_cols = [c for c in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age'] if c in df.columns]
    # eliminar filas que tengan NA en las columnas base
    if base_cols:
        df_base = df.dropna(subset=base_cols).loc[:, base_cols].copy()
    else:
        df_base = pd.DataFrame(columns=base_cols)
    # anadir columnas calculadas si existen errores
    df_full = add_calculated_e_cols(df.copy())
    # detectar columnas calculadas (low/high)
    calc_cols = [c for c in df_full.columns if c.endswith('_low_calc') or c.endswith('_high_calc')]
    # definir features que usaremos para muestrear: escoger subset commun de features con errores
    # preferimos usar Teff, logg, Meta, L si estan disponibles, como en el ejemplo
    sample_features = [f for f in ['Teff', 'logg', 'Meta', 'L'] if f in df_full.columns]
    # construir X_input y y_input en formato triple por caracteristica
    rows = []
    target_rows = []
    for idx, row in df_base.iterrows():
        vals = []
        for f in sample_features:
            v = float(row[f])
            err_cols = _find_error_cols(df_full, f)
            if err_cols is not None:
                # obtener errores desde df_full usando el indice original
                try:
                    e_low = float(df_full.at[idx, err_cols[0]])
                    e_high = float(df_full.at[idx, err_cols[1]])
                except Exception:
                    e_low = 0.05 * abs(v)
                    e_high = 0.05 * abs(v)
            else:
                # error sintetico 5 percent
                e_low = 0.05 * abs(v)
                e_high = 0.05 * abs(v)
            vals.extend([v, e_low, e_high])
        # target M y errores asociados si existen, si no usar 5 percent
        t = float(row['M'])
        err_target = _find_error_cols(df_full, 'M')
        if err_target is not None:
            try:
                ty_low = float(df_full.at[idx, err_target[0]])
                ty_high = float(df_full.at[idx, err_target[1]])
            except Exception:
                ty_low = 0.05 * abs(t)
                ty_high = 0.05 * abs(t)
        else:
            ty_low = 0.05 * abs(t)
            ty_high = 0.05 * abs(t)
        rows.append(vals)
        target_rows.append([t, ty_low, ty_high])
    # si no hay filas base devolver df_base
    if len(rows) == 0:
        if save_csv:
            out = csv_path if csv_path is not None else OUT_CSV_PATH
            os.makedirs(os.path.dirname(out), exist_ok=True)
            df_base.to_csv(out, index=False)
        return df_base
    X_input = np.array(rows, dtype=float)
    y_input = np.array(target_rows, dtype=float)
    # usar la funcion de muestreo para generar X_aug,y_aug
    X_aug, y_aug = data_augmentation_with_uncertainties_np(X_input, y_input, n_samples_per_row)
    # reconstruir DataFrame ampliado: original + muestras
    # X_aug contiene solo valores en el orden de sample_features
    df_rows = []
    rng = np.random.default_rng(1)
    # iterar sobre filas base para mantener otras columnas (R,Prot,Age,etc)
    for idx, base_row in df_base.iterrows():
        # construir fila original completa
        out_row = {}
        for c in base_cols:
            out_row[c] = base_row[c]
        df_rows.append(out_row)
        # generar n_samples_per_row muestras usando errores calculados anteriormente
        # para reproducibilidad usar rng but match X_aug sequence: we will take next n_samples rows from X_aug
    # Ahora, X_aug contiene (1 + n_samples_per_row) rows per original in secuencia
    # reconvertir X_aug y y_aug en DataFrame que contenga sample_features y M
    num_per_original = 1 + n_samples_per_row
    total_originals = len(df_base)
    # sanity check
    if X_aug.shape[0] != total_originals * num_per_original:
        # en caso inesperado construir df_more directamente con muestreo manual
        df_more_list = []
        for idx, base_row in df_base.iterrows():
            # original
            row0 = {c: base_row[c] for c in base_cols}
            df_more_list.append(row0)
            # muestras
            for s in range(n_samples_per_row):
                sample = {}
                # M sample
                t = float(base_row['M'])
                err_target = _find_error_cols(df_full, 'M')
                if err_target is not None:
                    try:
                        ty_low = float(df_full.at[idx, err_target[0]])
                        ty_high = float(df_full.at[idx, err_target[1]])
                    except Exception:
                        ty_low = 0.05 * abs(t)
                        ty_high = 0.05 * abs(t)
                else:
                    ty_low = 0.05 * abs(t)
                    ty_high = 0.05 * abs(t)
                sample['M'] = float(rng.uniform(t - ty_low, t + ty_high))
                for f in sample_features:
                    v = float(base_row[f])
                    err_cols = _find_error_cols(df_full, f)
                    if err_cols is not None:
                        try:
                            e_low = float(df_full.at[idx, err_cols[0]])
                            e_high = float(df_full.at[idx, err_cols[1]])
                        except Exception:
                            e_low = 0.05 * abs(v)
                            e_high = 0.05 * abs(v)
                    else:
                        e_low = 0.05 * abs(v)
                        e_high = 0.05 * abs(v)
                    sample[f] = float(rng.uniform(v - e_low, v + e_high))
                for c in base_cols:
                    if c not in sample:
                        sample[c] = base_row.get(c, pd.NA)
                df_more_list.append(sample)
        df_more = pd.DataFrame(df_more_list)
    else:
        # construir df_more extrayendo secuencias de X_aug
        df_more_list = []
        ii = 0
        for ori_idx in range(total_originals):
            # la primera fila corresponde al original
            seq = X_aug[ii: ii + num_per_original, :]
            seq_y = y_aug[ii: ii + num_per_original]
            # obtener fila original desde df_base
            base_row = df_base.iloc[ori_idx]
            # original
            row0 = {c: base_row[c] for c in base_cols}
            # si sample_features estan entre base_cols, ya incluidos
            # asegurarnos que M toma el valor original
            row0['M'] = float(base_row['M'])
            df_more_list.append(row0)
            # ahora las muestras en seq[1:]
            for s in range(1, seq.shape[0]):
                sample = {}
                sample['M'] = float(seq_y[s])
                for j, f in enumerate(sample_features):
                    sample[f] = float(seq[s, j])
                # rellenar las demas base_cols con el valor original si no estan en sample
                for c in base_cols:
                    if c not in sample:
                        sample[c] = base_row.get(c, pd.NA)
                df_more_list.append(sample)
            ii += num_per_original
        df_more = pd.DataFrame(df_more_list)
    # asegurar columnas en orden deseado
    for c in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age']:
        if c not in df_more.columns:
            df_more[c] = pd.NA
    df_more = df_more[[c for c in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age']]]
    # anadir class_M calculada por cuantiles si existe M
    if 'M' in df_more.columns:
        try:
            cs = pd.qcut(df_more['M'], q=num_clases, labels=False, duplicates='drop')
            cs = cs.fillna(0).astype(int)
            max_label = int(cs.max()) if len(cs) > 0 else 0
            cs = max_label - cs
            df_more['class_M'] = cs.values
        except Exception:
            df_more['class_M'] = 0
    # No replicamos features a 1024 columnas aqui
    # El usuario quiere solo las columnas originales mas class_M
    # Por tanto df_repl sera simplemente df_more con el orden de columnas deseado
    df_repl = df_more.copy()
    # asegurar que el orden de columnas incluye las 8 originales seguido de class_M
    ordered = [c for c in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age'] if c in df_repl.columns]
    if 'class_M' in df_repl.columns:
        ordered.append('class_M')
    df_repl = df_repl[ordered]

    # guardar CSV atomico con df_repl (ya contiene class_M)
    if save_csv:
        out = csv_path if csv_path is not None else OUT_CSV_PATH
        os.makedirs(os.path.dirname(out), exist_ok=True)
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(out), prefix=os.path.basename(out) + '.')
        os.close(fd)
        try:
            df_repl.to_csv(tmp, index=False)
            with open(tmp, 'rb') as f:
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, out)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return df_repl

# Fin del modulo dataset_moredata
