# Nuevo modulo dataset_moredata_ampliado.py
# Este modulo crea filas ampliadas a partir del TXT original y luego
# replica las columnas base hasta obtener 1024 columnas exactas.
# Comentarios en espanol sin tildes ni caracteres especiales.

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# importamos la funcion de replicacion de dataset_ampliado para mantener la misma logica
from utils import dataset_ampliado as ds_ampliado

# Rutas coherentes con el proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TXT_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801.txt')
OUT_CSV_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801_export_moredata_ampliado.csv')

# Funcion auxiliar: busca columnas de error para una columna base
def _find_error_cols(df: pd.DataFrame, col: str) -> Optional[Tuple[str, str]]:
    # comprobamos varios patrones comunes para errores
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

# Funcion que anade columnas calculadas a partir de columnas de error
def add_calculated_e_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    for c in cols:
        if not isinstance(c, str):
            continue
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

# Version numpy de la funcion de muestreo por incertidumbres
def data_augmentation_with_uncertainties_np(X_input: np.ndarray, y_input: np.ndarray, n_samples: int):
    X = X_input[:, 0::3]
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
    y = y_input[:, 0::3]
    ye = y_input[:, 1::]
    if n_samples == 0:
        return X, np.ravel(y)
    from numpy.random import default_rng
    rng = default_rng(1)
    X_aug_list = []
    y_aug_list = []
    for (s_x, s_xe, s_y, s_ye) in zip(X, Xe, y, ye):
        X_aug_list.append(s_x.copy())
        try:
            y_aug_list.append(float(s_y))
        except Exception:
            y_aug_list.append(float(np.ravel(s_y)[0]))
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
                x_new[ff] = float(rng.uniform(v - e_low, v + e_high))
                ee += 2
            X_aug_list.append(x_new)
            y_aug_list.append(y_new[ns])
    X_aug = np.vstack(X_aug_list)
    y_aug = np.array(y_aug_list)
    return X_aug, y_aug

# Funcion principal: genera CSV con 1024 columnas finales
def get_dataset_moredata_ampliado(save_csv: bool = True, csv_path: Optional[str] = None, n_samples_per_row: int = 10, num_clases: int = 2) -> pd.DataFrame:
    # leer el TXT original
    data = pd.read_csv(TXT_PATH, sep='\t', header=0)
    df = data.copy()
    # aplicamos filtro fisico si las columnas existen
    if {'class', 'M', 'Prot'}.issubset(df.columns):
        df = df.loc[(df['class'] == 'MS') & (df['M'] < 2) & (df['M'] > 0.7) & (df['Prot'] < 50)].copy()
    # columnas base que queremos mantener
    base_cols = [c for c in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age'] if c in df.columns]
    if base_cols:
        df_base = df.dropna(subset=base_cols).loc[:, base_cols].copy()
    else:
        df_base = pd.DataFrame(columns=base_cols)
    # anadir columnas calculadas si hay errores
    df_full = add_calculated_e_cols(df.copy())
    sample_features = [f for f in ['Teff', 'logg', 'Meta', 'L'] if f in df_full.columns]
    rows = []
    target_rows = []
    for idx, row in df_base.iterrows():
        vals = []
        for f in sample_features:
            v = float(row[f])
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
            vals.extend([v, e_low, e_high])
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
    if len(rows) == 0:
        if save_csv:
            out = csv_path if csv_path is not None else OUT_CSV_PATH
            os.makedirs(os.path.dirname(out), exist_ok=True)
            df_base.to_csv(out, index=False)
        return df_base
    X_input = np.array(rows, dtype=float)
    y_input = np.array(target_rows, dtype=float)
    X_aug, y_aug = data_augmentation_with_uncertainties_np(X_input, y_input, n_samples_per_row)
    # reconstruir df_more (original + muestras)
    num_per_original = 1 + n_samples_per_row
    total_originals = len(df_base)
    df_more_list = []
    ii = 0
    for ori_idx in range(total_originals):
        seq = X_aug[ii: ii + num_per_original, :]
        seq_y = y_aug[ii: ii + num_per_original]
        base_row = df_base.iloc[ori_idx]
        row0 = {c: base_row[c] for c in base_cols}
        row0['M'] = float(base_row['M'])
        df_more_list.append(row0)
        for s in range(1, seq.shape[0]):
            sample = {}
            sample['M'] = float(seq_y[s])
            for j, f in enumerate(sample_features):
                sample[f] = float(seq[s, j])
            for c in base_cols:
                if c not in sample:
                    sample[c] = base_row.get(c, pd.NA)
            df_more_list.append(sample)
        ii += num_per_original
    df_more = pd.DataFrame(df_more_list)
    # asegurar columnas en orden
    for c in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age']:
        if c not in df_more.columns:
            df_more[c] = pd.NA
    df_more = df_more[[c for c in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age']]]
    # añadir class_M
    if 'M' in df_more.columns:
        try:
            cs = pd.qcut(df_more['M'], q=num_clases, labels=False, duplicates='drop')
            cs = cs.fillna(0).astype(int)
            max_label = int(cs.max()) if len(cs) > 0 else 0
            cs = max_label - cs
            df_more['class_M'] = cs.values
        except Exception:
            df_more['class_M'] = 0
    # ahora replicamos las columnas para obtener exactamente 1024 columnas
    base_features = [f for f in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age'] if f in df_more.columns]
    try:
        TARGET_FINAL = 1024
        df_repl = ds_ampliado._replicate_features_exact(df_more, base_features, TARGET_FINAL)
        if 'class_M' not in df_repl.columns and 'class_M' in df_more.columns:
            df_repl['class_M'] = df_more['class_M'].values
    except Exception:
        # si falla, devolvemos df_more con las columnas originales y class_M
        df_repl = df_more.copy()
        ordered = [c for c in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age'] if c in df_repl.columns]
        if 'class_M' in df_repl.columns:
            ordered.append('class_M')
        df_repl = df_repl[ordered]
    # guardar CSV
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

# Si se ejecuta directamente, generar el CSV usando n_samples_per_row=1 por defecto
if __name__ == '__main__':
    get_dataset_moredata_ampliado(save_csv=True, n_samples_per_row=1, num_clases=2)

