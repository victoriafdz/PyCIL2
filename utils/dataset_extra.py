# Modulo para crear el CSV extra a partir del TXT original
# Este modulo genera un CSV con las columnas replicadas y las columnas calculadas si proceden
import os
# importar tipos opcionales y listas para anotaciones de funcion
from typing import Optional, List

# importar pandas para manipular DataFrames
import pandas as pd

# Calcular la ruta raiz del proyecto (un nivel arriba de la carpeta utils)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Carpeta data dentro del proyecto donde estan los archivos de entrada y salida
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# Asegurar que la carpeta data existe, crearla si no existe
os.makedirs(DATA_DIR, exist_ok=True)
# Ruta por defecto del CSV base (no usado aqui para escribir, se define por coherencia)
BASE_CSV_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801_export.csv')
# Ruta por defecto del CSV extra que generara esta funcion
EXTRA_CSV_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801_export_extra.csv')
# Ruta del fichero TXT original con los datos de entrada
TXT_PATH = os.path.join(DATA_DIR, 'gyro_tot_v20180801.txt')
# Numero objetivo de columnas tras replicacion
TARGET_FEATURES = 1024


# Funcion auxiliar: replicar exactamente la lista de 'features' hasta target_features columnas
# Recibe un DataFrame, una lista de nombres de columnas base y el numero objetivo de columnas
def _replicate_features_exact(df: pd.DataFrame, features: List[str], target_features: int) -> pd.DataFrame:
    # Filtrar las features de interes para quedarse solo con las que existen en el DataFrame
    existing = [f for f in features if f in df.columns]
    # Si no hay ninguna feature valida devolver DataFrame vacio con mismo numero de filas
    if len(existing) == 0:
        return pd.DataFrame(index=df.index)

    # Lista que contendra el orden final de columnas replicadas
    cols_order = []
    # contador de repeticiones para crear sufijos _rep2, _rep3, etc
    rep_num = 1
    # Repetir bloques de las columnas base hasta completar target_features
    while len(cols_order) < target_features:
        for f in existing:
            # En la primera repeticion usamos el nombre base, en las siguientes anadimos sufijo
            if rep_num == 1:
                col_name = f
            else:
                col_name = f"{f}_rep{rep_num}"
            cols_order.append(col_name)
            # Si ya alcanzamos el numero objetivo salir del bucle
            if len(cols_order) >= target_features:
                break
        rep_num += 1

    # Construir un diccionario con los datos para cada columna replicada
    data = {}
    for name in cols_order:
        # obtener el nombre base (sin sufijo _repX) para leer los datos originales
        if '_rep' in name:
            base = name.rsplit('_rep', 1)[0]
        else:
            base = name
        # asignar la serie original del base a la nueva columna replicada
        data[name] = df[base].values

    # Crear un DataFrame con las columnas replicadas en el orden deseado y devolverlo
    replicated_df = pd.DataFrame(data)
    return replicated_df


# Funcion que anade columnas calculadas low/high si existen columnas de error e<name>1/e<name>2
# Para cada pareja e<base>1/e<base>2 y si existe base, se anaden base_low_calc y base_high_calc
def add_calculated_e_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Obtener lista de nombres de columnas actuales
    cols = list(df.columns)
    # Recorrer los nombres para detectar patrones de columnas de error
    for c in cols:
        # Asegurar que el nombre es cadena
        if not isinstance(c, str):
            continue
        # Detectar columnas con formato 'e<name>1' que indican error inferior
        if c.startswith('e') and c.endswith('1'):
            # Extraer el nombre base quitando prefijo 'e' y el sufijo '1'
            base_name = c[1:-1]
            # Construir los nombres esperados de las columnas de error
            e1 = f"e{base_name}1"
            e2 = f"e{base_name}2"
            # Solo si existen ambas columnas de error y la columna base, procedemos
            if e1 in df.columns and e2 in df.columns and base_name in df.columns:
                # Nombres de las columnas calculadas que vamos a crear
                low_name = f"{base_name}_low_calc"
                high_name = f"{base_name}_high_calc"
                try:
                    # Calcular limites: low = base - e1 ; high = base + e2
                    df[low_name] = df[base_name] - df[e1]
                    df[high_name] = df[base_name] + df[e2]
                except Exception:
                    # Si hay error numerico rellenar con NA para no interrumpir el flujo
                    df[low_name] = pd.NA
                    df[high_name] = pd.NA
    # Devolver el DataFrame con las nuevas columnas (si se crearon)
    return df


# Funcion principal: genera el CSV extra (base + columnas calculadas replicadas) y devuelve el DataFrame
def get_dataset_extra(save_csv: bool = True, csv_path_extra: Optional[str] = None, num_clases: int = 5):
    # 1) leer datos desde el fichero TXT original con pandas
    data = pd.read_csv(TXT_PATH, sep='\t', header=0)

    # 2) eliminar filas con valores nulos en cualquier columna para mantener consistencia
    df_clean = data.dropna(axis=0).copy()

    # 3) aplicar filtro fisico si las columnas 'class','M','Prot' estan presentes
    if {'class', 'M', 'Prot'}.issubset(df_clean.columns):
        # Mantener solo clase 'MS' y rangos fisicos razonables para M y Prot
        df_clean = df_clean.loc[(df_clean['class'] == 'MS') & (df_clean['M'] < 2) & (df_clean['M'] > 0.7) & (df_clean['Prot'] < 50)].copy()

    # definir las 8 columnas base en el orden requerido, solo si existen en el dataframe
    base_features = [f for f in ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age'] if f in df_clean.columns]

    # 4) preparar el DataFrame con columnas calculadas a partir de errores (si procede)
    # Llamar a la funcion que anade <base>_low_calc y <base>_high_calc cuando se detecten e<base>1/e<base>2
    df_with_calc = add_calculated_e_cols(df_clean.copy())

    # 5) detectar las columnas calculadas que se acabaron de anadir
    calc_cols = [c for c in df_with_calc.columns if c.endswith('_low_calc') or c.endswith('_high_calc')]

    # 6) definir las features que queremos replicar en el CSV extra: base + calc_cols
    features_extra = base_features + [c for c in calc_cols if c not in base_features]

    # 7) replicar las features_extra exactamente hasta TARGET_FEATURES columnas
    df_extra_repl = _replicate_features_exact(df_with_calc, features_extra, TARGET_FEATURES)

    # 8) calcular la columna class_M usando la columna M original y anadirla como ultima columna
    if 'M' in df_with_calc.columns:
        # Calcular cuantiles discretos 0..num_clases-1
        class_series_extra = pd.qcut(df_with_calc['M'], q=num_clases, labels=False)
        # Invertir las etiquetas para que mayor M tenga label mayor
        max_label_extra = int(class_series_extra.max())
        class_series_extra = max_label_extra - class_series_extra
        # Anadir la columna class_M al DataFrame replicado
        df_extra_repl['class_M'] = class_series_extra.values

    # 9) determinar la ruta de salida para el CSV extra
    out_extra = csv_path_extra if csv_path_extra is not None else EXTRA_CSV_PATH
    # Si se solicita guardar, crear carpeta y escribir CSV
    if save_csv:
        os.makedirs(os.path.dirname(out_extra), exist_ok=True)
        df_extra_repl.to_csv(out_extra, index=False)

    # 10) devolver el DataFrame extra replicado con la columna class_M
    return df_extra_repl

# Fin del modulo dataset_extra
