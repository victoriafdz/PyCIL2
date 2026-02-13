# Archivo: utils/Img_tab_moredata.py
# Version adaptada de Img_tab.py para usar el CSV 'moredata'
# Comentarios en espanol sin tildes ni caracteres especiales

import pandas as pd
import os
import shutil
from utils.IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Esta funcion genera imagenes a partir del CSV 'moredata'.
# Por defecto usa el CSV generado por dataset_moredata_ampliado.
# dry_run True hace solo la fase de comprobacion y devuelve los datos normalizados y etiquetas

def generate_image_moredata(csv_path: str = None,
                            num_row: int = 32,
                            num_col: int = 32,
                            save_image_size: int = 3,
                            max_step: int = 30000,
                            val_step: int = 300,
                            result_dir_base: str = None,
                            dry_run: bool = False):
    # csv por defecto si no se pasa ninguno
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    default_csv = os.path.join(data_dir, 'gyro_tot_v20180801_export_moredata_ampliado.csv')
    if csv_path is None:
        csv_path = default_csv

    # leer CSV con pandas
    # low_memory False para evitar inferencias fragmentadas de tipos
    data = pd.read_csv(csv_path, sep=',', header=0, low_memory=False)

    # obtener etiquetas class_M (se asume que existe)
    if 'class_M' not in data.columns:
        raise ValueError('El CSV no contiene la columna class_M')
    labels = data['class_M'].astype(int).values

    # features: todas las columnas excepto class_M
    features = data.drop(columns=['class_M'])

    # seleccionar el numero de caracteristicas que necesitamos (num_row * num_col)
    num = num_row * num_col
    # select_features_by_variation devuelve indices de columnas con mayor variacion
    id_selected = select_features_by_variation(features, variation_measure='var', num=num)
    features_sel = features.iloc[:, id_selected]

    # normalizar al rango [0,1]
    norm_data = min_max_transform(features_sel.values)
    norm_df = pd.DataFrame(norm_data, columns=features_sel.columns, index=features_sel.index)

    # si dry_run True devolvemos informacion util y no ejecutamos IGTD
    if dry_run:
        print('dry_run: csv leido:', csv_path)
        print('filas (muestras):', norm_df.shape[0])
        print('features seleccionadas (num):', norm_df.shape[1])
        return norm_df, labels

    # determinar directorios de salida si no se pasan
    if result_dir_base is None:
        result_dir_base = os.path.join(project_root, 'data', 'Results', 'Gyro_Conversion', 'MoreData')
    result_dir = os.path.join(result_dir_base, 'Test_1')
    result_dir2 = os.path.join(result_dir_base, 'Test_1_RGB')

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir2, exist_ok=True)

    # Llamada al algoritmo IGTD para generar imagenes
    fea_dist_method = 'Euclidean'
    image_dist_method = 'Euclidean'
    error = 'abs'

    # table_to_image espera un DataFrame normalizado, la forma objetivo y parametros
    table_to_image(norm_df, [num_row, num_col], fea_dist_method, image_dist_method,
                   save_image_size, max_step, val_step, result_dir, error)

    # organizar las imagenes en carpetas por clase y generar versiones RGB
    organize_images_from_data(result_dir, result_dir2, labels)

    # segunda ejecucion opcional con otras metricas (si se desea se puede descomentar)
    # fea_dist_method = 'Pearson'
    # image_dist_method = 'Manhattan'
    # error = 'squared'
    # norm_data_reduced = norm_df.iloc[:, :800]
    # result_dir = os.path.join(result_dir_base, 'Test_2')
    # result_dir2 = os.path.join(result_dir_base, 'Test_2_RGB')
    # os.makedirs(result_dir, exist_ok=True)
    # os.makedirs(result_dir2, exist_ok=True)
    # table_to_image(norm_data_reduced, [num_row, num_col], fea_dist_method, image_dist_method,
    #                save_image_size, max_step, val_step, result_dir, error)
    # organize_images_from_data(result_dir, result_dir2, labels)

# Funcion auxiliar copiada y adaptada de Img_tab.py para organizar las imagenes

def organize_images_from_data(result_dir, result_dir2, labels):
    # carpeta donde IGTD deja las imagenes generadas
    data_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directorio de salida de IGTD no encontrado: {data_dir}")

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])

    dataRGB_dir = os.path.join(result_dir2, 'dataRGB')
    os.makedirs(dataRGB_dir, exist_ok=True)

    # copiar y mover las imagenes generadas por IGTD a carpetas por clase (grayscale)
    # ahora usamos el numero del fichero para mapear a la etiqueta correspondiente
    for idx, file in enumerate(files):
        img_path = os.path.join(data_dir, file)
        # copiar a carpeta temporal para generar la version RGB
        shutil.copy(img_path, os.path.join(dataRGB_dir, file))

        # elegir etiqueta usando el indice (si falta, usar la ultima etiqueta disponible)
        if idx < len(labels):
            chosen_label = int(labels[idx])
        else:
            chosen_label = int(labels[-1])

        # crear carpeta de clase y mover el fichero original alli
        class_dir = os.path.join(result_dir, f'class_{chosen_label}')
        os.makedirs(class_dir, exist_ok=True)
        # mover (esto elimina el fichero de data_dir)
        shutil.move(img_path, os.path.join(class_dir, file))

    # aplicar colormap a las copias y guardarlas en result_dir2 organizadas por clase
    cm = plt.get_cmap('gist_rainbow')
    rgb_files = sorted([f for f in os.listdir(dataRGB_dir) if f.endswith('.png')])
    for idx, file in enumerate(rgb_files):
        img_path = os.path.join(dataRGB_dir, file)
        gray_img = Image.open(img_path).convert('L')
        gray_array = np.array(gray_img) / 255.0

        colored_rgba = cm(gray_array)
        rgb_uint8 = (colored_rgba[:, :, :3] * 255).astype(np.uint8)

        color_img = Image.fromarray(rgb_uint8, mode='RGB')

        # asignar clase usando el indice ordenado
        if idx < len(labels):
            chosen_label = int(labels[idx])
        else:
            chosen_label = int(labels[-1])

        class_dir = os.path.join(result_dir2, f'class_{chosen_label}')
        os.makedirs(class_dir, exist_ok=True)
        color_img.save(os.path.join(class_dir, file))

    # limpieza de carpetas y ficheros generados por IGTD que no interesan
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if os.path.exists(dataRGB_dir):
        shutil.rmtree(dataRGB_dir)

    for fname in [
        'error_and_iteration.png',
        'error_and_runtime.png',
        'optimized_feature_ranking.png',
        'original_feature_ranking.png',
        'image_ranking.png',
        'Results.pkl',
        'Results_Auxiliary.pkl',
    ]:
        fpath = os.path.join(result_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    # eliminar subcarpetas residuales
    squared_dir = os.path.join(result_dir, 'squared')
    if os.path.exists(squared_dir):
        shutil.rmtree(squared_dir)

    abs_dir = os.path.join(result_dir, 'abs')
    if os.path.exists(abs_dir):
        shutil.rmtree(abs_dir)
