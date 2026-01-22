import pandas as pd
import os
import shutil
from utils.IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Importamos funciones específicas de IGTD:
# - min_max_transform: normaliza los datos al rango [0,1]
# - table_to_image: convierte una tabla de datos en imágenes
# - select_features_by_variation: selecciona las columnas con mayor variación
def generate_image_extra():
    # Parámetros de configuración
    num_row = 32                     # Número de filas de cada imagen generada
    num_col = 32                    # Número de columnas de cada imagen generada
    num = num_row * num_col          # Número total de características que se usarán (igual al número de píxeles)
    save_image_size = 3              # Tamaño de la imagen guardada (escala de salida)
    max_step = 30000                 # Número máximo de iteraciones del algoritmo IGTD
    val_step = 300                   # Cada cuántos pasos se valida el resultado

    # Ruta al archivo CSV de gyro
    data_path = '/home/victoria/PycharmProjects/PyCIL2/data/gyro_tot_v20180801_export_extra.csv'  # Ruta al dataset exportado
    data = pd.read_csv(data_path, sep=',', header=0, low_memory=False)
    # Leemos el CSV con pandas:
    # - sep=',' → separador por comas
    # - header=0 → primera fila contiene nombres de columnas
    # - low_memory=False → evita problemas con tipos de datos

    # Etiquetas: usamos la columna 'class' que contiene valores discretos 0..4
    labels = data['class_M'].astype(int).values

    # Features: todas las columnas excepto 'class'
    features = data.drop(columns=['class_M'])

    # Seleccionar las características con mayor variación
    id = select_features_by_variation(features, variation_measure='var', num=num)
    features = features.iloc[:, id]
    # Usamos IGTD para seleccionar las columnas con mayor varianza.
    # Esto reduce dimensionalidad y mejora la calidad de las imágenes.

    # Normalizar los datos con min-max
    norm_data = min_max_transform(features.values)
    norm_data = pd.DataFrame(norm_data, columns=features.columns, index=features.index)
    # Normalizamos los valores de cada característica al rango [0,1].
    # Luego reconstruimos un DataFrame con los datos normalizados.


    # Primera ejecución: distancia Euclidiana + error absoluto
    fea_dist_method = 'Euclidean'     # Metodo de distancia entre características
    image_dist_method = 'Euclidean'   # Metodo de distancia entre imágenes
    error = 'abs'                     # Tipo de error a minimizar (absoluto)
    result_dir = os.path.expanduser('~/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion_extra/Test_1')
    result_dir2 = os.path.expanduser('~/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion_extra/Test_1_RGB')

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir2, exist_ok=True)

    table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method,
                    save_image_size, max_step, val_step, result_dir, error)
    # Generamos las imágenes con IGTD usando distancia Euclidiana y error absoluto
    organize_images_from_data(result_dir, result_dir2, labels)      # Organizamos las imágenes en carpetas por class_M

    # Segunda ejecución: correlación de Pearson + distancia Manhattan + error cuadrático
    fea_dist_method = 'Pearson'       # Metodo de distancia entre características (correlación)
    image_dist_method = 'Manhattan'   # Metodo de distancia entre imágenes (Manhattan)
    error = 'squared'                 # Tipo de error a minimizar (cuadrático)
    norm_data = norm_data.iloc[:, :800]  # Opcional: reducimos dimensionalidad a 800 features
    result_dir = os.path.expanduser('~/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion_extra/Test_2')
    result_dir2 = os.path.expanduser('~/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion_extra/Test_2_RGB')

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir2, exist_ok=True)

    table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method,
                save_image_size, max_step, val_step, result_dir, error)
    #organizar
    organize_images_from_data(result_dir, result_dir2, labels)

# Función auxiliar para organizar imágenes en carpetas por clase

def organize_images_from_data(result_dir, result_dir2, labels):
    data_dir = os.path.join(result_dir, "data")                 # Carpeta donde IGTD guarda las imágenes originales
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".png")])  # Lista de imágenes .png

    dataRGB_dir = os.path.join(result_dir2, "dataRGB")          # Carpeta temporal para copias
    os.makedirs(dataRGB_dir, exist_ok=True)

    # Copiar originales y organizarlas en carpetas por clase (grayscale)
    for idx, file in enumerate(files):
        img_path = os.path.join(data_dir, file)
        shutil.copy(img_path, os.path.join(dataRGB_dir, file))  # Copiamos la imagen a dataRGB

        label = labels[idx]
        class_dir = os.path.join(result_dir, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)
        shutil.move(img_path, os.path.join(class_dir, file))    # Movemos la imagen original a su carpeta de clase

    # Ahora procesamos las imágenes en dataRGB y les aplicamos el colormap gist_rainbow
    cm = plt.get_cmap('gist_rainbow')
    rgb_files = sorted([f for f in os.listdir(dataRGB_dir) if f.endswith(".png")])
    for idx, file in enumerate(rgb_files):
        img_path = os.path.join(dataRGB_dir, file)
        gray_img = Image.open(img_path).convert("L")            # Abrimos como grayscale
        gray_array = np.array(gray_img) / 255.0                 # Normalizamos a [0,1]

        # Aplicamos el colormap → RGBA float [0,1]
        colored_rgba = cm(gray_array)
        rgb_uint8 = (colored_rgba[:, :, :3] * 255).astype(np.uint8)  # Convertimos a RGB uint8

        color_img = Image.fromarray(rgb_uint8, mode="RGB")

        label = labels[idx]
        class_dir = os.path.join(result_dir2, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)
        color_img.save(os.path.join(class_dir, file))           # Guardamos la imagen coloreada

    # Limpieza de carpetas intermedias
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if os.path.exists(dataRGB_dir):
        shutil.rmtree(dataRGB_dir)

    for fname in [
        "error_and_iteration.png",
        "error_and_runtime.png",
        "optimized_feature_ranking.png",
        "original_feature_ranking.png",
        "image_ranking.png",
        "Results.pkl",
        "Results_Auxiliary.pkl",
    ]:
        fpath = os.path.join(result_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    squared_dir = os.path.join(result_dir, "squared")
    if os.path.exists(squared_dir):
        shutil.rmtree(squared_dir)

    abs_dir = os.path.join(result_dir, "abs")
    if os.path.exists(abs_dir):
        shutil.rmtree(abs_dir)
