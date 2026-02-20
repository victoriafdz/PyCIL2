import pandas as pd
import os
import shutil
from utils.IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re

# Importamos funciones especificas de IGTD:
# - min_max_transform: normaliza los datos al rango [0,1]
# - table_to_image: convierte una tabla de datos en imagenes
# - select_features_by_variation: selecciona las columnas con mayor variacion
def generate_image(csv_path: str = None, result_dir_base: str = None):
    # Parametros de configuracion
    num_row = 32                     # Numero de filas de cada imagen generada
    num_col = 32                    # Numero de columnas de cada imagen generada
    num = num_row * num_col          # Numero total de caracteristicas que se usaran (igual al numero de pixeles)
    save_image_size = 3              # Tamano de la imagen guardada (escala de salida)
    max_step = 30000                 # Numero maximo de iteraciones del algoritmo IGTD
    val_step = 300                   # Cada cuantos pasos se valida el resultado

    # Ruta al archivo CSV de gyro
    data_path = csv_path  # Ruta al dataset exportado
    data = pd.read_csv(data_path, sep=',', header=0, low_memory=False)
    # Leemos el CSV con pandas:
    # - sep=',' -> separador por comas
    # - header=0 -> primera fila contiene nombres de columnas
    # - low_memory=False -> evita problemas con tipos de datos

    # Etiquetas: usamos la columna 'class' que contiene valores discretos 0..4
    labels = data['class_M'].astype(int).values

    # Features: todas las columnas excepto 'class'
    features = data.drop(columns=['class_M'])

    # Seleccionar las caracteristicas con mayor variacion
    id = select_features_by_variation(features, variation_measure='var', num=num)
    features = features.iloc[:, id]
    # Usamos IGTD para seleccionar las columnas con mayor varianza.
    # Esto reduce dimensionalidad y mejora la calidad de las imagenes.

    # Normalizar los datos con min-max
    norm_data = min_max_transform(features.values)
    norm_data = pd.DataFrame(norm_data, columns=features.columns, index=features.index)
    # Normalizamos los valores de cada caracteristica al rango [0,1].
    # Luego reconstruimos un DataFrame con los datos normalizados.


    # Determinar base de resultados: si se pasa result_dir_base lo usamos, si no usamos la ruta por defecto
    # Comentario en espanol sin tildes ni caracteres especiales
    if result_dir_base:
        base_results = os.path.expanduser(result_dir_base)
    else:
        base_results = os.path.expanduser('~/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion')

    # Primera ejecucion: distancia Euclidiana + error absoluto
    fea_dist_method = 'Euclidean'     # Metodo de distancia entre caracteristicas
    image_dist_method = 'Euclidean'   # Metodo de distancia entre imagenes
    error = 'abs'                     # Tipo de error a minimizar (absoluto)
    # Construimos las rutas de resultado usando la base
    result_dir = os.path.join(base_results, 'Test_1')
    result_dir2 = os.path.join(base_results, 'Test_1_RGB')

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir2, exist_ok=True)

    table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method,
                    save_image_size, max_step, val_step, result_dir, error)
    # Generamos las imagenes con IGTD usando distancia Euclidiana y error absoluto
    organize_images_from_data(result_dir, result_dir2, labels)      # Organizamos las imagenes en carpetas por class_M

    # Segunda ejecucion: correlacion de Pearson + distancia Manhattan + error cuadratico
    fea_dist_method = 'Pearson'       # Metodo de distancia entre caracteristicas (correlacion)
    image_dist_method = 'Manhattan'   # Metodo de distancia entre imagenes (Manhattan)
    error = 'squared'                 # Tipo de error a minimizar (cuadratico)
    norm_data = norm_data.iloc[:, :800]  # Opcional: reducimos dimensionalidad a 800 features
    # Construimos las rutas de resultado para la segunda ejecucion
    result_dir = os.path.join(base_results, 'Test_2')
    result_dir2 = os.path.join(base_results, 'Test_2_RGB')

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir2, exist_ok=True)

    table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method,
                save_image_size, max_step, val_step, result_dir, error)
    #organizar
    organize_images_from_data(result_dir, result_dir2, labels)

# Funcion auxiliar para organizar imagenes en carpetas por clase
def organize_images_from_data(result_dir, result_dir2, labels):
    # comprobamos la carpeta donde IGTD guarda las imagenes originales
    data_dir = os.path.join(result_dir, "data")
    # si la carpeta no existe salimos de la funcion para evitar errores
    if not os.path.exists(data_dir):
        print(f"warning: data_dir not found: {data_dir}")
        return

    # funcion auxiliar que extrae un numero del nombre de fichero para ordenar numericamente
    def _num_key(fname):
        # buscamos la primera secuencia de digitos en el nombre del fichero
        m = re.search(r"(\d+)", fname)
        # si hay digitos devolvemos su valor entero para usar como clave de ordenacion
        if m:
            return int(m.group(1))
        # si no hay digitos devolvemos el nombre en minusculas para ordenacion alfabetica estable
        return fname.lower()

    # listamos todos los ficheros png en la carpeta de datos y los ordenamos con la funcion clave
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".png")], key=_num_key)

    # carpeta temporal donde guardaremos copias RGB antes de organizarlas
    dataRGB_dir = os.path.join(result_dir2, "dataRGB")
    # nos aseguramos de que la carpeta temporal exista
    os.makedirs(dataRGB_dir, exist_ok=True)

    # convertimos las etiquetas a un array numpy para acceso rapido por indice
    labels_arr = np.asarray(labels)
    # contamos el numero de ficheros encontrados
    n_files = len(files)
    # contamos el numero de etiquetas disponibles
    n_labels = len(labels_arr)
    # si los numeros no coinciden avisamos al usuario pero continuamos en modo seguro
    if n_files != n_labels:
        print(f"warning: number of images ({n_files}) != number of labels ({n_labels}). Using min length to avoid misalignment.")

    # longitud minima para evitar index errors si hay desajuste
    L = min(n_files, n_labels)

    # bucle principal: copiamos cada imagen a dataRGB y movemos el original a la carpeta de su clase
    for idx in range(L):
        # nombre del fichero actual según la lista ordenada
        file = files[idx]
        # ruta completa del fichero original
        img_path = os.path.join(data_dir, file)
        # ruta destino para la copia temporal en color
        dst_copy = os.path.join(dataRGB_dir, file)
        # intentamos copiar el fichero original a la carpeta temporal
        try:
            shutil.copy(img_path, dst_copy)
        except Exception as e:
            # si falla la copia mostramos advertencia y saltamos esta imagen
            print(f"warning: failed to copy {img_path} -> {dst_copy}: {e}")
            continue

        # asignacion de etiqueta: preferimos usar el numero dentro del nombre de fichero si existe
        m = re.search(r"(\d+)", file)
        # si encontramos un numero en el nombre intentamos usarlo como indice en labels
        if m:
            try:
                file_index = int(m.group(1))
            except Exception:
                file_index = None
        else:
            file_index = None

        # determinamos la etiqueta de la imagen
        if file_index is not None and 0 <= file_index < n_labels:
            # usamos la etiqueta correspondiente al indice extraido del nombre
            try:
                label = int(labels_arr[file_index])
            except Exception:
                # en caso de fallo convertimos a cero
                label = 0
        else:
            # en caso de no disponer de indice valido usamos la posicion ordenada como respaldo
            try:
                label = int(labels_arr[idx])
            except Exception:
                label = 0

        # construimos la ruta de la carpeta de la clase dentro del directorio principal
        class_dir = os.path.join(result_dir, f"class_{label}")
        # nos aseguramos de que la carpeta de la clase exista
        os.makedirs(class_dir, exist_ok=True)
        # ruta final donde moveremos el fichero original
        dst_move = os.path.join(class_dir, file)
        # si ya existe un fichero con el mismo nombre lo eliminamos para evitar errores
        if os.path.exists(dst_move):
            try:
                os.remove(dst_move)
            except Exception:
                pass
        # intentamos mover el fichero original a su carpeta de clase
        try:
            shutil.move(img_path, dst_move)
        except Exception as e:
            # si falla el movimiento mostramos advertencia y continuamos
            print(f"warning: failed to move {img_path} -> {dst_move}: {e}")

    # ahora procesamos las copias en dataRGB: aplicamos colormap y guardamos en result_dir2 por clase
    cm = plt.get_cmap('gist_rainbow')
    # listamos y ordenamos las copias en la carpeta temporal
    rgb_files = sorted([f for f in os.listdir(dataRGB_dir) if f.endswith(".png")], key=_num_key)
    # usamos la longitud minima entre copias y etiquetas para evitar desalineacion
    L2 = min(len(rgb_files), n_labels)

    for idx in range(L2):
        # nombre del fichero en la carpeta temporal
        file = rgb_files[idx]
        # ruta completa del fichero temporal
        img_path = os.path.join(dataRGB_dir, file)
        # intentamos abrir la imagen en modo grayscale
        try:
            gray_img = Image.open(img_path).convert("L")
        except Exception as e:
            # si no se puede abrir la imagen mostramos advertencia y continuamos
            print(f"warning: failed to open {img_path}: {e}")
            continue
        # convertimos la imagen a un array normalizado en [0,1]
        gray_array = np.array(gray_img) / 255.0
        # aplicamos el colormap devolviendo un array RGBA float en [0,1]
        colored_rgba = cm(gray_array)
        # convertimos los canales RGB a uint8 en [0,255]
        rgb_uint8 = (colored_rgba[:, :, :3] * 255).astype(np.uint8)
        # creamos una imagen RGB desde el array
        color_img = Image.fromarray(rgb_uint8, mode="RGB")

        # mapeo de etiqueta igual que antes: preferimos indice en nombre si existe
        m = re.search(r"(\d+)", file)
        if m:
            try:
                file_index = int(m.group(1))
            except Exception:
                file_index = None
        else:
            file_index = None

        if file_index is not None and 0 <= file_index < n_labels:
            try:
                label = int(labels_arr[file_index])
            except Exception:
                label = 0
        else:
            try:
                label = int(labels_arr[idx])
            except Exception:
                label = 0

        # carpeta objetivo en result_dir2 para la imagen coloreada
        class_dir = os.path.join(result_dir2, f"class_{label}")
        # nos aseguramos de que la carpeta exista
        os.makedirs(class_dir, exist_ok=True)
        # ruta final donde guardaremos la imagen coloreada
        out_path = os.path.join(class_dir, file)
        # intentamos salvar la imagen coloreada
        try:
            color_img.save(out_path)
        except Exception as e:
            print(f"warning: failed to save {out_path}: {e}")

    # limpieza de carpetas intermedias si existen
    if os.path.exists(data_dir):
        try:
            shutil.rmtree(data_dir)
        except Exception as e:
            print(f"warning: failed to remove {data_dir}: {e}")
    if os.path.exists(dataRGB_dir):
        try:
            shutil.rmtree(dataRGB_dir)
        except Exception as e:
            print(f"warning: failed to remove {dataRGB_dir}: {e}")

    # eliminamos ficheros auxiliares que genera IGTD dentro de result_dir si existen
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
            try:
                os.remove(fpath)
            except Exception:
                pass

    # eliminamos carpetas auxiliares si existen
    squared_dir = os.path.join(result_dir, "squared")
    if os.path.exists(squared_dir):
        try:
            shutil.rmtree(squared_dir)
        except Exception:
            pass

    abs_dir = os.path.join(result_dir, "abs")
    if os.path.exists(abs_dir):
        try:
            shutil.rmtree(abs_dir)
        except Exception:
            pass
