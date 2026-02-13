import os
import json
import subprocess
import shutil
import csv
from utils.dataset_ampliado import get_dataset
from utils.Img_tab import generate_image
from trainer import _train   # importamos la función que devuelve métricas
from utils.dataset_moredata import get_dataset_moredata
from utils.dataset_moredata_ampliado import get_dataset_moredata_ampliado
from utils.Img_tab_moredata import generate_image_moredata

EXPS_DIR = "./exps"
RESULTS_DIR = "/home/victoria/PycharmProjects/PyCIL2/data/Results"

if os.path.exists(RESULTS_DIR):
   shutil.rmtree(RESULTS_DIR)

N_CLASES = 3
MODELS = ["finetune", "icarl", "der", "wa"]
NETWORKS = ["resnet18", "resnet34", "resnet50"]


BASE_DATASET = "gyro_rgb"
DATA_PATH = "/home/victoria/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion/Test_1_RGB"
CSV_PATH = "data/gyro_tot_v20180801_export.csv"
# Rutas para la tercera variante (moredata ampliado)
DATA_PATH3 = "/home/victoria/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion_MoreDataTest_1_RGB"
CSV_PATH3 = "data/gyro_tot_v20180801_export_moredata_ampliado.csv"
IN_CHANNELS = 3
# Forzar que el numero de clases iniciales sea siempre 2
INIT_CLS = 2
INCREMENT = 1
MEMORY_SIZE = 100
MEMORY_PER_CLASS = 10
FIXED_MEMORY = False
SHUFFLE = False
SEED = [1993]
DEVICE = ['0']
PREFIX = "reproduce"
TOPK = 1


get_dataset(save_csv=True, num_clases=N_CLASES)
get_dataset_moredata_ampliado(save_csv=True, num_clases=N_CLASES)
generate_image()
generate_image_moredata()



# Normalizar variables que deben ser iterables
# Definir una funcion auxiliar que garantiza que la entrada sea una lista
def _ensure_list(x):
    # si ya es lista o tupla, retornarla como lista
    if isinstance(x, (list, tuple)):
        return list(x)
    # si es cadena o entero, envolver en lista
    if isinstance(x, (str, int)):
        return [x]
    # intentar iterar: si es iterable distinto de str, convertir a lista
    try:
        iter(x)
        return list(x)
    except Exception:
        # en cualquier otro caso, devolver la entrada envuelta en lista
        return [x]

# Aplicar normalizacion a MODELS y NETWORKS (INIT_CLS es entero fijo)
MODELS = _ensure_list(MODELS)
NETWORKS = _ensure_list(NETWORKS)

def make_config(model, net):
    # Construir configuracion y forzar init_cls = INIT_CLS
    return {
        "prefix": PREFIX,
        "dataset": BASE_DATASET,
        "data_path": DATA_PATH,
        "csv_path": CSV_PATH,
        "convnet_type": net,
        "in_channels": IN_CHANNELS,
        "model_name": model,
        "init_cls": INIT_CLS,
        "increment": INCREMENT,
        "memory_size": MEMORY_SIZE,
        "memory_per_class": MEMORY_PER_CLASS,
        "fixed_memory": FIXED_MEMORY,
        "shuffle": SHUFFLE,
        "seed": SEED,
        "device": DEVICE,
        "topk": TOPK
    }

def save_config(config, filename):
    path = os.path.join(EXPS_DIR, filename)
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    return path

def make_config3(model, net):
    # Configuracion para la tercera variante (moredata_ampliado)
    return {
        "prefix": PREFIX,
        "dataset": BASE_DATASET,
        "data_path": DATA_PATH3,
        "csv_path": CSV_PATH3,
        "convnet_type": net,
        "in_channels": IN_CHANNELS,
        "model_name": model,
        "init_cls": INIT_CLS,
        "increment": INCREMENT,
        "memory_size": MEMORY_SIZE,
        "memory_per_class": MEMORY_PER_CLASS,
        "fixed_memory": FIXED_MEMORY,
        "shuffle": SHUFFLE,
        "seed": SEED,
        "device": DEVICE,
        "topk": TOPK
    }

def run_experiment(config_path, config):
    print(f"\n=== Entrenando con {config_path} ===")
    # ejecutamos entrenamiento y recogemos métricas
    avg_acc, forgetting = _train(config)

    # guardamos resultados en summary.csv
    results_dir = "/home/victoria/PycharmProjects/PyCIL2/data"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "summary.csv")
    header = ["model_name", "convnet_type", "init_cls", "increment", "average_accuracy", "forgetting"]

    row = [
        config.get("model_name", ""),
        config.get("convnet_type", ""),
        config.get("init_cls", ""),
        config.get("increment", ""),
        avg_acc,
        forgetting
    ]

    file_exists = os.path.isfile(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

if __name__ == "__main__":
    os.makedirs(EXPS_DIR, exist_ok=True)
    # Fase 1: ejecutar experimentos usando DATA_PATH y CSV_PATH (config generada por make_config)
    for model in MODELS:
        for net in NETWORKS:
            # construir nombre de fichero para la fase 1
            filename = f"{model}_{net}_{BASE_DATASET}_init{INIT_CLS}.json"
            # generar configuracion con las rutas DATA_PATH/CSV_PATH
            config = make_config(model, net)
            config_path = save_config(config, filename)
            # ejecutar el experimento
            run_experiment(config_path, config)

    # Fase 2: ejecutar experimentos usando DATA_PATH3 y CSV_PATH3 (config generada por make_config3)
    for model in MODELS:
        for net in NETWORKS:
            # construir nombre de fichero para la fase 3 (añadir sufijo _moredata_ampliado)
            filename = f"{model}_{net}_{BASE_DATASET}_moredata_ampliado_init{INIT_CLS}.json"
            # generar configuracion con las rutas DATA_PATH3/CSV_PATH3
            config = make_config3(model, net)
            config_path = save_config(config, filename)
            # ejecutar el experimento
            run_experiment(config_path, config)
