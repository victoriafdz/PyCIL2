import os
import json
import subprocess
import shutil
import csv
from utils.dataset_ampliado import get_dataset
from utils.Img_tab import generate_image
from trainer import _train   # importamos la función que devuelve métricas

EXPS_DIR = "./exps"
RESULTS_DIR = "/home/victoria/PycharmProjects/PyCIL2/data/Results"

if os.path.exists(RESULTS_DIR):
   shutil.rmtree(RESULTS_DIR)

N_CLASES = 3
MODELS = ["finetune", "icarl", "der", "wa"]
NETWORKS = ["resnet18", "resnet34", "resnet50"]
INIT_CLASSES = [1, 2]

BASE_DATASET = "gyro_rgb"
DATA_PATH = "/home/victoria/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion/Test_1_RGB"
CSV_PATH = "data/gyro_tot_v20180801_export.csv"
IN_CHANNELS = 3
INCREMENT = 1
MEMORY_SIZE = 100
MEMORY_PER_CLASS = 10
FIXED_MEMORY = False
SHUFFLE = False
SEED = [1993]
DEVICE = ['0']
PREFIX = "reproduce"
TOPK = 1

get_dataset(num_clases=N_CLASES)
generate_image()

def make_config(model, net, init_cls):
    return {
        "prefix": PREFIX,
        "dataset": BASE_DATASET,
        "data_path": DATA_PATH,
        "csv_path": CSV_PATH,
        "convnet_type": net,
        "in_channels": IN_CHANNELS,
        "model_name": model,
        "init_cls": init_cls,
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
    for model in MODELS:
        for net in NETWORKS:
            for init_cls in INIT_CLASSES:
                filename = f"{model}_{net}_{BASE_DATASET}_init{init_cls}.json"
                config = make_config(model, net, init_cls)
                config_path = save_config(config, filename)
                run_experiment(config_path, config)
