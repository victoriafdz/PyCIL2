import os
import json
import subprocess

# Carpeta donde se guardarán los JSONs
EXPS_DIR = "./exps"

# Parámetros variables
MODELS = ["finetune", "icarl", "der", "wa"]          # modelo
NETWORKS = ["resnet18", "resnet34", "resnet50"]      # tipos de red
INIT_CLASSES = [2, 3]                                # número de clases iniciales

# Parámetros fijos
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

def make_config(model, net, init_cls):
    """Genera un diccionario de configuración para un experimento."""
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
        "device": DEVICE
    }

def save_config(config, filename):
    """Guarda el JSON en la carpeta exps."""
    path = os.path.join(EXPS_DIR, filename)
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    return path

def run_experiment(config_path):
    """Ejecuta el experimento con main.py."""
    print(f"\n=== Entrenando con {config_path} ===")
    cmd = ["python", "main.py", "--config", config_path]
    subprocess.run(cmd)

if __name__ == "__main__":
    os.makedirs(EXPS_DIR, exist_ok=True)
    for model in MODELS:
        for net in NETWORKS:
            for init_cls in INIT_CLASSES:
                filename = f"{model}_{net}_{BASE_DATASET}_init{init_cls}.json"
                config = make_config(model, net, init_cls)
                config_path = save_config(config, filename)
                run_experiment(config_path)
