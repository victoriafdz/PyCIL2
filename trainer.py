import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        return _train(args)   # devolvemos métricas


def _train(args):

    # Si init_cls == increment, PyCIL fuerza init_cls = 0 (comportamiento original)
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]

    # Carpeta donde se guardarán los logs
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])

    # Crear carpeta si no existe
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    # Nombre del archivo de log
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )

    # Configuración del logger (igual que PyCIL original)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Semillas y dispositivo (comportamiento original)
    _set_random()
    _set_device(args)
    print_args(args)

    # === IMPORTANTE ===
    # PyCIL original requiere data_path como último parámetro
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["aug"] if "aug" in args else 1,
        args["data_path"]     # ← ESTA LÍNEA ES CRÍTICA
    )

    # Crear modelo incremental
    model = factory.get_model(args["model_name"], args)

    # Curvas de accuracy por task
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}

    # Matrices de accuracy por clase (PyCIL original)
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager.nb_tasks):

        # Log de parámetros del modelo
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))

        # Entrenamiento incremental del task
        model.incremental_train(data_manager)

        # Evaluación del task
        cnn_accy, nme_accy = model.eval_task()

        # Post-procesado del modelo (PyCIL original)
        model.after_task()

        if nme_accy is not None:

            # Log de accuracies agrupadas
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            # Extraer claves tipo "00-09" para matriz CNN
            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            # Extraer claves tipo "00-09" para matriz NME
            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_keys_sorted = sorted(nme_keys)
            nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
            nme_matrix.append(nme_values)

            # Registrar top‑1 (siempre existe)
            cnn_curve["top1"].append(cnn_accy["top1"])
            nme_curve["top1"].append(nme_accy["top1"])

            # Registrar top‑5 SOLO si topk >= 5 y existe en el diccionario
            if args.get("topk", 1) >= 5 and "top5" in cnn_accy:
                cnn_curve["top5"].append(cnn_accy["top5"])
            if args.get("topk", 1) >= 5 and "top5" in nme_accy:
                nme_curve["top5"].append(nme_accy["top5"])

            # Logs de curvas
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            # Impresión de accuracy media por task
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"]) / len(nme_curve["top1"]))

        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            # Matriz CNN
            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            # Registrar top‑1
            cnn_curve["top1"].append(cnn_accy["top1"])

            # Registrar top‑5 si existe
            if args.get("topk", 1) >= 5 and "top5" in cnn_accy:
                cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))

    forgetting = 0.0  # valor por defecto

    # ===== MATRIZ CNN =====
    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])

        # Rellenar matriz triangular
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)

        # Transponer (PyCIL original)
        np_acctable = np_acctable.T

        # Cálculo original de forgetting
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])

        print('Accuracy Matrix (CNN):')
        print(np_acctable)
        print('Forgetting (CNN):', forgetting)
        logging.info('Forgetting (CNN): {}'.format(forgetting))

    # ===== MATRIZ NME =====
    if len(nme_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])

        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)

        np_acctable = np_acctable.T

        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])

        print('Accuracy Matrix (NME):')
        print(np_acctable)
        print('Forgetting (NME):', forgetting)
        logging.info('Forgetting (NME): {}'.format(forgetting))

    # Accuracy media final (la que va al CSV)
    avg_acc_final = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])

    print("\n=== MÉTRICAS FINALES ===")
    print("Average Accuracy (FINAL):", avg_acc_final)
    print("Forgetting (FINAL):", forgetting)

    logging.info("Average Accuracy (FINAL): {}".format(avg_acc_final))
    logging.info("Forgetting (FINAL): {}".format(forgetting))

    # Devolver métricas a Experiments.py
    return avg_acc_final, forgetting


def _set_device(args):
    """
    Normaliza args['device'] y lo convierte en una lista de objetos torch.device.
    Acepta entradas de varios tipos: int, str ('cpu','cuda:0','0'), -1, o torch.device.
    Si CUDA no esta disponible, cualquier intento de usar cuda se redirige a CPU.
    """
    device_type = args.get("device", [])
    cuda_ok = torch.cuda.is_available()
    gpus = []

    # asegurar iterable
    if not isinstance(device_type, (list, tuple)):
        devices = [device_type]
    else:
        devices = list(device_type)

    for device in devices:
        dev = None
        # ya es torch.device
        if isinstance(device, torch.device):
            if device.type == 'cuda' and not cuda_ok:
                logging.warning('CUDA no disponible, usando CPU en su lugar')
                dev = torch.device('cpu')
            else:
                dev = device
        else:
            # caso int
            if isinstance(device, int):
                if device == -1:
                    dev = torch.device('cpu')
                else:
                    dev = torch.device(f'cuda:{device}') if cuda_ok else torch.device('cpu')
            # caso string
            elif isinstance(device, str):
                s = device.strip().lower()
                if s in ('cpu', 'none'):
                    dev = torch.device('cpu')
                elif s.isdigit():
                    idx = int(s)
                    dev = torch.device(f'cuda:{idx}') if cuda_ok else torch.device('cpu')
                else:
                    # intentar interpretar 'cuda:0' u otras formas
                    try:
                        tmp = torch.device(s)
                        if tmp.type == 'cuda' and not cuda_ok:
                            logging.warning('CUDA no disponible, usando CPU en su lugar')
                            dev = torch.device('cpu')
                        else:
                            dev = tmp
                    except Exception:
                        # fallback seguro
                        dev = torch.device('cpu')
            else:
                # intentar convertir a string y crear device
                try:
                    tmp = torch.device(str(device))
                    if tmp.type == 'cuda' and not cuda_ok:
                        dev = torch.device('cpu')
                    else:
                        dev = tmp
                except Exception:
                    dev = torch.device('cpu')

        gpus.append(dev)

    args['device'] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
