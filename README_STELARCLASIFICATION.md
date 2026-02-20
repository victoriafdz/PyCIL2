README - PyCIL: Uso para la conversion estelar y experimentos
=============================================================

Este documento explica de forma practica y concisa como funciona `Experiments.py`
en este repositorio adaptado para generar datasets a partir de los ficheros
raw, ampliar esos datasets, generar imagenes y ejecutar experimentos de
clasificacion incremental usando los modelos existentes.

Aviso: los nombres de ficheros y rutinas siguen la estructura del proyecto.
Los comandos de ejemplo usan rutas relativas al directorio raiz del repo.

Resumen rapido
--------------
- `Experiments.py` es el script maestro que crea las configuraciones, llama a
  funciones de preprocesado (dataset, ampliacion, generacion de imagenes) y
  ejecuta los experimentos (entrenamiento) usando `trainer._train`.
- El flujo principal es:
  1) Preparar/crear CSV(s) y carpetas de imagenes usando utilidades en `utils/`.
  2) Generar un conjunto de configuraciones JSON en `exps/` (una por experimento).
  3) Llamar a `trainer._train(config)` para cada configuracion para realizar el
     entrenamiento incremental y registrar resultados.

Como usar (forma general)
-------------------------
1) Preparar el entorno Python con dependencias: torch (con o sin CUDA), numpy,
   pandas, torchvision, pillow, scikit-learn, joblib, matplotlib, etc.

2) Generar los CSV(s) necesarios si no existen. Hay utilidades en `utils/`:
   - `dataset.py` / `dataset_ampliado.py` / `dataset_moredata*.py` generan CSVs
     con distintas variaciones (base, ampliada en columnas, ampliada en filas).

3) Generar imagenes (si tu flujo usa imagenes) llamando a `utils/Img_tab.py`.
   `Img_tab` transforma tablas (CSV) en imagenes y las ordena en carpetas por clase.

4) Ejecutar `Experiments.py` (usa las configuraciones en `exps/` o crea las suyas):
   ```bash
   python Experiments.py
   ```
   Esto crea archivos JSON en `exps/` y ejecuta training para cada combinacion.

Descripcion de ficheros claves
------------------------------
A continuacion se describen los ficheros .py mas relevantes que participan en
el flujo cuando ejecutas `Experiments.py`.

- `Experiments.py`
  - Papel: archivo maestro que orquesta todo el proceso de experimentacion.
  - Funcionamiento principal:
    * Define rutas y parametros globales (DATA_PATH, CSV_PATH, MODELS, NETWORKS, etc.).
    * Decide dinamicamente `DEVICE` segun disponibilidad de CUDA en el interprete
      que ejecuta el script.
    * Contiene `make_config` y `make_config3` para crear diccionarios de
      configuracion que luego se guardan como JSON en `exps/`.
    * Recorre modelos y backbones (MODELS x NETWORKS) y llama a `run_experiment`
      para cada configuracion.
    * `run_experiment` llama a `trainer._train(config)` y guarda los resultados
      en `data/summary.csv`.
  - Notas importantes:
    * `DEVICE` se resuelve al inicio: si ejecutas el script con un Python que
      tiene PyTorch con CUDA activa, usara GPU 0; si no, usara CPU.
    * Si quieres forzar un dispositivo distinto, modifica `DEVICE` o las
      configuraciones JSON en `exps/`.

- `trainer.py`
  - Papel: gestiona el flujo de entrenamiento incremental por experimento.
  - Funciones clave:
    * `_set_device(args)`: normaliza y convierte `args['device']` a objetos
      `torch.device` validos, teniendo en cuenta si CUDA esta disponible.
    * `_set_random()`: fija semillas y parametros de cuDNN para reproducibilidad.
    * `_train(args)`: punto central que prepara logs, crea `DataManager`, crea el
      modelo incremental desde `utils.factory` y ejecuta el bucle de tasks
      (model.incremental_train, model.eval_task, model.after_task).
  - Salida: guarda logs, summary, y devuelve metricas de accuracy / forgetting.

- `utils/data.py` y `utils/data_manager.py`
  - `utils/data.py` encapsula la lectura y preprocesado de datasets (CIFAR,
    ImageNet, o CSVs genericos). Contiene la logica de descarga/validacion.
  - `utils/data_manager.py` es la capa que ofrece al resto del codigo
    iteradores/datasets/loader por task. Construye `Dataset` y `DataLoader` para
    entrenamiento y test. Sus metodos mas usados son `get_dataset(...)` y
    funciones internas que devuelven `nb_tasks`, tamanos por task, etc.
  - Notas: si ves errores relacionados con rutas o archivos no encontrados,
    revisa `args['data_path']` y la estructura de carpetas de imagenes.

- `utils/dataset_ampliado.py` y `utils/dataset_moredata_ampliado.py`
  - Objetivo: generar CSVs con la representacion tabular que luego se convertira
    a imagenes por `Img_tab`.
  - `dataset_ampliado.py`: replica las columnas base hasta conseguir 1024
    columnas (estructura usada por `Img_tab`). Genera `gyro_tot_v20180801_export.csv`.
  - `dataset_moredata_ampliado.py`: crea filas adicionales (muestreo por
    incertidumbres) y luego aplica la replicacion de columnas para producir
    un CSV con 1024 columnas. Archivo de salida: `gyro_tot_v20180801_export_moredata_ampliado.csv`.
  - Ambos ficheros ofrecen funciones `get_dataset(...)` o nombres similares que
    aceptan `save_csv=True` y `num_clases`.

- `utils/Img_tab.py` y `utils/IGTD_Functions.py`
  - `Img_tab.py`:
    * Toma un CSV y genera una imagen por fila usando metodos de distribucion
      de features. Organiza las imagenes en carpetas por clase.
    * Parametros importantes: `csv_path`, `result_dir_base`, `num_row`, `num_col`.
  - `IGTD_Functions.py`:
    * Contiene funciones para transformar tablas en imagenes, calcular rankings
      de features e indices de transformacion.
    * Parte del codigo puede ser costosa en tiempo (algoritmos O(n^2)); hay
      mensajes de progreso y puntos donde puede interrumpirse manualmente.

- `utils/dataset.py` (version base)
  - Crea el CSV base desde el TXT original (`gyro_tot_v20180801.txt`) con las
    8 columnas base que interesan: ['M','R','Teff','L','Meta','logg','Prot','Age']
    y la columna `class_M` (clasificacion por masa).
  - Esta funcion se usa cuando quieres solo la version simple del dataset.

- `models/*.py` (por ejemplo `models/finetune.py`, `models/icarl.py`, etc.)
  - Cada fichero define una clase `BaseLearner` derivada o una implementacion
    del algoritmo incremental.
  - `models/finetune.py` es el ejemplo basico: carga el `IncrementalNet`
    (archivo `utils/inc_net.py`), prepara DataLoaders, mueve la red al device,
    entrena y evalua.
  - Notas sobre `device` y CUDA:
    * El proyecto ahora normaliza devices en `trainer._set_device` y en la
      inicializacion de `BaseLearner` para que no falle si ejecutas con un
      interprete que no tiene PyTorch compilado con CUDA.

Flujo completo de ejecucion (paso a paso)
----------------------------------------
1) Ejecutas `Experiments.py` con el interprete Python deseado:
   - Si quieres GPU, usa el python del entorno que tiene PyTorch+CUDA.
   - Si quieres CPU, ejecuta con el entorno que no tenga CUDA o fuerza DEVICE = ['cpu'].

2) `Experiments.py` decide DEVICE segun `torch.cuda.is_available()` y fija rutas.
   Luego genera JSONs de configuracion para cada combinacion model x network y
   lanza `run_experiment` para cada uno.

3) `run_experiment` llama a `trainer._train(config)`.

4) `trainer._train`:
   - llama a `_set_device` y `_set_random`.
   - crea `DataManager` pasandole `data_path` (rutas a imagenes) y `csv_path`.
   - crea el modelo incremental con `factory.get_model(model_name, args)`.
   - itera sobre tasks: `model.incremental_train(data_manager)` y `model.eval_task()`.
   - guarda las metricas en `data/summary.csv`.

5) `model.incremental_train` (en `models/<algo>.py`):
   - actualiza la cabeza final para las nuevas clases con `update_fc()`.
   - crea DataLoaders usando `data_manager.get_dataset` para train y test.
   - llama internamente a rutinas de entrenamiento (`_init_train` / `_update_representation`).
   - usa `self._device` para mover tensores y modelos.

Puntos de control y resolucion de problemas
-------------------------------------------
- Error tipo "Torch not compiled with CUDA enabled": significa que el
  python con el que ejecutas no tiene PyTorch con CUDA; soluciona ejecutando
  con el python del entorno que instalaste con CUDA (o instala PyTorch-CUDA
  en el entorno actual). 

- Error tipo "Numpy is not available" o warnings de ABI mismatch: reinstala
  numpy en el entorno donde ejecutas, preferiblemente con conda-forge:
  ```bash
  conda install -n <env> -c conda-forge numpy pandas -y
  ```

- Si el DataLoader falla en workers (RuntimeError en worker): prueba a
  reducir `num_workers` a 0 en los DataLoaders (esto evita procesos hijos y
  facilita el debug), o arregla la instalacion de dependencias en el entorno.

- Problemas con CSVs (faltan columnas o valores NA): revisa las funciones
  `utils/dataset*.py` para entender que columnas generan y cuales son
  obligatorias; valida el TXT original `data/gyro_tot_v20180801.txt`.

Consejos practicos
------------------
- Para reproducibilidad, preferible ejecutar en un entorno conda limpio y
  con paquetes instalados desde conda-forge o los canales oficiales de PyTorch.
- Si vas a usar GPU, instala PyTorch con la version de CUDA compatible con
  tus drivers (nvidia-smi muestra Driver Version y CUDA Version).
- Ejecuta primero las utilidades de generacion de CSV e imagenes y comprueba
  manualmente que los CSV esten correctos antes de lanzar los experimentos.



