"""
MeanSTDCalculator.py
Script para calcular la media y desviación estándar de un dataset de imágenes.
- Funciona tanto para imágenes RGB como L (grayscale).
- Recorre recursivamente todas las subcarpetas.
- Devuelve valores listos para usar en transforms.Normalize().
"""

import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ======= CONFIGURACIÓN =======
IMG_DIR_RGB = "/home/victoria/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion/Test_1_RGB"
IMG_DIR_L   = "/home/victoria/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion/Test_1_L"

# Transformación: convierte a tensor [0,1]
to_tensor = transforms.ToTensor()

def calculate_mean_std(img_dir, mode="RGB"):
    """
    Calcula la media y desviación estándar de todas las imágenes en img_dir.
    mode = "RGB" → 3 canales
    mode = "L"   → 1 canal
    """
    channels = 3 if mode == "RGB" else 1
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    total_images = 0

    # Recorrer recursivamente todas las subcarpetas
    for root, _, files in os.walk(img_dir):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                img_path = os.path.join(root, fname)
                img = Image.open(img_path).convert(mode)
                tensor = to_tensor(img)  # shape: [C, H, W]

                # Acumular medias y std por canal
                mean += tensor.mean(dim=[1,2])
                std += tensor.std(dim=[1,2])
                total_images += 1

    if total_images > 0:
        mean /= total_images
        std /= total_images
        print(f"Resultados para {mode} en {img_dir}:")
        print("Mean:", mean.tolist())
        print("Std:", std.tolist())
    else:
        print(f"No se encontraron imágenes válidas en {img_dir}")

# ======= MAIN =======
if __name__ == "__main__":
    calculate_mean_std(IMG_DIR_RGB, mode="RGB")
    calculate_mean_std(IMG_DIR_L, mode="L")
