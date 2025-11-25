import os
from PIL import Image

def check_image_modes(folder):
    """
    Recorre todas las imágenes .png de una carpeta y muestra su modo (RGB, L, etc.)
    """
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    if not files:
        print(f"No se encontraron imágenes en {folder}")
        return

    for file in files:
        img_path = os.path.join(folder, file)
        try:
            img = Image.open(img_path)
            print(f"{file}: {img.mode}")  # Muestra el modo de la imagen
            if img.mode == "RGB":
                print("   → La imagen está en color (RGB).")
            elif img.mode == "L":
                print("   → La imagen está en blanco y negro (grayscale).")
            else:
                print(f"   → La imagen está en modo {img.mode}.")
        except Exception as e:
            print(f"Error al abrir {file}: {e}")

if __name__ == "__main__":
    # Cambia esta ruta por la carpeta que quieras revisar
    folder = "/home/victoria/PycharmProjects/PyCIL2/data/Results/Gyro_Conversion/Test_1/class_1"
    check_image_modes(folder)
