import os
from torchvision.datasets import ImageFolder
from torchvision import transforms

class GyroIData:
    def __init__(self, data_path="./Results/Gyro_Conversion/Test_1"):
        self.data_path = data_path

        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None
        self.use_path = True

        # Transformaciones
        self.common_trsf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        self.train_trsf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.test_trsf = self.common_trsf

        self.nb_classes = 0
        self.class_order = []

    def download_data(self):
        self._load()

    def _load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        train_dataset = ImageFolder(root=self.data_path, transform=self.train_trsf)
        test_dataset = ImageFolder(root=self.data_path, transform=self.test_trsf)

        # Extraer rutas y etiquetas
        self.train_data = [path for (path, _) in train_dataset.samples]
        self.train_targets = [label for (_, label) in train_dataset.samples]

        self.test_data = [path for (path, _) in test_dataset.samples]
        self.test_targets = [label for (_, label) in test_dataset.samples]

        # Aquí defines explícitamente el número de clases y el orden
        self.nb_classes = 2
        self.class_order = [0, 1]

        # Debug opcional
        print("Clases detectadas:", train_dataset.classes)
        print("Mapeo de clases:", train_dataset.class_to_idx)
        print("Etiquetas en train_targets:", set(self.train_targets))
        print("class_order:", self.class_order)
