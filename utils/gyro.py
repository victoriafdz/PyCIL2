import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

class GyroIData:
    def __init__(self, data_path="./Results/Gyro_Conversion/Test_1", batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size

        # PyCIL espera este flag
        self.use_path = True

        # Variables para datasets y dataloaders
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None

        # Lo que DataManager espera
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

        # Número de clases y orden
        self.nb_classes = 0
        self.class_order = []

        # Transformaciones comunes (como lista)
        self.common_trsf = [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]

        # Transformaciones para entrenamiento (como lista)
        self.train_trsf = [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),

        ]

        # Transformaciones para test (como lista)
        self.test_trsf = [
            transforms.Resize((32, 32)),

        ]

    def download_data(self):
        self._load()

    def _load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        # Cargar datasets usando Compose al aplicar
        train_dataset = ImageFolder(root=self.data_path, transform=transforms.Compose(self.train_trsf))
        test_dataset = ImageFolder(root=self.data_path, transform=transforms.Compose(self.test_trsf))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Crear DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Extraer rutas y etiquetas
        self.train_data = [path for (path, _) in train_dataset.samples]
        self.train_targets = [label for (_, label) in train_dataset.samples]

        self.test_data = [path for (path, _) in test_dataset.samples]
        self.test_targets = [label for (_, label) in test_dataset.samples]

        # Número de clases y orden
        self.nb_classes = len(train_dataset.classes)
        self.class_order = list(range(self.nb_classes))

        # Debug opcional
        print("Clases detectadas:", train_dataset.classes)
        print("Mapeo de clases:", train_dataset.class_to_idx)
        print("Número de clases:", self.nb_classes)
        print("class_order:", self.class_order)