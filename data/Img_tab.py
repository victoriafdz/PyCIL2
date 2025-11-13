import pandas as pd
import os
from IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation

# Parámetros de configuración
num_row = 30
num_col = 30
num = num_row * num_col
save_image_size = 3
max_step = 30000
val_step = 300

# Ruta al archivo CSV de gyro
data_path = '../data/gyro_tot_v20180801_export.csv'
data = pd.read_csv(data_path, sep=',', header=0, index_col=0, low_memory=False)

# Seleccionar las características con mayor variación
id = select_features_by_variation(data, variation_measure='var', num=num)
data = data.iloc[:, id]

# Normalizar los datos con min-max
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

# Primera ejecución: distancia Euclidiana + error absoluto
fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'
result_dir = '../Results/Gyro_Conversion/Test_1'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method,
               save_image_size, max_step, val_step, result_dir, error)

# Segunda ejecución: correlación de Pearson + distancia Manhattan + error cuadrático
fea_dist_method = 'Pearson'
image_dist_method = 'Manhattan'
error = 'squared'
norm_data = norm_data.iloc[:, :800]  # opcional: reducir dimensionalidad
result_dir = '../Results/Gyro_Conversion/Test_2'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method,
               save_image_size, max_step, val_step, result_dir, error)
