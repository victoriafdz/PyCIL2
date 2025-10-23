import pandas as pd
import os
from typing import Optional

# the script imports the data from a data file with the information of the stars
def get_dataset(save_csv: bool = True, csv_path: Optional[str] = None):
    """Load and clean the star dataset.

    Args:
        save_csv: if True, save the cleaned DataFrame to `csv_path` (or a default path).
        csv_path: destination CSV path. If None and `save_csv` is True, a default file
            next to the data file will be used.

    Returns:
        pandas.DataFrame: the cleaned and filtered dataset.
    """
    # locate the input data file relative to this module
    data_file = os.path.join(os.path.dirname(__file__), 'gyro_tot_v20180801.txt')
    data = pd.read_csv(data_file, sep="\t", header=0)
    df = data[['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age', 'eAge1', 'eAge2', 'class']]
    # age limits, only for graphics
    df['low_age'] = df.Age - df.eAge1
    df['high_age'] = df.Age + df.eAge2
    # clean NA values
    df.dropna(inplace=True, axis=0)
    # filter the datasets because of the physics behind gyrochronology
    df = df.loc[(df['class'] == 'MS') & (df['M'] < 2) & (df['M'] > 0.7) & (df['Prot'] < 50)]
    # sort the dataframe by age
    df = df.sort_values(by=['Age'])
    # chose target variable: age
    # y = np.array(df['Age'])
    # selection of the data to be used
    # X = np.array(df[['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot']])

    # optionally save to csv
    if save_csv:
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), 'gyro_tot_v20180801_export.csv')
        # ensure parent dir exists (usually it's the datasets folder which does exist)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        # create a copy for export and drop auxiliary columns requested by the user
        export_df = df.drop(columns=['low_age', 'high_age', 'eAge1', 'eAge2', 'class'], errors='ignore')
        export_df.to_csv(csv_path, index=False)

    return df

get_dataset()