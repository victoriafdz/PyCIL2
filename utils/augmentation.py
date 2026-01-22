import numpy as np
import pandas as pd
from typing import Tuple


def data_augmentation_with_uncertainties(X: pd.DataFrame, y: pd.DataFrame, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Augment data sampling uniformly within asymmetric error bars.

    X: DataFrame with feature columns and optional error columns named e<feature>1 and e<feature>2.
    y: DataFrame or Series with target and optional error columns [target, etarget1, etarget2].
    n_samples: number of synthetic samples to generate per original row. If 0, returns originals.

    Returns X_aug (N*n_samples, n_features) and y_aug (N*n_samples, 1)
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if not isinstance(y, (pd.DataFrame, pd.Series)):
        raise ValueError("y must be a pandas DataFrame or Series")

    cols = list(X.columns)
    # identify base features (exclude error columns of form e<name>1 / e<name>2)
    base_features = []
    for c in cols:
        if c.startswith('e') and (c.endswith('1') or c.endswith('2')):
            continue
        base_features.append(c)

    if isinstance(y, pd.Series):
        y = y.to_frame()

    def _strip_errors_from_X(df: pd.DataFrame):
        return df[[f for f in base_features if f in df.columns]].values

    if n_samples <= 0:
        X_out = _strip_errors_from_X(X)
        y_out = y.iloc[:, 0].values.reshape(-1, 1) if y.shape[1] >= 1 else y.values
        return X_out, y_out

    X_aug_list = []
    y_aug_list = []

    feature_error_info = {}
    for f in base_features:
        e1 = f"e{f}1"
        e2 = f"e{f}2"
        feature_error_info[f] = (e1 in X.columns and e2 in X.columns, e1, e2)

    for idx, row in X.reset_index(drop=True).iterrows():
        # handle y row safely
        try:
            y_row = y.loc[idx]
        except Exception:
            y_row = y.iloc[idx]
        has_y_err = False
        if y.shape[1] >= 3:
            has_y_err = True
            y_val = float(y_row.iloc[0])
            y_e1 = float(y_row.iloc[1])
            y_e2 = float(y_row.iloc[2])
        else:
            y_val = float(y_row.iloc[0])

        for s in range(n_samples):
            sample_feats = []
            for f in base_features:
                has_err, e1_name, e2_name = feature_error_info.get(f, (False, None, None))
                v = row[f] if f in row.index else None
                if pd.isna(v):
                    sample_val = v
                elif has_err:
                    e1 = row[e1_name]
                    e2 = row[e2_name]
                    if pd.isna(e1) or pd.isna(e2):
                        sample_val = v
                    else:
                        low = float(v) - float(e1)
                        high = float(v) + float(e2)
                        if low >= high:
                            sample_val = v
                        else:
                            sample_val = np.random.uniform(low, high)
                else:
                    sample_val = v
                sample_feats.append(sample_val)

            if has_y_err:
                if np.isnan(y_e1) or np.isnan(y_e2):
                    y_sample = y_val
                else:
                    y_sample = np.random.uniform(y_val - y_e1, y_val + y_e2)
            else:
                y_sample = y_val

            X_aug_list.append(sample_feats)
            y_aug_list.append(y_sample)

    X_aug = np.array(X_aug_list)
    y_aug = np.array(y_aug_list).reshape(-1, 1)
    return X_aug, y_aug

