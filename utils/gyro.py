# python
import os
import numpy as np
import pandas as pd
from torchvision import transforms

class GyroIData:
    """
    Minimal CSV adapter for the 'gyro' dataset.

    Attributes expected by DataManager:
      - download_data()
      - train_data, train_targets
      - test_data, test_targets
      - train_trsf, test_trsf, common_trsf
      - use_path (bool)
      - class_order (list)
    """
    def __init__(self, csv_path=None, label_col='age', n_bins=5, test_ratio=0.2, seed=1993, bin_edges=None):
        self.csv_path = csv_path or os.path.join('data', 'gyro_tot_v20180801_export.csv')
        self.label_col = label_col
        self.n_bins = n_bins
        self.bin_edges = bin_edges
        self.test_ratio = test_ratio
        self.seed = seed

        # placeholders
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None
        self.train_trsf = [transforms.Lambda(lambda x: x)]
        self.test_trsf = [transforms.Lambda(lambda x: x)]
        self.common_trsf = []
        self.use_path = False
        self.class_order = None
        self._loaded = False

    def age_to_bins(self, values, n_bins=5, bin_edges=None):
        vals = np.asarray(values, dtype=float)
        if bin_edges is None:
            bin_edges = np.linspace(np.nanmin(vals), np.nanmax(vals), n_bins + 1)
        labels = pd.cut(vals, bins=bin_edges, labels=False, include_lowest=True)
        return labels.astype(int), bin_edges

    def download_data(self):
        # For local CSV dataset this is a no-op loader.
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if self.label_col not in df.columns:
            raise ValueError(f"Label column `{self.label_col}` not found in {self.csv_path}")

        # select numeric columns (excluding label) as features; fallback to converting others
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != self.label_col]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c != self.label_col]
            df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        X = df[feature_cols].values.astype(float)
        y, edges = self.age_to_bins(df[self.label_col].values, n_bins=self.n_bins, bin_edges=self.bin_edges)

        # simple deterministic split
        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(y))
        rng.shuffle(indices)
        split = int(len(indices) * (1 - self.test_ratio))
        train_idx, test_idx = indices[:split], indices[split:]

        self.train_data = X[train_idx]
        self.train_targets = y[train_idx]
        self.test_data = X[test_idx]
        self.test_targets = y[test_idx]

        # class order: natural order of integer labels
        unique_labels = np.unique(y)
        self.class_order = unique_labels.tolist()
