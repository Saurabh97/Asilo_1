import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Asilo_1.fl.trainers.tabular_logreg import TabularLogRegTrainer

class WESADData:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing WESAD CSV at {csv_path}")
        df = pd.read_csv(csv_path)
        # expected columns: ts,label,feat_0..feat_D
        y = df['label'].astype(int).to_numpy()
        X = df[[c for c in df.columns if c.startswith('feat_')]].to_numpy(dtype=float)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

    def make_trainer(self) -> TabularLogRegTrainer:
        return TabularLogRegTrainer(self.X_train, self.y_train, self.X_val, self.y_val)