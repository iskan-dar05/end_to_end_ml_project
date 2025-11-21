import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import kagglehub
import shutil
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


dst_folder = Path('dataset')
dst_folder.mkdir(parents=True, exist_ok=True)

path_str = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

path = Path(path_str)

csv_file = 'creditcard.csv'

csv_path = path / csv_file

if not csv_path.exists():
    print(f"Error: {csv_path} not found")

else:
    final_path = dst_folder / csv_file
    shutil.move(str(csv_path), str(final_path))
    print(f"Moving file from {csv_path} to {final_path}")
    



df = pd.read_csv(final_path)

print(df.head())

print(df.info())

print(df.describe())







