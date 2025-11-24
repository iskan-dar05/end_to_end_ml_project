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
plot_dir = Path('plot_images')
dst_folder.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)
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


ax = df['Class'].value_counts().plot(kind="bar")

plt.title("Fraud vs Non-Fraud Count")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.savefig(plot_dir / 'barplot.png')

plt.close()


X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

sm = SMOTE(random_state=42)




