"""
Mirror script for notebooks/optuna/kaggle_dataset.ipynb

Auto-generated from the notebook code cells.
See the corresponding tutorial in docs/tutorials/ for context.
Original notebook: 0 markdown cells, 2 code cells.
"""

import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("arjunbhasin2013/ccdata")
# print("Path to dataset files:", path)
dataset_folder = path
csv_file = None
for file in os.listdir(dataset_folder):
    if file.endswith(".csv"):
        csv_file = os.path.join(dataset_folder, file)
        break

if not csv_file:
    print("No se encontró un archivo CSV.")
    exit()

df = pd.read_csv(csv_file)
df.head()

# %%

