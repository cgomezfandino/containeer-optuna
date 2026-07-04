"""
Mirror script for notebooks/customer_personality_analysis/model.ipynb

Auto-generated from the notebook code cells.
See the corresponding tutorial in docs/tutorials/ for context.
Original notebook: 0 markdown cells, 11 code cells.
"""

import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Descargar el dataset
path = kagglehub.dataset_download("imakash3011/customer-personality-analysis")
print("Path to dataset files:", path)

# Buscar y leer el archivo CSV
dataset_folder = path
csv_file = None
for file in os.listdir(dataset_folder):
    if file.endswith(".csv"):
        csv_file = os.path.join(dataset_folder, file)
        break

if not csv_file:
    print("No se encontró un archivo CSV.")
    exit()

# %%
# Leer el dataset
df = pd.read_csv(csv_file, sep="\t")    
# Limpieza: Eliminar filas con valores nulos y columnas no relevantes
df = df.dropna()
df = df.drop(['ID', 'Dt_Customer'], axis=1, errors='ignore')  # Eliminar ID y fecha
df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=False, dtype=int)
print(df.columns)
df.head()

# %%
df.describe()

# %%
# correlation_matrix = df[col_features].corr()
correlation_matrix = df.drop(['Z_CostContact', 'Z_Revenue'], axis=1).corr()
# Plot the heatmap
plt.figure(figsize=(25, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5) ##cmap=["coolwarm", "viridis", "plasma", "inferno", "magma", "cividis"]

plt.title("Feature Correlation Matrix")
plt.show()

# %%
# Supón que tus variables numéricas están en el DataFrame 'df'
# Puedes especificar solo ciertas columnas si quieres: df = df[['var1', 'var2', 'var3']]
cols = ['Year_Birth', 'Income', 'Recency', 'MntWines',
       'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']


# Crea un PairGrid con tus variables
g = sns.PairGrid(df[cols], diag_sharey=False)

# Diagonal: distribución univariada (histograma o kde)
g.map_diag(sns.kdeplot , fill=True, color='skyblue')   # Usa sns.kdeplot si prefieres densidad, sns.histplot

# Fuera de la diagonal: scatterplot + línea de tendencia
g.map_offdiag(sns.regplot, scatter_kws={'alpha':0.6}, line_kws={"color":"red"})

# Títulos y espacio
plt.suptitle("Matriz de dispersión con líneas de tendencia", y=1.01, fontsize=16)
plt.tight_layout()
plt.show()

# %%

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_scaled

# %%

# Aplicar PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
print("Varianza total explicada:", sum(pca.explained_variance_ratio_))

# %%

# Método del codo para elegir k
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel('Número de clústeres (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.show()

# Aplicar K-Means (suponiendo k=3 basado en el codo)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)
df['Cluster'] = clusters

# Visualizar los clústeres
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Segmentación de Clientes con PCA y K-Means')
plt.colorbar(label='Cluster')
plt.show()

# %%
# Estadísticas por clúster
print("Estadísticas por clúster:")
df.groupby('Cluster').mean()

# %%
from sklearn.metrics import silhouette_score
print("Silhouette Score:", round(silhouette_score(X_pca, clusters), 2))

# %%
# Seleccionar variables clave para los boxplots
key_columns = ['Income', 'MntWines', 'MntFruits', 'NumWebPurchases', 'NumStorePurchases']

# Configurar el estilo de los gráficos
sns.set_theme(style="whitegrid")

# Crear boxplots para cada variable por clúster
plt.figure(figsize=(15, 10))
for i, column in enumerate(key_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Cluster', y=column, data=df, hue='Cluster', palette='viridis', legend=False)
    plt.title(f'Distribución de {column} por Clúster')
    plt.xlabel('Cluster')
    plt.ylabel(column)
plt.tight_layout()
plt.show()
