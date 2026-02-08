#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import pickle
import json
import os
import logging

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ruta relativa al archivo actual
ruta_dataset = os.path.join(os.path.dirname(__file__), "dataset_empleados_kmeans.xlsx")
logging.info(f"Intentando cargar dataset desde: {ruta_dataset}")  # Log

try:
    dataset = pd.read_excel(ruta_dataset)
except FileNotFoundError:
    logging.error(f"No se encontró el archivo: {ruta_dataset}")
    print(json.dumps({"error": f"No se encontró el archivo: {ruta_dataset}"}))
    exit()  # Importante: Salir del script si el archivo no existe

# --- Código original de Ceci (sin caracteres especiales) ---
codificador = OneHotEncoder()
codificacion = codificador.fit_transform(dataset[["Rendimiento ACTUAL"]])
nuevas_cols = pd.DataFrame(codificacion.toarray(), columns=codificador.get_feature_names_out(["Rendimiento ACTUAL"]))
dataset = pd.concat([dataset, nuevas_cols], axis="columns")
dataset = dataset.drop("Rendimiento ACTUAL", axis=1)

columnas_numericas = dataset.columns.difference(["Nombre", "Ciclo"]).tolist()
dataset_agrupado_por_Nombre = dataset.groupby("Nombre")[columnas_numericas].sum().reset_index()

escalador = MinMaxScaler()
columnas_a_escalar = [
    "Ausencias Injustificadas", "Llegadas tarde",
    "Rendimiento ACTUAL_Alto", "Rendimiento ACTUAL_Bajo",
    "Rendimiento ACTUAL_Medio", "Salidas tempranas"
]
dataset_agrupado_por_Nombre_escalado = dataset_agrupado_por_Nombre.copy()
dataset_agrupado_por_Nombre_escalado[columnas_a_escalar] = escalador.fit_transform(
    dataset_agrupado_por_Nombre_escalado[columnas_a_escalar]
)

n_clusters = 3
X = dataset_agrupado_por_Nombre_escalado.drop(['Nombre'], axis=1)
kmeans = KMeans(n_clusters=n_clusters, random_state=12)
dataset_agrupado_por_Nombre_escalado['Cluster'] = kmeans.fit_predict(X)

dataset_agrupado_por_Nombre["Cluster"] = dataset_agrupado_por_Nombre_escalado["Cluster"]
dataset_agrupado_por_Nombre["Probabilidad de Rotacion"] = dataset_agrupado_por_Nombre["Cluster"].map({
    2: "ALTA",
    0: "BAJA",
    1: "MEDIA"
})

# Salida JSON
if __name__ == "__main__":
    resultados = {
        "data": dataset_agrupado_por_Nombre.to_dict(orient="records"),
        "clusters": n_clusters
    }
    print(json.dumps(resultados))
