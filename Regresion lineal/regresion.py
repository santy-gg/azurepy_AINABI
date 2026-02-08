import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import json
import os
import logging
import sys

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rutas relativas
# Asegurarse de que el modelo se guarde en 'azurepy/' un nivel arriba
ruta_modelo = os.path.join(os.path.dirname(os.path.dirname(__file__)), "azurepy", "modelo_desempenio_futuro.pkl")
# Carga el CSV generado por 'generar_synthetic_training_data.py'
ruta_csv_training = os.path.join(os.path.dirname(__file__), "synthetic_training_data.csv")

def entrenar_modelo():
    """
    Entrena un modelo Random Forest usando datos de entrenamiento sintéticos.
    """
    try:
        logging.info(f"Cargando datos de entrenamiento desde: {ruta_csv_training}")
        df = pd.read_csv(ruta_csv_training, encoding="utf-8")
        logging.info(f"Datos cargados. Filas: {len(df)}")

        # Mapeo de categorías a numéricos si aún no lo están (deberían estarlo por el generador)
        mapa_jerarquia = {'trainee': 0, 'junior': 1, 'senior': 2}
        mapa_desempenio = {'bajo': 0, 'medio': 1, 'alto': 2}
        
        # Solo mapear si la columna existe y no está ya en formato numérico
        if 'jerarquia' in df.columns and df['jerarquia'].dtype == 'object':
            df['jerarquia'] = df['jerarquia'].map(mapa_jerarquia)
        if 'desempenio' in df.columns and df['desempenio'].dtype == 'object':
            df['desempenio'] = df['desempenio'].map(mapa_desempenio)

        # One-hot encoding para la columna 'area'
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Verificar si 'area' existe y si ohe debe aplicarse
        if 'area' in df.columns:
            area_encoded = ohe.fit_transform(df[['area']])
            area_encoded_df = pd.DataFrame(area_encoded, columns=ohe.get_feature_names_out(['area']), index=df.index)
            df_final = pd.concat([df.drop(['area'], axis=1), area_encoded_df], axis=1)
            logging.info("One-Hot Encoding aplicado a 'area'.")
        else:
            df_final = df.copy() # Si no hay 'area', usar el DataFrame tal cual
            logging.warning("Columna 'area' no encontrada en el CSV de entrenamiento. Saltando OHE para 'area'.")


        # Separar features y target
        # Asegurarse de que las columnas a dropear existan
        cols_to_drop = ['nombre', 'desempenio_futuro']
        existing_cols_to_drop = [col for col in cols_to_drop if col in df_final.columns]
        X = df_final.drop(columns=existing_cols_to_drop, errors='ignore')
        
        if 'desempenio_futuro' not in df_final.columns:
            raise KeyError("La columna 'desempenio_futuro' es necesaria para el entrenamiento y no se encontró.")
        y = df_final['desempenio_futuro']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        logging.info(f"Datos divididos en entrenamiento ({len(X_train)} filas) y prueba ({len(X_test)} filas).")

        # Escalado
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Datos escalados.")

        # Entrenar Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        logging.info("Entrenando modelo RandomForestClassifier...")
        model.fit(X_train_scaled, y_train)
        logging.info("Modelo entrenado.")

        # Evaluar
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        reporte = classification_report(y_test, y_pred, output_dict=True)
        logging.info(f"Precisión del modelo: {acc * 100:.2f}%")

        # Guardar modelo
        # Asegúrate de que la carpeta 'azurepy' exista antes de guardar
        os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
        with open(ruta_modelo, 'wb') as archivo:
            pickle.dump({
                'modelo': model,
                'columnas': list(X.columns), # Guardar las columnas utilizadas para el entrenamiento
                'encoder': ohe,
                'scaler': scaler
            }, archivo)
        logging.info(f"Modelo y preprocesadores guardados en: {ruta_modelo}")

        # Salida
        resultados = {
            "accuracy": f"{acc * 100:.2f}%",
            "precision_por_clase": {
                str(k): f"{v['precision'] * 100:.2f}%" for k, v in reporte.items() if k in ['0', '1', '2']
            },
            "status": "Modelo entrenado y guardado"
        }
        return json.dumps(resultados, ensure_ascii=False)

    except FileNotFoundError as e:
        logging.error(f"❌ Error (FileNotFoundError): No se encontró el archivo: {e.filename}", exc_info=True)
        return json.dumps({"error": f"No se encontró el archivo: {e.filename}"})
    except KeyError as e:
        logging.error(f"❌ Error (KeyError): Falta la columna '{e}' al procesar los datos.", exc_info=True)
        return json.dumps({"error": f"Error de clave: falta la columna '{e}' en los datos."})
    except Exception as e:
        logging.error(f"❌ Ocurrió un error inesperado durante el entrenamiento del modelo: {e}", exc_info=True)
        return json.dumps({"error": f"Ocurrió un error inesperado: {e}"})

if __name__ == '__main__':
    logging.info("Ejecutando entrenamiento del modelo desde main de regresion.py")
    resultado = entrenar_modelo()
    print(resultado)
