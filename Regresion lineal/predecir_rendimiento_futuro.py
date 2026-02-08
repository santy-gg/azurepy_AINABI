import pandas as pd
import pickle
import sys
import json
import os
import logging

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predecir_rendimiento_futuro(archivo_csv):
    """
    Realiza la predicción del desempeño futuro usando un modelo Random Forest previamente entrenado.
    El modelo ya fue entrenado con datos que incorporan las reglas del analista y ruido.
    """
    try:
        # La ruta del modelo debe ser la misma donde regresion.py lo guarda
        modelo_guardado_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "azurepy", "modelo_desempenio_futuro.pkl")
        
        logging.info(f"Cargando modelo desde: {modelo_guardado_path}")
        with open(modelo_guardado_path, 'rb') as archivo_cargado:
            datos_cargados = pickle.load(archivo_cargado)

        modelo_cargado = datos_cargados['modelo']
        columnas_entrenamiento = datos_cargados['columnas']
        ohe = datos_cargados['encoder']
        scaler = datos_cargados['scaler']
        logging.info("Modelo y preprocesadores cargados exitosamente.")

        nuevos_df = pd.read_csv(archivo_csv, encoding="utf-8")
        logging.info(f"CSV de predicción cargado desde: {archivo_csv}. Filas: {len(nuevos_df)}")

        # --- Preprocesamiento (igual que en el entrenamiento) ---
        # Mapeo de categorías a numéricos
        mapa_jerarquia = {'trainee': 0, 'junior': 1, 'senior': 2}
        mapa_desempenio = {'bajo': 0, 'medio': 1, 'alto': 2}
        
        if 'jerarquia' in nuevos_df.columns:
            nuevos_df['jerarquia'] = nuevos_df['jerarquia'].map(mapa_jerarquia).fillna(nuevos_df['jerarquia'])
            logging.info("Columna 'jerarquia' mapeada.")
        
        if 'desempenio' in nuevos_df.columns:
            nuevos_df['desempenio'] = nuevos_df['desempenio'].map(mapa_desempenio).fillna(nuevos_df['desempenio'])
            logging.info("Columna 'desempenio' mapeada.")

        # Aplicar One-Hot Encoding si 'area' está presente y el encoder fue cargado
        if 'area' in nuevos_df.columns and ohe:
            area_encoded = ohe.transform(nuevos_df[['area']])
            area_encoded_df = pd.DataFrame(area_encoded, columns=ohe.get_feature_names_out(['area']), index=nuevos_df.index)
            df_final = pd.concat([nuevos_df.drop(['area'], axis=1)], axis=1) # No usar area_encoded_df directamente aquí aún
            # Asegurarse de que las columnas generadas por el OHE se añadan correctamente.
            # Necesitamos un DataFrame que contenga todas las columnas que el modelo espera.
            # La forma más segura es crear un nuevo DataFrame con las columnas del entrenamiento y rellenarlo.
            logging.info("One-Hot Encoding aplicado a 'area'.")
        else:
            df_final = nuevos_df.copy() # Si no hay 'area' o encoder, usar el DF original
            logging.warning("Columna 'area' no encontrada o OneHotEncoder no cargado. Saltando OHE para 'area'.")
        
        # Asegurarse de que X_nuevos contenga solo las columnas esperadas por el modelo
        # Esto es crucial para evitar errores si el CSV de predicción tiene columnas extra o faltantes
        x_nuevos = pd.DataFrame(columns=columnas_entrenamiento)
        for col in columnas_entrenamiento:
            if col in df_final.columns:
                # Si la columna existe en df_final (original o después de mapeo categórico)
                x_nuevos[col] = pd.to_numeric(df_final[col], errors='coerce') 
            elif col in ohe.get_feature_names_out(['area']) if 'area' in nuevos_df.columns and ohe else []:
                # Si es una columna de one-hot encoding y el 'area' original estaba presente
                # Necesitamos extraer el valor one-hot del 'area_encoded_df' si existe
                if 'area' in nuevos_df.columns and ohe:
                     # Buscar el índice correspondiente en 'nuevos_df'
                     idx = nuevos_df.index
                     # Extraer el valor de la columna one-hot del df generado en la OHE
                     x_nuevos[col] = area_encoded_df[col]
                else:
                    x_nuevos[col] = 0 # O un valor predeterminado si no se pudo mapear
            else:
                x_nuevos[col] = 0 # O un valor predeterminado si la columna no existe

        # Asegurarse de que el orden de las columnas sea el mismo que en el entrenamiento
        x_nuevos = x_nuevos[columnas_entrenamiento]
        
        # Escalar antes de predecir
        x_nuevos_scaled = scaler.transform(x_nuevos)
        logging.info("Datos de predicción escalados.")

        # --- Predicción ---
        predicciones_futuras_numericas = modelo_cargado.predict(x_nuevos_scaled)
        logging.info("Predicciones del modelo obtenidas.")

        # Mapear de las predicciones numéricas a etiquetas de texto para la salida final
        mapa_rendimiento_numerico_a_simbolico = {0: 'bajo', 1: 'medio', 2: 'alto'}
        nuevos_df["desempenio_futuro"] = [mapa_rendimiento_numerico_a_simbolico.get(p, p) for p in predicciones_futuras_numericas]
        
        # Retornar resultados
        # Asegurarse de que las columnas categóricas originales se muestren como texto si fueron mapeadas
        mapa_jerarquia_numerico_a_simbolico = {0: 'trainee', 1: 'junior', 2: 'senior'}
        mapa_desempenio_numerico_a_simbolico = {0: 'bajo', 1: 'medio', 2: 'alto'}

        if 'jerarquia' in nuevos_df.columns and nuevos_df['jerarquia'].dtype != 'object':
            nuevos_df['jerarquia'] = nuevos_df['jerarquia'].map(mapa_jerarquia_numerico_a_simbolico).fillna(nuevos_df['jerarquia'])
        if 'desempenio' in nuevos_df.columns and nuevos_df['desempenio'].dtype != 'object':
            nuevos_df['desempenio'] = nuevos_df['desempenio'].map(mapa_desempenio_numerico_a_simbolico).fillna(nuevos_df['desempenio'])

        resultados = nuevos_df.to_dict(orient="records")
        logging.info("Resultados de predicción preparados para retorno.")
        return json.dumps(resultados, ensure_ascii=False)

    except FileNotFoundError as e:
        logging.error(f"❌ Error (FileNotFoundError): No se encontró el archivo: {e.filename}", exc_info=True)
        return json.dumps({"error": f"No se encontró el archivo: {e.filename}"})
    except KeyError as e:
        logging.error(f"❌ Error (KeyError): Falta la columna '{e}' en los datos o en las columnas de entrenamiento.", exc_info=True)
        return json.dumps({"error": f"Error de clave: falta la columna '{e}' en los datos."})
    except Exception as e:
        logging.error(f"❌ Ocurrió un error inesperado al predecir rendimiento futuro: {e}", exc_info=True)
        return json.dumps({"error": f"Ocurrió un error inesperado: {e}"})

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Error: Se debe proporcionar la ruta al archivo CSV como argumento.")
        print(json.dumps({"error": "Uso: python predecir_rendimiento_futuro.py <archivo_csv_prediccion>"}))
        sys.exit(1)

    archivo_csv = sys.argv[1]
    logging.info(f"Iniciando predicción para CSV: {archivo_csv}")
    resultado = predecir_rendimiento_futuro(archivo_csv)
    print(resultado)
