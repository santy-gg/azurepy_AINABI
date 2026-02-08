import pandas as pd
import numpy as np
import json
import sys
import logging
import os
from collections import Counter

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clasificar_fila_con_ruido(fila, reglas, p_ruido=0.01): # <--- p_ruido AÚN MÁS REDUCIDO
    """
    Clasifica una fila basándose en las reglas proporcionadas y añade ruido controlado.
    """
    puntajes = []
    
    # Mapeo de categorías a numéricos para aplicar reglas internamente
    mapa_jerarquia_simbolica_a_numerico = {'trainee': 0, 'junior': 1, 'senior': 2}
    mapa_desempenio_simbolica_a_numerico = {'bajo': 0, 'medio': 1, 'alto': 2}

    for columna, rangos_clase in reglas.items():
        valor_fila = fila.get(columna)

        # Convertir valores categóricos de la fila a numéricos para aplicar reglas
        if columna == 'desempenio' and isinstance(valor_fila, str):
            valor_fila = mapa_desempenio_simbolica_a_numerico.get(valor_fila.lower(), valor_fila)
        elif columna == 'jerarquia' and isinstance(valor_fila, str):
            valor_fila = mapa_jerarquia_simbolica_a_numerico.get(valor_fila.lower(), valor_fila)

        if valor_fila is None:
            continue
        
        valor_fila_num = pd.to_numeric(valor_fila, errors='coerce')
        if pd.isna(valor_fila_num):
            continue

        # Aplicar lógica de reglas:
        if isinstance(rangos_clase, dict):
            # Caso 1: Reglas con formato {"1":[min, max]} para columnas numéricas
            if "1" in rangos_clase and isinstance(rangos_clase["1"], list) and len(rangos_clase["1"]) == 2:
                min_val, max_val = rangos_clase["1"]
                if min_val <= valor_fila_num <= max_val:
                    puntajes.append(1) # Tiende a "medio" (1) si cumple la regla
                elif valor_fila_num < min_val:
                    puntajes.append(0) # Tiende a "bajo" (0)
                elif valor_fila_num > max_val:
                    puntajes.append(2) # Tiende a "alto" (2)
            
            # Caso 2: Reglas con múltiples clases codificadas (ej. para 'desempenio')
            elif any(k in rangos_clase for k in ["0", "1", "2"]):
                 for clase_codificada, (min_val, max_val) in rangos_clase.items():
                     try:
                         clase_codificada_int = int(clase_codificada)
                         if min_val <= valor_fila_num <= max_val:
                             puntajes.append(clase_codificada_int)
                             break # Una vez que coincide con una clase, pasar a la siguiente columna
                     except ValueError:
                         logging.warning(f"Clave de clase codificada '{clase_codificada}' no es un entero. Ignorando.")
                         continue
    
    # Lógica de desempate y consolidación final de los puntajes
    if puntajes:
        contador = Counter(puntajes)
        # Priorizar el resultado más frecuente. Si hay empates, min() rompe a favor del valor más bajo.
        # Para mayor precisión, podríamos forzar el "alto" si existe o el "bajo" si existe
        # antes de caer en "medio" en caso de empate.
        max_frecuencia = max(contador.values())
        candidatos = [k for k, v in contador.items() if v == max_frecuencia]
        resultado = min(candidatos) 
        
        # Introducir ruido controlado
        if np.random.rand() < p_ruido:
            # El ruido tiene una pequeña probabilidad de mover el resultado a una clase adyacente,
            # pero con mayor probabilidad de quedarse en la clase original.
            if resultado == 0: # bajo
                resultado = np.random.choice([0, 1], p=[0.8, 0.2]) # 80% bajo, 20% medio
            elif resultado == 2: # alto
                resultado = np.random.choice([1, 2], p=[0.2, 0.8]) # 20% medio, 80% alto
            else: # medio (1)
                resultado = np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1]) # 10% bajo, 80% medio, 10% alto
        return resultado
    
    # Si ninguna regla aplica o no hay puntajes, devuelve una predicción aleatoria (con tendencia a medio)
    return np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1])


def generar_datos_sinteticos_con_reglas(reglas, n_samples=3000, p_ruido=0.01): # <--- n_samples MODIFICADO a 3000
    """
    Genera un DataFrame con datos sintéticos y aplica las reglas para definir desempenio_futuro.
    """
    areas = [
        'reposicion', 'ventas', 'atencion al cliente', 'administracion',
        'caja', 'logistica', 'deposito'
    ]
    jerarquias = ['trainee', 'junior', 'senior']
    desempenios = ['bajo', 'medio', 'alto']

    data = {
        'nombre': [f'Empleado {i+1}' for i in range(n_samples)],
        'area': np.random.choice(areas, n_samples),
        'jerarquia': np.random.choice(jerarquias, n_samples, p=[0.3, 0.4, 0.3]),
        'puntaje': np.random.randint(30, 100, n_samples),
        'cantidad_proyectos': np.random.randint(1, 6, n_samples),
        'desempenio': np.random.choice(desempenios, n_samples, p=[0.2, 0.5, 0.3]),
        'personas_equipo': np.random.randint(2, 31, n_samples),
        'horas_extra': np.random.randint(0, 21, n_samples),
        'asistencia_puntualidad': np.random.randint(40, 101, n_samples)
    }
    df = pd.DataFrame(data)

    # Mapear a numéricos ANTES de aplicar las reglas si las reglas internas esperan números
    mapa_jerarquia_numerico = {"trainee": 0, "junior": 1, "senior": 2}
    mapa_desempenio_numerico = {"bajo": 0, "medio": 1, "alto": 2}
    df_temp_mapped = df.copy() # Trabajar en una copia para el mapeo
    df_temp_mapped["jerarquia"] = df_temp_mapped["jerarquia"].map(mapa_jerarquia_numerico).fillna(df_temp_mapped["jerarquia"])
    df_temp_mapped["desempenio"] = df_temp_mapped["desempenio"].map(mapa_desempenio_numerico).fillna(df_temp_mapped["desempenio"])
    
    # Asegurarse de que las columnas usadas en las reglas sean numéricas si lo requieren
    for col in reglas.keys():
        if col in df_temp_mapped.columns:
            df_temp_mapped[col] = pd.to_numeric(df_temp_mapped[col], errors='coerce')

    df["desempenio_futuro"] = df_temp_mapped.apply(lambda fila: clasificar_fila_con_ruido(fila, reglas, p_ruido), axis=1)
    
    # Volver a mapear las columnas originales si es necesario para el CSV de salida
    # Es crucial que las columnas 'jerarquia' y 'desempenio' se mantengan en su formato original de cadena
    # o que se conviertan de nuevo a cadena si el modelo lo espera así en el entrenamiento.
    # Dado que 'regresion.py' mapea de string a numérico, aquí deben permanecer como strings
    # si esa es la entrada inicial esperada, o numéricas si se preprocesan antes.
    # Si el mapeo a numérico es solo para la función clasificar_fila_con_ruido,
    # el DataFrame final debería tener los valores originales.
    
    # Comentamos las líneas que mapeaban a numérico para el CSV de salida,
    # ya que se espera que el CSV se genere con los valores categóricos originales
    # o que el script de regresión maneje el mapeo a numérico.
    # if "jerarquia" in df.columns:
    #     df["jerarquia"] = df["jerarquia"].map(mapa_jerarquia_numerico).fillna(df["jerarquia"]) 
    # if "desempenio" in df.columns:
    #     df["desempenio"] = df["desempenio"].map(mapa_desempenio_numerico).fillna(df["desempenio"])

    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Uso: python generar_synthetic_training_data.py <reglas.json>")
        print(json.dumps({"error": "Uso: python generar_synthetic_training_data.py <reglas.json>"}), file=sys.stderr)
        sys.exit(1)
    
    reglas_path = sys.argv[1]
    logging.info(f"Cargando reglas desde: {reglas_path}")
    try:
        with open(reglas_path, "r", encoding="utf-8") as f:
            reglas = json.load(f)
        logging.info(f"Reglas cargadas: {json.dumps(reglas, indent=2)}")
    except Exception as e:
        logging.error(f"Error al cargar el archivo de reglas JSON: {e}", exc_info=True)
        print(json.dumps({"error": f"Error al cargar reglas: {str(e)}"}), file=sys.stderr)
        sys.exit(1)

    logging.info(f"Generando datos sintéticos con n_samples={3000} y p_ruido={0.01}...")
    # Pasar p_ruido y n_samples explícitamente a la función
    df = generar_datos_sinteticos_con_reglas(reglas, n_samples=3000, p_ruido=0.01) 
    
    output_csv_path = os.path.join(os.path.dirname(sys.argv[0]), "synthetic_training_data.csv")
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    
    # Log la distribución de desempenio_futuro
    desempenio_counts = df['desempenio_futuro'].value_counts(normalize=True).to_dict()
    logging.info(f"Distribución de 'desempenio_futuro' en el CSV sintético: {desempenio_counts}")

    print(json.dumps({"message": f"Datos sintéticos generados y guardados en '{os.path.basename(output_csv_path)}'. Distribución de desempeño_futuro: {desempenio_counts}"}), file=sys.stdout)
    logging.info("Generación de datos sintéticos completada.")
