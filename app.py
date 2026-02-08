import tempfile
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, auth
import subprocess
import os
import json
import sys # Importar sys para sys.executable en run_script
from config_postgres import get_connection # Asumo que esta función existe y es funcional
from psycopg2.extras import execute_values
import pandas as pd # Se mantiene por si hay otras funciones que lo usen
from datetime import datetime
import logging
from collections import Counter # Se mantiene por si hay otras funciones que lo usen
import random # Se mantiene por si hay otras funciones que lo usen
import jwt # Se encuentra en las importaciones originales del usuario
import time # Se encuentra en las importaciones originales del usuario

# Configura Flask y CORS
app = Flask(__name__)
CORS(app)

# =========================================================================
# === INICIO DE CONFIGURACIÓN ===
# =========================================================================

# Configuración de Firebase
# ASEGÚRATE de que 'firebase-service.json' esté en la misma carpeta que este 'app.py'
FIREBASE_SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), "firebase-service.json")

# Variable para controlar la autenticación (True/False)
ENABLE_AUTH = False # Cambiado a False para simplificar las pruebas iniciales

# Rutas fijas de CSV en el servidor (estas variables no se usan directamente en el nuevo flujo
# para generar CSV de entrenamiento, pero se mantienen por compatibilidad si es necesario)
CSV_ENTRADA_PATH = os.path.join(os.path.dirname(__file__), "prediccion_rendimiento_training.csv")
CSV_SALIDA_PATH = os.path.join(os.path.dirname(__file__), "prediccion_rendimiento_training_completo.csv")


# Inicializa Firebase
try:
    cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logging.info("Firebase inicializado correctamente.")
except Exception as e:
    logging.error(f"Error al inicializar Firebase: {e}")
    # Considera una forma de manejar esto si Firebase es crítico para la app.

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================================================================
# === LÓGICA DE BASE DE DATOS PARA LAS REGLAS ===
# =========================================================================

def init_db_rules():
    """
    Inicializa la tabla 'reglas_aplicadas' en la base de datos PostgreSQL si no existe.
    Debe ser llamada al iniciar la aplicación.
    """
    logging.info("Iniciando verificación/creación de la tabla 'reglas_aplicadas'...")
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        # Usamos SERIAL para id_regla y TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP para fecha_aplicacion
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reglas_aplicadas (
                id_regla SERIAL PRIMARY KEY,
                fecha_aplicacion TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                nombre_csv_generado TEXT NOT NULL, -- Se usará un timestamp ID para identificar la corrida
                detalles_reglas JSONB NOT NULL -- JSONB es más eficiente para PostgreSQL
            );
        ''')
        conn.commit()
        logging.info("Tabla 'reglas_aplicadas' verificada/creada exitosamente en PostgreSQL.")
    except Exception as e:
        logging.error(f"❌ Error al inicializar la base de datos para reglas: {e}")
        # En una aplicación real, podrías querer levantar la excepción o manejarla más robustamente
    finally:
        if cursor:
            cursor.close()
            logging.info("Cursor de init_db_rules cerrado.")
        if conn:
            conn.close()
            logging.info("Conexión de init_db_rules cerrada.")

# =========================================================================
# === FUNCIONES DE APOYO ===
# =========================================================================

def run_script(script_path, *args):
    """
    Ejecuta un script Python como un subproceso y captura su salida.
    Si el script devuelve un JSON con una clave 'error', lanza una excepción.
    Acepta argumentos adicionales para pasar al script.
    """
    logging.info(f"Preparando para ejecutar script: {script_path}")
    try:
        command = [sys.executable, script_path] # Usar sys.executable para mayor compatibilidad
        command.extend(args) # Añadir todos los argumentos
        logging.info(f"Comando a ejecutar: {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True # Esto levantará CalledProcessError si el script devuelve un código de error distinto de 0
        )
        logging.info(f"Script {script_path} ejecutado exitosamente.")
        
        # --- Lógica: Procesar la salida JSON del script ---
        # Algunos scripts pueden imprimir solo un mensaje y no JSON.
        # Intentamos parsear como JSON, pero si falla, retornamos el texto crudo.
        try:
            parsed_output = json.loads(result.stdout)
            if isinstance(parsed_output, dict) and 'error' in parsed_output:
                logging.error(f"El script {script_path} devolvió un error en su salida JSON: {parsed_output['error']}")
                raise Exception(parsed_output['error']) # Lanzar el error del script
            return parsed_output # Devolver la salida JSON válida (no-error)
        except json.JSONDecodeError:
            # Si no es JSON, simplemente devolvemos el texto stdout
            logging.warning(f"El script {script_path} no devolvió una salida JSON válida. STDOUT: {result.stdout.strip()}")
            return {"message": result.stdout.strip()} # Devuelve un diccionario para consistencia

    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Error en el script {script_path} (Código: {e.returncode}): {e.stderr}")
        try:
            error_details = json.loads(e.stderr)
            raise Exception(f"Error en el script: {error_details.get('error', 'Error desconocido del script (stderr)')}")
        except json.JSONDecodeError:
            raise Exception(f"El script no devolvió un JSON válido o error: {e.stderr.strip()}")
    except FileNotFoundError:
        logging.error(f"❌ Script no encontrado: {script_path}")
        raise Exception(f"Script no encontrado: {script_path}")
    except Exception as e:
        logging.error(f"❌ Error inesperado al ejecutar el script {script_path}: {e}")
        raise Exception(f"Error inesperado al ejecutar el script: {str(e)}")

# --- Funciones `clasificar_fila` y `aplicar_reglas_y_guardar` ELIMINADAS ---
# Ya no son necesarias en el nuevo flujo de trabajo, donde
# `generar_synthetic_training_data.py` maneja la aplicación de reglas para
# la generación de datos de entrenamiento sintéticos.


# =========================================================================
# === ENDPOINTS DE LA API ===
# =========================================================================

@app.route('/', methods=['GET'])
def index():
    logging.info("Llamada a la ruta principal '/'.")
    return jsonify({
        "message": "Bienvenido a la API de predicción",
        "status": "OK",
        "endpoints_disponibles": [
            "/health",
            "/api/predict/rotation",
            "/api/predict/performance_train",
            "/test",
            "/api/predict/future_performance",
            "/interfaz",
            "/api/data/regresion",
            "/api/predict/generar_csv_training",
            "/api/data/reglas_previas",
            "/api/data/regla_por_id/<int:rule_id>",
            "/api/predict/train_with_historical" # Nuevo endpoint para entrenar con reglas históricas
        ]
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    logging.info("Llamada a la ruta de salud '/health'.")
    return jsonify({
        "status": "OK",
        "message": "El servidor está funcionando correctamente",
        "endpoints": {
            "kmeans": "/api/predict/rotation",
            "entrenar_regresion": "/api/predict/performance_train",
            "future_performance": "/api/predict/future_performance",
            "get_regresion_data": "/api/data/regresion",
            "generar_csv_training": "/api/predict/generar_csv_training",
            "get_reglas_previas": "/api/data/reglas_previas",
            "get_regla_por_id": "/api/data/regla_por_id/<int:rule_id>",
            "train_with_historical": "/api/predict/train_with_historical" # Nuevo endpoint
        }
    }), 200

@app.route('/api/predict/rotation', methods=['POST'])
def predict_rotation():
    logging.info("➡️ Se ha llamado al endpoint /api/predict/rotation.")
    if ENABLE_AUTH:
        logging.info("Autenticación habilitada para /api/predict/rotation. Verificando token...")
        try:
            token = request.headers.get('Authorization', '').split(" ")[1]
            auth.verify_id_token(token)
            logging.info("Token de autenticación verificado.")
        except Exception as e:
            logging.error(f"❌ Error de autenticación en /api/predict/rotation: {e}")
            return jsonify({"error": f"Error de autenticación: {str(e)}"}), 401
    else:
        logging.info("Autenticación deshabilitada para /api/predict/rotation.")

    try:
        script_path = os.path.join(os.path.dirname(__file__), "K-Means", "K-Means-Rotacion.py")
        output = run_script(script_path)
        logging.info("Predicción de rotación completada exitosamente.")
        return jsonify(output), 200
    except Exception as e:
        logging.error(f"❌ Error en endpoint /api/predict/rotation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/generar_csv_training', methods=['POST'])
def generar_csv_entrenamiento_endpoint():
    logging.info("➡️ Se ha llamado al endpoint /api/predict/generar_csv_training (generar sintéticos y guardar reglas).")
    try:
        reglas_json = request.get_json()
        if not reglas_json:
            logging.warning("No se recibieron reglas JSON en la solicitud. Cuerpo de la solicitud vacío o inválido.")
            return jsonify({"error": "No se enviaron reglas JSON"}), 400

        logging.info(f"Reglas JSON recibidas del frontend: {json.dumps(reglas_json, indent=2)}")

        # Guardar reglas en archivo temporal para pasar al generador sintético
        reglas_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as reglas_file:
                json.dump(reglas_json, reglas_file)
                reglas_file_path = reglas_file.name
            logging.info(f"Reglas guardadas en archivo temporal: {reglas_file_path}")

            # Llamar al generador sintético con las reglas
            script_path = os.path.join(os.path.dirname(__file__), "Regresion lineal", "generar_synthetic_training_data.py")
            # run_script ahora acepta múltiples argumentos
            synthetic_gen_output = run_script(script_path, reglas_file_path) 
            
            # El script generador imprime un mensaje simple o un JSON con error
            if isinstance(synthetic_gen_output, dict) and 'error' in synthetic_gen_output:
                 logging.error(f"Error del generador sintético: {synthetic_gen_output['error']}")
                 return jsonify({"error": synthetic_gen_output['error']}), 500
            elif isinstance(synthetic_gen_output, dict) and 'message' in synthetic_gen_output:
                logging.info(f"Mensaje del generador sintético: {synthetic_gen_output['message']}")
            else:
                logging.info(f"Salida inesperada del generador sintético: {synthetic_gen_output}")

            # Leer el CSV generado para vista previa
            csv_path = os.path.join(os.path.dirname(__file__), "Regresion lineal", "synthetic_training_data.csv")
            if not os.path.exists(csv_path):
                logging.error(f"El archivo CSV sintético no fue generado por {script_path}: {csv_path}")
                return jsonify({"error": "El archivo CSV sintético no fue generado"}), 500
                
            df_resultado = pd.read_csv(csv_path)
            logging.info(f"CSV sintético generado exitosamente con {len(df_resultado)} filas.")

        except Exception as gen_e:
            logging.error(f"Error en la generación del CSV sintético: {gen_e}", exc_info=True)
            return jsonify({"error": f"Error en la generación del CSV sintético: {str(gen_e)}"}), 500
        finally:
            # Limpiar archivo temporal de reglas
            if reglas_file_path and os.path.exists(reglas_file_path):
                os.unlink(reglas_file_path)
                logging.info(f"Archivo temporal de reglas eliminado: {reglas_file_path}")

        # --- Guardar reglas en la base de datos ---
        logging.info("Intentando establecer conexión a la base de datos para guardar reglas...")
        conn = None
        cursor = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            logging.info("Conexión a la base de datos establecida.")
            
            timestamp_id = datetime.now().strftime("training_run_%Y%m%d_%H%M%S")
            reglas_string = json.dumps(reglas_json)
            
            logging.info(f"Preparando inserción de reglas: ID={timestamp_id}, Detalles={reglas_string[:100]}...")
            cursor.execute(
                "INSERT INTO reglas_aplicadas (fecha_aplicacion, nombre_csv_generado, detalles_reglas) VALUES (NOW(), %s, %s)",
                (timestamp_id, reglas_string)
            )
            conn.commit()
            logging.info(f"✅ Reglas guardadas exitosamente en la base de datos para ID: {timestamp_id}. Commited.")
        except Exception as db_e:
            logging.error(f"❌ Error al guardar reglas en la base de datos: {db_e}. Rolback de la transacción.", exc_info=True)
            if conn:
                conn.rollback()
            return jsonify({"error": f"CSV sintético generado, pero error al guardar reglas en DB: {str(db_e)}"}), 500
        finally:
            if cursor:
                cursor.close()
                logging.info("Cursor de reglas_aplicadas cerrado.")
            if conn:
                conn.close()
                logging.info("Conexión de reglas_aplicadas cerrada.")
        # --- Fin de guardar reglas en la base de datos ---

        logging.info("Respondiendo al frontend tras la generación de CSV sintético y guardado de reglas.")
        return jsonify({
            "mensaje": "CSV de entrenamiento sintético generado exitosamente y reglas guardadas.",
            "archivo": os.path.basename(csv_path),
            "vista_previa": df_resultado.head(10).to_dict(orient="records")
        }), 200

    except Exception as e:
        logging.error(f"❌ Error general en /api/predict/generar_csv_training: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/performance_train', methods=['POST'])
def performance_train_endpoint(): # Renombrado para evitar conflicto si se usa `predict_performance` en otro lado
    logging.info("➡️ Se ha llamado al endpoint /api/predict/performance_train (entrenamiento del modelo).")
    if ENABLE_AUTH:
        logging.info("Autenticación habilitada para /api/predict/performance_train. Verificando token...")
        try:
            token = request.headers.get('Authorization', '').split(" ")[1]
            auth.verify_id_token(token)
            logging.info("Token de autenticación verificado.")
        except Exception as e:
            logging.error(f"❌ Error de autenticación en /api/predict/performance_train: {e}")
            return jsonify({"error": f"Error de autenticación: {str(e)}"}), 401
    else:
        logging.info("Autenticación deshabilitada para /api/predict/performance_train.")

    script_path = os.path.join(os.path.dirname(__file__), "Regresion lineal", "regresion.py")
    
    try:
        logging.info(f"Ejecutando script de entrenamiento del modelo (regresion.py): {script_path}")
        output = run_script(script_path)
        logging.info("Script de entrenamiento del modelo finalizado exitosamente.")
        return jsonify(output), 200
    except Exception as e:
        logging.error(f"❌ Error en endpoint /api/predict/performance_train: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/train_with_historical', methods=['POST'])
def train_with_historical_rules():
    logging.info("➡️ Se ha llamado al endpoint /api/predict/train_with_historical.")
    try:
        data = request.get_json()
        if not data or 'rule_id' not in data:
            return jsonify({"error": "Se requiere rule_id"}), 400

        rule_id = data['rule_id']
        logging.info(f"Entrenando con regla histórica ID: {rule_id}")

        # Obtener reglas de la base de datos
        conn = None
        cursor = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT detalles_reglas FROM reglas_aplicadas WHERE id_regla = %s;", (rule_id,))
            result = cursor.fetchone()
            
            if not result:
                logging.warning(f"Regla con ID {rule_id} no encontrada en la base de datos.")
                return jsonify({"error": f"Regla con ID {rule_id} no encontrada"}), 404
                
            reglas_json = result[0] # JSONB se carga como dict directamente
            logging.info(f"Reglas obtenidas de la BD para ID {rule_id}: {json.dumps(reglas_json, indent=2)}")
            
        except Exception as db_e:
            logging.error(f"Error al obtener reglas de la BD para ID {rule_id}: {db_e}", exc_info=True)
            return jsonify({"error": f"Error al obtener reglas: {str(db_e)}"}), 500
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        # Generar CSV sintético con las reglas históricas
        reglas_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as reglas_file:
                json.dump(reglas_json, reglas_file)
                reglas_file_path = reglas_file.name
            logging.info(f"Reglas históricas guardadas en archivo temporal: {reglas_file_path}")

            # Llamar al generador sintético
            script_path = os.path.join(os.path.dirname(__file__), "Regresion lineal", "generar_synthetic_training_data.py")
            synthetic_gen_output = run_script(script_path, reglas_file_path)
            
            if isinstance(synthetic_gen_output, dict) and 'error' in synthetic_gen_output:
                 logging.error(f"Error del generador sintético con reglas históricas: {synthetic_gen_output['error']}")
                 return jsonify({"error": synthetic_gen_output['error']}), 500
            elif isinstance(synthetic_gen_output, dict) and 'message' in synthetic_gen_output:
                logging.info(f"Mensaje del generador sintético con reglas históricas: {synthetic_gen_output['message']}")
            else:
                logging.info(f"Salida inesperada del generador sintético con reglas históricas: {synthetic_gen_output}")

            # Entrenar el modelo
            train_script_path = os.path.join(os.path.dirname(__file__), "Regresion lineal", "regresion.py")
            logging.info(f"Ejecutando entrenamiento del modelo con CSV sintético basado en regla ID {rule_id}.")
            train_output = run_script(train_script_path)
            
            logging.info("Entrenamiento con reglas históricas completado exitosamente.")
            return jsonify({
                "mensaje": f"Modelo entrenado exitosamente con regla ID {rule_id}",
                "resultado_entrenamiento": train_output
            }), 200

        except Exception as gen_e:
            logging.error(f"Error en el proceso de entrenamiento con regla histórica: {gen_e}", exc_info=True)
            return jsonify({"error": f"Error en el proceso: {str(gen_e)}"}), 500
        finally:
            if reglas_file_path and os.path.exists(reglas_file_path):
                os.unlink(reglas_file_path)
                logging.info(f"Archivo temporal de reglas históricas eliminado: {reglas_file_path}")

    except Exception as e:
        logging.error(f"❌ Error general en /api/predict/train_with_historical: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/future_performance', methods=['POST'])
def predict_future_performance():
    logging.info("➡️ Se ha llamado al endpoint /api/predict/future_performance.")
    if ENABLE_AUTH:
        logging.info("Autenticación habilitada para /api/predict/future_performance. Verificando token...")
        try:
            token = request.headers.get('Authorization', '').split(" ")[1]
            auth.verify_id_token(token)
            logging.info("Token de autenticación verificado.")
        except Exception as e:
            logging.error(f"❌ Error de autenticación en /api/predict/future_performance: {e}")
            return jsonify({"error": f"Error de autenticación: {str(e)}"}), 401
    else:
        logging.info("Autenticación deshabilitada para /api/predict/future_performance.")

    archivo_temporal_path = None
    conn = None
    cursor = None
    conn_rules = None
    cursor_rules = None

    try:
        if 'file' not in request.files:
            logging.warning("No se recibió ningún archivo CSV en la solicitud de predicción futura.")
            return jsonify({"error": "No se proporcionó ningún archivo CSV"}), 400
        archivo_csv = request.files['file']
        if archivo_csv.filename == '' or not archivo_csv.filename.endswith('.csv'):
            logging.warning(f"Archivo subido inválido: {archivo_csv.filename}")
            return jsonify({"error": "Por favor, sube un archivo CSV válido"}), 400
        
        # --- Obtener id_regla_seleccionada del formulario ---
        # Este ID es el que se DEBE usar para guardar la predicción
        id_regla_para_guardar = request.form.get('id_regla_seleccionada')
        if id_regla_para_guardar:
            try:
                id_regla_para_guardar = int(id_regla_para_guardar)
                logging.info(f"Se recibió id_regla_seleccionada para la predicción: {id_regla_para_guardar}")
            except ValueError:
                logging.warning(f"id_regla_seleccionada no es un entero válido: {request.form.get('id_regla_seleccionada')}. Se ignorará.")
                id_regla_para_guardar = None # Resetear si no es válido
        else:
            logging.info("No se recibió id_regla_seleccionada. Se buscará el último id_regla_aplicada del entrenamiento.")
            # Si no se selecciona una regla específica, obtenemos la última generada/entrenada
            try:
                conn_rules = get_connection()
                cursor_rules = conn_rules.cursor()
                cursor_rules.execute("SELECT id_regla FROM reglas_aplicadas ORDER BY fecha_aplicacion DESC LIMIT 1;")
                result = cursor_rules.fetchone()
                if result:
                    id_regla_para_guardar = result[0]
                    logging.info(f"Obtenido id_regla_aplicada (último entrenamiento) para la predicción: {id_regla_para_guardar}")
                else:
                    logging.warning("No se encontró ningún id_regla_aplicada reciente en la base de datos. Se insertará NULL.")
            except Exception as e_rules:
                logging.error(f"❌ Error al obtener el último id_regla_aplicada desde la DB: {e_rules}", exc_info=True)
            finally:
                if cursor_rules:
                    cursor_rules.close()
                if conn_rules:
                    conn_rules.close()
        # --- FIN: Obtener id_regla_seleccionada ---

        logging.info(f"Guardando archivo CSV de predicción temporal: {archivo_csv.filename}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            archivo_csv.save(tmp_file.name)
            archivo_temporal_path = tmp_file.name
        logging.info(f"Archivo temporal guardado en: {archivo_temporal_path}")

        script_path = os.path.join(os.path.dirname(__file__), "Regresion lineal", "predecir_rendimiento_futuro.py")
        logging.info(f"Ejecutando script de predicción futura: {script_path} con archivo: {archivo_temporal_path}")
        output = run_script(script_path, archivo_temporal_path)
        logging.info("Script de predicción futura finalizado exitosamente.")

        # --- Insertar resultados en la base de datos ---
        logging.info("Intentando insertar resultados de predicción en random_forest_resultados...")
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
            INSERT INTO random_forest_resultados
            (nombre, area, jerarquia, puntaje, cantidad_proyectos, desempenio, personas_equipO, horas_extra, asistencia_puntualidad, desempenio_futuro, fecha, id_regla_aplicada)
            VALUES %s
        """
        fecha_actual = datetime.now()
        # Asegurarse de que estos mapas coincidan con los valores que produce predecir_rendimiento_futuro.py
        # Si predecir_rendimiento_futuro.py ya devuelve cadenas, no se necesita mapeo aquí.
        mapa_jerarquia_num_a_str = {0: 'trainee', 1: 'junior', 2: 'senior'}
        mapa_desempenio_num_a_str = {0: 'bajo', 1: 'medio', 2: 'alto'}
        mapa_rendimiento_num_a_str = {0: 'bajo', 1: 'medio', 2: 'alto'}
        
        if not isinstance(output, list):
            logging.error(f"El script de predicción futura no devolvió una lista: {output}")
            raise ValueError("Formato de salida de predicción futura inesperado.")

        valores = [
            (
                d.get('nombre'),
                d.get('area'),
                # Usar los mapeos solo si el valor es numérico y necesita ser convertido a string
                mapa_jerarquia_num_a_str.get(d.get('jerarquia'), d.get('jerarquia')),
                d.get('puntaje'),  
                d.get('cantidad_proyectos'),
                mapa_desempenio_num_a_str.get(d.get('desempenio'), d.get('desempenio')),
                d.get('personas_equipo'),
                d.get('horas_extra'), 
                d.get('asistencia_puntualidad'),
                mapa_rendimiento_num_a_str.get(d.get('desempenio_futuro'), d.get('desempenio_futuro')), # Aquí ya debería ser el rendimiento final del modelo
                fecha_actual,
                id_regla_para_guardar # Usamos el ID determinado aquí
            ) for d in output
        ]
        
        logging.info(f"Preparando inserción de {len(valores)} filas en random_forest_resultados.")
        execute_values(cursor, query, valores)
        conn.commit()
        logging.info("✅ Datos de predicción futura guardados en PostgreSQL exitosamente.")
        
        return jsonify({"mensaje": "Datos guardados en PostgreSQL exitosamente", "resultados": output}), 200
        
    except Exception as e:
        logging.error(f"❌ Error general en /api/predict/future_performance: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if archivo_temporal_path and os.path.exists(archivo_temporal_path):
            os.remove(archivo_temporal_path)
            logging.info(f"Archivo temporal eliminado: {archivo_temporal_path}")
        if cursor:
            cursor.close()
            logging.info("Cursor de random_forest_resultados cerrado.")
        if conn:
            conn.close()
            logging.info("Conexión de random_forest_resultados cerrada.")


# --- ENDPOINT: Obtener reglas previamente aplicadas (LISTA) ---
@app.route('/api/data/reglas_previas', methods=['GET'])
def get_reglas_previas():
    logging.info("➡️ Se ha llamado al endpoint /api/data/reglas_previas.")
    if ENABLE_AUTH:
        logging.info("Autenticación habilitada para /api/data/reglas_previas. Verificando token...")
        try:
            token = request.headers.get('Authorization', '').split(" ")[1]
            auth.verify_id_token(token)
            logging.info("Token de autenticación verificado.")
        except Exception as e:
            logging.error(f"❌ Error de autenticación en /api/data/reglas_previas: {e}")
            return jsonify({"error": f"Error de autenticación: {str(e)}"}), 401
    else:
        logging.info("Autenticación deshabilitada para /api/data/reglas_previas.")

    conn = None
    cursor = None
    try:
        logging.info("Intentando establecer conexión a la base de datos para obtener reglas previas (lista).")
        conn = get_connection()
        cursor = conn.cursor()
        logging.info("Conexión a la base de datos establecida.")
        
        cursor.execute("SELECT id_regla, fecha_aplicacion, detalles_reglas FROM reglas_aplicadas ORDER BY fecha_aplicacion DESC;")
        
        column_names = [desc[0] for desc in cursor.description]
        
        reglas_data = []
        for row in cursor.fetchall():
            row_dict = dict(zip(column_names, row))
            if 'fecha_aplicacion' in row_dict and isinstance(row_dict['fecha_aplicacion'], datetime):
                row_dict['fecha_aplicacion'] = row_dict['fecha_aplicacion'].isoformat()
            reglas_data.append(row_dict)
        
        logging.info(f"Obtenidas {len(reglas_data)} reglas previas (lista).")
        return jsonify(reglas_data), 200
    except Exception as e:
        logging.error(f"❌ Error al obtener reglas previas de la tabla 'reglas_aplicadas': {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            cursor.close()
            logging.info("Cursor de reglas_aplicadas (lista) cerrado.")
        if conn:
            conn.close()
            logging.info("Conexión de reglas_aplicadas (lista) cerrada.")


# --- ENDPOINT: Obtener una regla específica por ID ---
@app.route('/api/data/regla_por_id/<int:rule_id>', methods=['GET'])
def get_regla_por_id(rule_id):
    logging.info(f"➡️ Se ha llamado al endpoint /api/data/regla_por_id/{rule_id}.")
    if ENABLE_AUTH:
        logging.info("Autenticación habilitada para /api/data/regla_por_id. Verificando token...")
        try:
            token = request.headers.get('Authorization', '').split(" ")[1]
            auth.verify_id_token(token)
            logging.info("Token de autenticación verificado.")
        except Exception as e:
            logging.error(f"❌ Error de autenticación en /api/data/regla_por_id: {e}")
            return jsonify({"error": f"Error de autenticación: {str(e)}"}), 401
    else:
        logging.info("Autenticación deshabilitada para /api/data/regla_por_id.")

    conn = None
    cursor = None
    try:
        logging.info(f"Intentando obtener la regla con ID: {rule_id}.")
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id_regla, fecha_aplicacion, detalles_reglas FROM reglas_aplicadas WHERE id_regla = %s;", (rule_id,))
        result = cursor.fetchone()
        
        if result:
            column_names = [desc[0] for desc in cursor.description]
            regla_data = dict(zip(column_names, result))
            if 'fecha_aplicacion' in regla_data and isinstance(regla_data['fecha_aplicacion'], datetime):
                regla_data['fecha_aplicacion'] = regla_data['fecha_aplicacion'].isoformat()
            
            logging.info(f"✅ Regla ID {rule_id} encontrada y enviada.")
            return jsonify(regla_data), 200
        else:
            logging.warning(f"❌ Regla con ID {rule_id} no encontrada.")
            return jsonify({"error": f"Regla con ID {rule_id} no encontrada."}), 404
            
    except Exception as e:
        logging.error(f"❌ Error al obtener regla con ID {rule_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            cursor.close()
            logging.info("Cursor de regla_por_id cerrado.")
        if conn:
            conn.close()
            logging.info("Conexión de regla_por_id cerrada.")


@app.route('/test', methods=['GET'])
def test_page():
    logging.info("Llamada a la ruta '/test'.")
    return send_file('test.html')

@app.route('/interfaz', methods=['GET'])
def interfaz_page():
    logging.info("Llamada a la ruta '/interfaz'.")
    with open('interfaz.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return render_template_string(html_content)

@app.route('/api/data/regresion', methods=['GET'])
def get_regresion_data():
    logging.info("➡️ Se ha llamado al endpoint /api/data/regresion.")
    if ENABLE_AUTH:
        logging.info("Autenticación habilitada para /api/data/regresion. Verificando token...")
        try:
            token = request.headers.get('Authorization', '').split(" ")[1]
            auth.verify_id_token(token)
            logging.info("Token de autenticación verificado.")
        except Exception as e:
            logging.error(f"❌ Error de autenticación en /api/data/regresion: {e}")
            return jsonify({"error": f"Error de autenticación: {str(e)}"}), 401
    else:
        logging.info("Autenticación deshabilitada para /api/data/regresion.")

    conn = None
    cursor = None
    try:
        logging.info("Intentando establecer conexión a la base de datos para obtener datos de regresión.")
        conn = get_connection()
        cursor = conn.cursor()
        logging.info("Conexión a la base de datos establecida.")
        
        logging.info("Ejecutando consulta SELECT para random_forest_resultados.")
        cursor.execute("SELECT id,nombre, area, jerarquia, puntaje, cantidad_proyectos, desempenio, personas_equipO, horas_extra, asistencia_puntualidad, desempenio_futuro, fecha, id_regla_aplicada FROM random_forest_resultados ORDER BY fecha DESC")
        
        column_names = [desc[0] for desc in cursor.description]
        logging.info(f"Columnas obtenidas: {column_names}")
        
        data = []
        for row in cursor.fetchall():
            row_dict = dict(zip(column_names, row))
            if 'fecha' in row_dict and isinstance(row_dict['fecha'], datetime):
                row_dict['fecha'] = row_dict['fecha'].isoformat()
            data.append(row_dict)
        
        logging.info(f"Obtenidos {len(data)} filas de datos de regresión. Respondiendo.")
        return jsonify(data), 200
    except Exception as e:
        logging.error(f"❌ Error al obtener datos de la tabla random_forest_resultados: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            cursor.close()
            logging.info("Cursor de random_forest_resultados (lectura) cerrado.")
        if conn:
            conn.close()
            logging.info("Conexión de random_forest_resultados (lectura) cerrada.")

if __name__ == '__main__':
    logging.info("Iniciando la aplicación Flask.")
    with app.app_context(): 
        init_db_rules()
    app.run(host='0.0.0.0', port=5000, debug=True)
