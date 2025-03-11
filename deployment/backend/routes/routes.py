from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os


mongo_uri = os.getenv('MONGO_URI', 'mongodb://root:password@mongodb:27017/alzheimer_db?authSource=admin')
client = MongoClient(mongo_uri)
db = client['alzheimer_db']
patients_collection = db['patients']



MODEL_PATH = 'CNN_Categorical_Crossentropy.h5'
model = load_model(MODEL_PATH)
CLASS_NAMES = ['No Impairment', 'Moderate Impairment', 'Mild Impairment', 'Very Mild Impairment']


# Definir el blueprint para las rutas
patients = Blueprint('patients', __name__)

@patients.route('/register-patient', methods=['POST'])
def register_patient():
    try:
        data = request.get_json()
        if not data.get('name') or not data.get('edad') or not data.get('diagnostico'):
            return jsonify({"status": "Error", "message": "Faltan datos obligatorios"}), 400

        new_patient = {
            "name": data['name'],
            "edad": data['edad'],
            "diagnostico": data['diagnostico']
        }
        result = patients_collection.insert_one(new_patient)
        return jsonify({"status": "Paciente registrado", "id": str(result.inserted_id)}), 201

    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500


@patients.route('/get-patients', methods=['GET'])
def get_patients():
    try:
        patients = list(patients_collection.find({}, {'_id': 0}))  # Excluir el campo _id para simplificar
        return jsonify(patients), 200
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

# Ruta para actualizar paciente
@patients.route('/update-patient/<string:name>', methods=['PATCH'])
def update_patient(name):
    try:
        data = request.get_json()
        update_data = {}

        if 'edad' in data:
            update_data['edad'] = data['edad']
        if 'diagnostico' in data:
            update_data['diagnostico'] = data['diagnostico']

        if not update_data:
            return jsonify({"status": "Error", "message": "No hay datos para actualizar"}), 400

        result = patients_collection.update_one(
            {"name": name},
            {"$set": update_data}
        )

        if result.matched_count == 0:
            return jsonify({"status": "Error", "message": "Paciente no encontrado"}), 404

        return jsonify({"status": "Paciente actualizado"}), 200

    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500


# Ruta para eliminar paciente
@patients.route('/delete-patient/<string:name>', methods=['DELETE'])
def delete_patient(name):
    try:
        result = patients_collection.delete_one({"name": name})

        if result.deleted_count == 0:
            return jsonify({"status": "Error", "message": "Paciente no encontrado"}), 404

        return jsonify({"status": "Paciente eliminado"}), 200

    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

# Ruta para diagnosticar paciente
@patients.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "Error", "message": "No se encontró la imagen"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "Error", "message": "Nombre de archivo inválido"}), 400

        # Leer la imagen y preprocesarla
        img = Image.open(io.BytesIO(file.read())).resize((128, 128)).convert('L')  # Convertir a escala de grises
        img_array = np.array(img) / 255.0  # Normalizar a valores entre 0 y 1
        img_array = np.expand_dims(img_array, axis=0)  # Añadir solo el batch size
        img_array = img_array.reshape((1, 128, 128, 1))  # Asegurar las dimensiones (batch, 128, 128, 1)


        # Hacer la predicción
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            "status": "Diagnóstico exitoso",
            "diagnostico": predicted_class,
            "confianza": f"{confidence:.2f}%"
        }), 200

    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500
