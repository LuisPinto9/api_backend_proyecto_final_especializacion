# Built-in libraries
import os
import re
import json
import uuid
import shutil
import warnings
from io import BytesIO
from typing import Optional

# Third-party libraries
import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import dialogflow_v2 as dialogflow
from google.oauth2 import service_account

# TensorFlow Keras imports
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Ignore warnings
warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar secret de Google Cloud - Secret Manager
credentials_info = json.loads(os.getenv("SERVICE_ACCOUNT_FILE"))

project_id = credentials_info["project_id"]
session_id = "session1"
language_code = "es"

diccionario_es = {
    0: 'Manzana Sarna del manzano',
    1: 'Manzana Podredumbre negra',
    2: 'Manzana Oxido del cedro y manzano',
    3: 'Manzana sana',
    4: 'Arándano sano',
    5: 'Cereza (incluyendo agria) mildiú polvoroso',
    6: 'Cereza (incluyendo agria) sana',
    7: 'Maíz Mancha foliar de Cercospora Mancha gris de la hoja',
    8: 'Maíz Roya común',
    9: 'Maíz Tizón foliar del norte',
   10: 'Maíz sano',
   11: 'Uva Podredumbre negra',
   12: 'Uva Esca (Yesca)',
   13: 'Uva Tizón de la hoja (Mancha foliar de Isariopsis)',
   14: 'Uva sana',
   15: 'Naranja Huanglongbing (Greening de los cítricos)',
   16: 'Melocotón Mancha bacteriana',
   17: 'Melocotón sano',
   18: 'Pimiento Mancha bacteriana',
   19: 'Pimiento sano',
   20: 'Papa Tizón temprano',
   21: 'Papa Tizón tardío',
   22: 'Papa sana',
   23: 'Frambuesa sana',
   24: 'Soja sana',
   25: 'Calabaza Oídio',
   26: 'Fresa Quemadura de la hoja',
   27: 'Fresa sana',
   28: 'Tomate Mancha bacteriana',
   29: 'Tomate Tizón temprano',
   30: 'Tomate Tizón tardío',
   31: 'Tomate Moho de la hoja',
   32: 'Tomate Mancha foliar de Septoria',
   33: 'Tomate Araña roja Ácaro de dos puntos',
   34: 'Tomate Mancha objetivo',
   35: 'Tomate Virus del enrollamiento de la hoja amarilla del tomate',
   36: 'Tomate Virus del mosaico del tomate',
   37: 'Tomate sano'
}

# Función conversacional completa
def detec_intent_texts_full(project_id, session_id, text, language_code):
  session_client = dialogflow.SessionsClient()
  session = session_client.session_path(project_id, session_id)

  text_input = dialogflow.TextInput(text=text, language_code=language_code)
  query_input = dialogflow.QueryInput(text=text_input)

  response = session_client.detect_intent(
      request={"session" : session, "query_input": query_input}
  )

  # Extraer información importante:
  fulfillment_text = response.query_result.fulfillment_text
  intent = response.query_result.intent.display_name
  confianza = response.query_result.intent_detection_confidence
  parametros = dict(response.query_result.parameters)

  # Crear un diccionario con la información
  return {
      "respuesta": fulfillment_text,
      "intencion": intent,
      "confianza": confianza,
      "parametros": dict(parametros)
    }

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_cnn = None

@app.on_event("startup")
def load_keras_model():
    global model_cnn
    try:
        model_cnn = load_model("plant_disease_efficientnetb4.h5")
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        raise

def AnalizarEnfermedadHoja(path_imagen):
    global model_cnn
    try:
        # Detectar si es una URL o una ruta local
        if path_imagen.startswith("http://") or path_imagen.startswith("https://"):
            response = requests.get(path_imagen)
            if response.status_code != 200:
                return "No se pudo acceder a la imagen desde la URL"
            img = image.load_img(BytesIO(response.content), target_size=(380, 380))
        else:
            img = image.load_img(path_imagen, target_size=(380, 380))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model_cnn.predict(img_array)
        predicted_index = np.argmax(predictions)
        return diccionario_es.get(predicted_index, "Desconocido")

    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"

@app.get("/")
async def home():
    return JSONResponse(content="API del backend del proyecto integrador")

@app.post("/conversar")
async def conversar(
    mensaje: str = Form(""),
    imagen: UploadFile | None = File(None)
):
    try:
        resultado = detec_intent_texts_full(project_id, session_id, mensaje, language_code)

        if imagen:
            # Procesar imagen subida
            filename = imagen.filename
            ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
            unique_name = f"{uuid.uuid4()}.{ext}" if ext else str(uuid.uuid4())
            path = os.path.join(UPLOAD_FOLDER, unique_name)

            with open(path, "wb") as buffer:
                shutil.copyfileobj(imagen.file, buffer)

            resultado["imagen_guardada"] = path
            resultado["prediccion"] = AnalizarEnfermedadHoja(path)

        else:
            # Buscar URL en el mensaje y analizarla si es una imagen
            url_match = re.search(r'(https?://\S+)', mensaje)
            if url_match:
                url = url_match.group(0)
                resultado["url_detectada"] = url
                resultado["prediccion"] = AnalizarEnfermedadHoja(url)

        return JSONResponse(content=resultado)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)