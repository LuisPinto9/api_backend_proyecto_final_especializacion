import openai
from openai import OpenAI
from fastapi import FastAPI, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json, os, joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import shap
from google.oauth2 import service_account
from google.cloud import dialogflow_v2 as dialogflow

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Autenticación
credentials_info = json.loads(os.getenv("SERVICE_ACCOUNT_FILE"))
credentials = service_account.Credentials.from_service_account_info(credentials_info)

project_id = credentials_info["project_id"]
session_id = "session1"
language_code = "es"

credentials = service_account.Credentials.from_service_account_info(credentials_info)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Utilización de ChatGPT para extracción de características
client = OpenAI(api_key=openai.api_key)

def extraer_variables_desde_mensaje(mensaje_usuario):
    with open("gpt_prompt.txt", "r", encoding="utf-8") as f:
        assistant_instruction = f.read()

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": assistant_instruction
            },
            {
                "role": "user",
                "content": mensaje_usuario
            }
        ],
        temperature=1,
        max_tokens=1500
    )
    json_str = response.choices[0].message.content.strip()
    return json.loads(json_str)

# Modelo y datos
columnas_numericas = ['admin_page_qty', 'admin_duration_seconds', 'info_page_qty',
       'info_duration_seconds', 'product_page_qty', 'product_duration_seconds',
       'bounce_rate', 'exit_rate', 'page_value_amount', 'is_special_day']

df_normalize = pd.read_csv('df_normalize.csv')
scaler = StandardScaler()
df_normalize[columnas_numericas] = scaler.fit_transform(df_normalize[columnas_numericas])
df_base = df_normalize.drop(columns=['has_revenue'])

modelo = joblib.load('modelo_xgb.pkl')
columnas_modelo = joblib.load('columnas_modelo.pkl')
explainer = shap.TreeExplainer(modelo)

# Función Dialogflow
def detec_intent_texts_full(project_id, session_id, text, language_code):
    session_client = dialogflow.SessionsClient(credentials=credentials)
    session = session_client.session_path(project_id, session_id)
    text_input = dialogflow.TextInput(text=text, language_code=language_code)
    query_input = dialogflow.QueryInput(text=text_input)
    response = session_client.detect_intent(request={"session": session, "query_input": query_input})

  # Extraer información importante:
    fulfillment_text = response.query_result.fulfillment_text # texto que retorna el agente
    intent = response.query_result.intent.display_name # intención que retorna el agente
    confianza = response.query_result.intent_detection_confidence # confianza del agente al clasificar esa intención 0 - 1
    parametros = dict(response.query_result.parameters) # ... parámetros de la intención

    # Crear un diccionario con la información
    return {
        "respuesta": fulfillment_text,
        "intencion": intent,
        "confianza": confianza,
        "parametros": dict(parametros)
    }

def completar_input_usuario_mejorado(input_parcial, columnas_modelo, columnas_numericas, df_normalize, scaler, k=5):
    # Busca registros similares y promedia el resto de los campos.
    df_filtrado = df_base.copy()

    # Filtrar por columnas presentes en el input
    for col, val in input_parcial.items():
        df_filtrado = df_filtrado[df_filtrado[col] == val]

    if len(df_filtrado) >= k:
        muestra = df_filtrado.sample(n=k, random_state=42)
    elif len(df_filtrado) > 0:
        muestra = df_filtrado
    else:
        # Si no hay coincidencias, usar KNNImputer
        df_input = pd.DataFrame([input_parcial], columns=df_base.columns)
        imputer = KNNImputer(n_neighbors=5)
        imputado = imputer.fit_transform(pd.concat([df_input, df_base]).fillna(0))
        imputado_df = pd.DataFrame(imputado[:1], columns=df_base.columns)
        return imputado_df

    # Promediar la muestra
    input_completo = pd.DataFrame(columns=columnas_modelo)

    # Agrega los valores dados por el usuario
    for col in input_parcial:
        input_completo.at[0, col] = input_parcial[col]

    # Filtra el dataset por filas similares (según las columnas ingresadas)
    condiciones = True
    for col in input_parcial:
        condiciones &= df_normalize[col] == input_parcial[col]

    df_similares = df_normalize[condiciones]

    # Si no encuentra datos similares, usa la media
    if df_similares.empty:
        for col in columnas_modelo:
            if col not in input_parcial:
                if col in columnas_numericas:
                    input_completo.at[0, col] = df_normalize[col].mean()
                else:
                    input_completo.at[0, col] = df_normalize[col].mode()[0]
    else:
        for col in columnas_modelo:
            if col not in input_parcial:
                input_completo.at[0, col] = df_similares[col].mean() if col in columnas_numericas else df_similares[col].mode()[0]

    # Normaliza las columnas numéricas con el scaler ya entrenado
    input_completo[columnas_numericas] = scaler.transform(input_completo[columnas_numericas])

    return input_completo

def predecir_y_mostrar_factores(
    input_parcial, modelo, columnas_modelo, columnas_numericas,
    df_normalize, scaler, top_n=18, k=5
):
    # Realiza la predicción de compra y devuelve un mensaje explicativo con los factores clave.

    # Completar input del usuario con KNN
    input_modelo = completar_input_usuario_mejorado(
        input_parcial, columnas_modelo, columnas_numericas, df_normalize, scaler, k=k
    )

    # Predicción
    probabilidad = modelo.predict_proba(input_modelo)[0, 1]
    pred_clase = modelo.predict(input_modelo)[0]

    # Desnormalizar para mostrar valores reales
    input_legible = input_modelo.copy()
    input_legible[columnas_numericas] = scaler.inverse_transform(input_legible[columnas_numericas])

    # Calcular valores SHAP
    explainer = shap.TreeExplainer(modelo)
    shap_values_all = explainer.shap_values(input_modelo)
    shap_values = shap_values_all[1] if isinstance(shap_values_all, list) else shap_values_all

    # Tabla de factores
    factores_df = pd.DataFrame({
        'columna': columnas_modelo,
        'importancia': shap_values[0]
    })

    ingresadas = set(input_parcial.keys())
    factores_df['tipo'] = factores_df['columna'].apply(lambda col: 'Usuario' if col in ingresadas else 'Imputado')
    factores_df = factores_df.reindex(factores_df.importancia.abs().sort_values(ascending=False).index)
    factores_top = factores_df.head(top_n)

    # Construir mensaje explicativo
    mensaje = []

    # Parte 1: Resultado principal
    mensaje.append(f"Según el análisis del sistema, la probabilidad de compra para este visitante es de {probabilidad * 100:.2f}%.")
    mensaje.append(f"Esto sugiere que el modelo predice una como clase: {'Compra' if pred_clase == 1 else 'No compra'}.\n")

    # Parte 2: Características proporcionadas
    mensaje.append("Estas son las caracteristicas ingresadas por el usuaio sobre el visitante:")
    for k, v in input_parcial.items():
        mensaje.append(f"- {k}: {v}")

    # Parte 3: Factores influyentes (separados por tipo)
    mensaje.append(f"\nPrincipales factores ingresados por el usuario (top {top_n}):")
    factores_usuario = factores_top[factores_top["tipo"] == "Usuario"].head(top_n)
    for i, fila in factores_usuario.iterrows():
        mensaje.append(f" - {fila['columna']:<25} {fila['importancia']:.6f}   Usuario")

    mensaje.append(f"\nPrincipales factores imputados por el sistema (top {top_n}):")
    factores_imputados = factores_top[factores_top["tipo"] == "Imputado"].head(top_n)
    for i, fila in factores_imputados.iterrows():
        mensaje.append(f"- {fila['columna']:<25} {fila['importancia']:.6f}   Imputado")

    # Parte 4: Recomendación
    if probabilidad >= 0.8:
        mensaje.append("\nRecomendación: Este usuario muestra alta intención de compra. Considera activar promociones o recomendaciones personalizadas.")
    elif probabilidad <= 0.2:
        mensaje.append("\nRecomendación: Este usuario muestra baja intención de compra. Revisa elementos críticos como duración o tipo de páginas vistas.")

    return "\n".join(mensaje)

@app.post("/conversar")
async def conversar(request: Request):
    try:
        datos = await request.json()
        mensaje_usuario = datos.get("mensaje", "")
        
        if not mensaje_usuario:
            return {"error": "Falta el mensaje."}

        # Ejecutar Dialogflow para detectar intención
        dialogflow_result = detec_intent_texts_full(
            project_id=project_id,
            session_id=session_id,
            text=mensaje_usuario,
            language_code=language_code
        )

        if dialogflow_result["intencion"]=="compra":
            # Extraer variables desde el mensaje usando GPT
            try:
                input_parcial = extraer_variables_desde_mensaje(mensaje_usuario)
            except Exception as extraction_error:
                return {
                    "error": "No se pudieron extraer variables del mensaje.",
                    "detalle": str(extraction_error)
                }

            # Realizar predicción con el modelo
            resultado = predecir_y_mostrar_factores(
                input_parcial=input_parcial,
                modelo=modelo,
                columnas_modelo=columnas_modelo,
                columnas_numericas=columnas_numericas,
                df_normalize=df_normalize,
                scaler=scaler
            )
        else:
          resultado = None
          input_parcial = None

        return {
            "dialogflow": dialogflow_result,
            "prediccion": resultado
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/conversar-test")
async def conversar(request: Request):
    try:
        datos = await request.json()
        mensaje_usuario = datos.get("mensaje", "")
        
        if not mensaje_usuario:
            return {"error": "Falta el mensaje."}

        # Ejecutar Dialogflow para detectar intención
        dialogflow_result = detec_intent_texts_full(
            project_id=project_id,
            session_id=session_id,
            text=mensaje_usuario,
            language_code=language_code
        )

        if dialogflow_result["intencion"]=="compra":
            
            input_parcial =  {
                "admin_page_qty": 12,
                "product_page_qty": 42,
                "month_number": 2,
                "visitor_type": 1,
            }

            # Realizar predicción con el modelo
            resultado = predecir_y_mostrar_factores(
                input_parcial=input_parcial,
                modelo=modelo,
                columnas_modelo=columnas_modelo,
                columnas_numericas=columnas_numericas,
                df_normalize=df_normalize,
                scaler=scaler
            )
        else:
          resultado = None
          input_parcial = None

        return {
            "dialogflow": dialogflow_result,
            "prediccion": resultado
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def home():
    return JSONResponse(content="API para la predicción de posibles usuarios compradores.")