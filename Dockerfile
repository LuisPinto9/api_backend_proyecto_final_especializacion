# Usa una imagen base oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios
COPY . /app

# Asegura permisos de ejecución para el script de arranque
RUN chmod +x start.sh

# Actualiza pip y instala dependencias
RUN pip install --upgrade pip && \
    pip install \
        tensorflow-cpu==2.15.0 \
        tokenizers==0.14 \
        fastapi \
        uvicorn \
        python-multipart \
        numpy \
        pillow \
        requests \
        google-cloud-dialogflow \
        google-auth 

# Expone el puerto que usará la app (Google Cloud Run requiere el 8080)
EXPOSE 8080

# Comando por defecto al ejecutar el contenedor
CMD ["./start.sh"]
