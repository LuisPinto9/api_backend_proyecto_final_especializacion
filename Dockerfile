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
        fastapi \
        uvicorn \
        pandas \
        scikit-learn \
        numpy \
        shap \
        xgboost \
        google-cloud-dialogflow \
        google-auth \
        openia

# Expone el puerto que usará la app
EXPOSE 8080

# Comando por defecto al ejecutar el contenedor
CMD ["./start.sh"]
