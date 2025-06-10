#!/bin/bash

# Ejecutar con uvicorn en modo producción
exec uvicorn run:app --host 0.0.0.0 --port 8080 --workers 4