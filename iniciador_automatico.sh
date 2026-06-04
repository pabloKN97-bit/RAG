#!/bin/bash

# Guardar la ruta actual del proyecto RAG de Pablo
RUTA_PROYECTO="$HOME/proyecto_ia/RAG"

echo " Iniciando el entorno del proyecto RAG..."

# 1. Terminal para el BACKEND (Activa el entorno virtual y arranca main.py)
gnome-terminal --title=" BACKEND (FastAPI/Ollama)" -- bash -c "
    cd '$RUTA_PROYECTO';
    source env/bin/activate;
    python3 main.py;
    exec bash
"

# 2. Terminal para el FRONTEND (Arranca el servidor web en el puerto 8080 y abre Firefox)
gnome-terminal --title=" FRONTEND (Servidor Web)" -- bash -c "
    cd '$RUTA_PROYECTO';
    echo 'iniciando servidor web local';
    python3 -m http.server 8080 &
    sleep 2;
    xdg-open 'http://localhost:8080/frontend.html';
    exec bash
"

echo " furula perfecto."
