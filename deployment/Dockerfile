# Usa la imagen base de Python 3.10
FROM python:3.10-slim

# Define el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todos los archivos al contenedor
COPY ../backend/ .

# Instala las dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto 5000 para Flask
EXPOSE 5000

# Comando para iniciar Flask
CMD ["python", "app.py"]
