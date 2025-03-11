# Usa la imagen base de Python
FROM python:3.10-slim

# Define el directorio de trabajo
WORKDIR /app

# Copia los archivos de backend
COPY ../backend/ .

# Instala las dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expone el puerto
EXPOSE 5000

# Ejecuta Flask
CMD ["python", "app.py"]
