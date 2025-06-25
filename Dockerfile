# Usar una imagen base de Python estable y compatible
FROM python:3.9-slim-buster

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias para algunas librerías
# Esto es crucial para pandas, numpy, scipy, matplotlib, prophet
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instalar las dependencias de Python
# Instala gunicorn primero para asegurar que está disponible como entrada en el PATH
RUN pip install gunicorn && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto de tu aplicación
COPY . .

# Exponer el puerto en el que correrá la aplicación
EXPOSE 8000

# Comando para iniciar la aplicación cuando el contenedor se ejecute
CMD ["gunicorn", "dashboard_app:server", "--bind", "0.0.0.0:8000"]