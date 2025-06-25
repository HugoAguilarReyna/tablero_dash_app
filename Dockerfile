# Usar una imagen base de Miniconda
# Esta imagen ya tiene Python, Conda y muchas dependencias científicas preinstaladas o fáciles de instalar
FROM continuumio/miniconda3

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo environment.yml que creamos antes
# Si lo borraste en pasos anteriores, tendrás que recrearlo o usar solo requirements.txt con conda install
# Para simplificar, vamos a usar directamente pip install para las dependencias.
# NO necesitas environment.yml si solo usas pip install
# Vamos a pegar tu requirements.txt directamente

# Copiar el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instalar las dependencias de Python usando pip (dentro del entorno conda)
# Conda viene con pip, y pip puede instalar los paquetes que Conda no tiene directamente.
# Esto es generalmente más robusto que compilar desde cero.
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copiar el resto de tu aplicación
COPY . .

# Exponer el puerto en el que correrá la aplicación
EXPOSE 8000

# Comando para iniciar la aplicación cuando el contenedor se ejecute
CMD ["gunicorn", "dashboard_app:server", "--bind", "0.0.0.0:8000"]