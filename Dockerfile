# Usar una imagen base de Miniconda
FROM continuumio/miniconda3

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar las dependencias principales usando conda (desde conda-forge para binarios precompilados)
# y especificar explícitamente Python 3.9, una versión muy estable.
# Se ha eliminado 'prophet' de la lista de instalación.
RUN conda install -y \
    python=3.9 \             # Forzar Python 3.9 para compatibilidad
    pandas=1.4.4 \           # Versión de Pandas más antigua y robusta para evitar problemas de compilación
    numpy=1.23.5 \           # Versión de NumPy compatible con Pandas 1.4.4 y Python 3.9
    plotly=5.22.0 \
    dash=2.17.0 \
    requests=2.31.0 \
    openpyxl=3.1.2 \
    gunicorn=22.0.0 \
    matplotlib=3.8.4 \
    lxml=5.4.0 \
    -c conda-forge && \
    conda clean --all -f -y

# Copiar el resto de tu aplicación
COPY . .

# Exponer el puerto en el que correrá la aplicación
EXPOSE 8000

# Comando para iniciar la aplicación cuando el contenedor se ejecute
CMD ["gunicorn", "dashboard_app:server", "--bind", "0.0.0.0:8000"]
