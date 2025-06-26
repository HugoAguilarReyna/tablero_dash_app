# Usar una imagen base de Miniconda
FROM continuumio/miniconda3

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar las dependencias críticas (pandas y prophet) con conda
# Esto usa los binarios precompilados de conda-forge, evitando problemas de compilación
RUN conda install -y \
    pandas=2.0.3 \
    prophet=1.1.1 \
    numpy=1.26.4 \
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