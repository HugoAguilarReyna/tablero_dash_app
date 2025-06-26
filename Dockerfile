# ... (código anterior) ...

RUN conda install -y \
    python=3.9 \
    pandas=1.5.3 \
    numpy=1.23.5 \
    plotly=5.22.0 \
    dash=2.17.0 \
    prophet=0.7.1 \           # <--- ¡Versión 0.7.1 de Prophet!
    requests=2.31.0 \
    openpyxl=3.1.2 \
    gunicorn=22.0.0 \
    matplotlib=3.8.4 \
    lxml=5.4.0 \
    -c conda-forge && \
    conda clean --all -f -y

# ... (resto del código) ...