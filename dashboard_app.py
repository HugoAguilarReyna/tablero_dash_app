# -*- coding: utf-8 -*-

# ==============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib as mpl
from dash import dash_table
import io
import base64
import json # Necesario para guardar y cargar JSON (aunque no para este caso específico, es útil)
from prophet import Prophet # Importar Prophet para Machine Learning


# ==============================================================================
# 2. CONFIGURACIÓN DE ESTILO Y COLORES
# ==============================================================================
def colorFader(c1,c2,mix=0):
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

colors = {}
colors['primary_accent'] = '#00EE87' # Verde vibrante

colors['level1'] = "#00EE87"
colors['level2'] = "#2ECC71"
colors['level3'] = "#27AE60"
colors['level4'] = "#1ABC9C"
colors['level5'] = "#00C78C"

colors['background'] = '#1E1E1E'
colors['text'] = '#FFFFFF'
colors['grid'] = '#4A4A4A'
colors['background_divs'] = '#2A2A2A'
colors['neon_yellow'] = '#CCFF00' # Amarillo Neón añadido aquí
colors['bright_blue'] = '#00BFFF' # Azul brillante para mayor contraste

MOTIVUS_LOGO_URL = "/assets/logo-dark.svg"


# ==============================================================================
# 3. CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================
print("Cargando y procesando datos...")

df_holi = None
df_oil = None
df_stores = None
df_trans = None
df_train = None
df_full = None
df_geo = None # Nuevo DataFrame para el catálogo geográfico
data_loaded_successfully = False

try:
    # df_holi = pd.read_csv(r'D:\DEVS\Dash y Plotly\INPUTS\holidays_events.csv')
    # df_oil = pd.read_csv(r'D:\DEVS\Dash y Plotly\INPUTS\oil.csv')
    # df_stores = pd.read_csv(r'D:\DEVS\Dash y Plotly\INPUTS\stores.csv')
    # df_trans = pd.read_csv(r'D:\DEVS\Dash y Plotly\INPUTS\transactions.csv')
    # df_train = pd.read_csv(r'D:\DEVS\Dash y Plotly\INPUTS\train.csv')
    
    df_holi = pd.read_csv('data/holidays_events.csv')
    df_oil = pd.read_csv('data/oil.csv')
    df_stores = pd.read_csv('data/stores.csv')
    df_trans = pd.read_csv('data/transactions.csv')
    df_train = pd.read_csv('data/train.csv')

    # Cargar el nuevo catálogo geográfico
    # Asegúrate de que este archivo exista en la ruta especificada
    # df_geo = pd.read_excel(r'D:\DEVS\Dash y Plotly\INPUTS\cat_geo.xlsx')
    df_geo = pd.read_excel('data/cat_geo.xlsx') # También para el archivo Excel
    # Renombrar columnas para la unión si es necesario
    df_geo = df_geo.rename(columns={
        'Ciudad': 'city',
        'Estado/Provincia': 'state',
        'Latitud': 'latitude',
        'Longitud': 'longitude'
    })


    df_full = df_train.merge(df_holi, on='date', how='left')
    df_full = df_full.merge(df_oil, on='date', how='left')
    df_full = df_full.merge(df_stores, on='store_nbr', how='left')
    df_full = df_full.merge(df_trans, on=['date', 'store_nbr'], how='left')
    df_full = df_full.rename(columns={"type_x": "holiday_type", "type_y": "store_type"})

    # Unir df_full con df_geo para obtener latitud y longitud
    # Usamos left merge para mantener todas las filas de df_full
    # y traer las coordenadas donde haya coincidencias de ciudad y estado.
    df_full = df_full.merge(df_geo[['city', 'state', 'latitude', 'longitude']],
                            on=['city', 'state'],
                            how='left')


    df_full['date'] = pd.to_datetime(df_full['date'])
    df_full.sort_values(by=['date'], inplace=True, ascending=True)
    df_full['year'] = df_full['date'].dt.year
    df_full['month'] = df_full['date'].dt.month
    df_full['day_of_week'] = df_full['date'].dt.day_name()

    df_full['dcoilwtico'] = df_full['dcoilwtico'].ffill().bfill()
    df_full['transactions'] = df_full['transactions'].fillna(0)
    df_full['holiday_type'] = df_full['holiday_type'].fillna('Ninguno')
    df_full['locale_name'] = df_full['locale_name'].fillna('Desconocido')

    print("Datos cargados exitosamente.")
    data_loaded_successfully = True

except FileNotFoundError as e:
    print(f"Error CRÍTICO: No se encontró el archivo {e.filename}. Por favor, asegúrate de que los archivos CSV y cat_geo.xlsx estén en la carpeta correcta. El dashboard no se iniciará.")
except Exception as e:
    print(f"Ocurrió un error inesperado durante la carga o procesamiento de datos: {e}. El dashboard no se iniciará.")


# ==============================================================================
# 4. FUNCIÓN PARA GENERAR EL CATÁLOGO EN EXCEL (Herramienta externa)
# ==============================================================================
# Este código se ejecutaría una vez para generar el archivo, no es parte de la app Dash
def generate_catalog_excel():
    data = [
        ['Ecuador', 'Quito', 'Pichincha', -0.1807, -78.4678],
        ['Ecuador', 'Cuenca', 'Azuay', -2.9000, -79.0058],
        ['Ecuador', 'Machala', 'El Oro', -3.2586, -79.9610],
        ['Ecuador', 'Esmeraldas', 'Esmeraldas', 0.9667, -79.6583],
        ['Ecuador', 'Libertad', 'Guayas', -2.2333, -80.9167],
        ['Ecuador', 'Playas', 'Guayas', -2.6367, -80.3853],
        ['Ecuador', 'Guayaquil', 'Guayas', -2.1962, -79.8862],
        ['Ecuador', 'Loja', 'Loja', -3.9931, -79.2042],
        ['Ecuador', 'El Carmen', 'Manabi', -0.2847, -79.4589],
        ['Ecuador', 'Manta', 'Manabi', -0.9500, -80.7000],
        ['Ecuador', 'Ambato', 'Tungurahua', -1.2417, -78.6186],
        ['Ecuador', 'Santo Domingo', 'Santo Domingo de los Tsachilas', -0.2520, -79.1754],
        ['Ecuador', 'Quevedo', 'Los Rios', -1.0253, -79.4627],
        ['Ecuador', 'Guaranda', 'Bolivar', -1.5833, -78.9833],
        ['Ecuador', 'Ibarra', 'Imbabura', 0.3524, -78.1180],
        ['Ecuador', 'Cayambe', 'Pichincha', 0.0400, -78.1400],
        ['Ecuador', 'Latacunga', 'Cotopaxi', -0.9333, -78.6167],
        ['Ecuador', 'Riobamba', 'Chimborazo', -1.6708, -78.6472],
        ['Ecuador', 'Babahoyo', 'Los Rios', -1.8000, -79.4833],
        ['Ecuador', 'Puyo', 'Pastaza', -1.4833, -77.9667],
        ['Ecuador', 'Daule', 'Guayas', -1.8833, -80.0167],
        ['Ecuador', 'Salinas', 'Santa Elena', -2.2139, -80.9575]
    ]
    df_catalog = pd.DataFrame(data, columns=['País', 'Ciudad', 'Estado/Provincia', 'Latitud', 'Longitud'])
    output_path = r'D:\DEVS\Dash y Plotly\INPUTS\cat_geo.xlsx'
    try:
        df_catalog.to_excel(output_path, index=False)
        print(f"Catálogo generado exitosamente en: {output_path}")
    except Exception as e:
        print(f"Error al generar el catálogo: {e}")

# Ejecutar esta función si deseas generar el archivo Excel por primera vez
# generate_catalog_excel()


# ==============================================================================
# 5. INICIALIZACIÓN Y LAYOUT DE LA APLICACIÓN DASH
# ==============================================================================
if data_loaded_successfully:
    app = dash.Dash(__name__)
    app.title = "Dashboard de Ventas"

    # Preparar datos para Prophet (antes del layout para que se entrene una sola vez)
    # Agregamos las ventas a nivel diario para el pronóstico
    df_prophet = df_full.groupby('date')['sales'].sum().reset_index()
    df_prophet = df_prophet.rename(columns={'date': 'ds', 'sales': 'y'})

    # Entrenar el modelo Prophet
    # Se puede añadir seasonality_mode='multiplicative' o holidays si es necesario
    m = Prophet()
    try:
        m.fit(df_prophet)
        print("Modelo Prophet entrenado exitosamente.")
    except Exception as e:
        print(f"Error al entrenar el modelo Prophet: {e}")
        m = None # Asegurar que m sea None si el entrenamiento falla


    app.layout = html.Div(style={'backgroundColor': colors['background'], 'color': colors['text'], 'fontFamily': 'Arial, sans-serif'}, children=[

        html.Div(
            html.Img(src=MOTIVUS_LOGO_URL, alt="Motivus Logo",
                     style={'height': '20px', 'width': 'auto', 'margin-left': '20px'}),
            style={
                'backgroundColor': colors['background_divs'],
                'padding': '10px',
                'borderRadius': '5px',
                'margin': '10px',
                'display': 'flex',
                'justifyContent': 'flex-start',
                'alignItems': 'center'
            }
        ),

        html.H1("Dashboard Interactivo de Ventas de Tiendas", style={'textAlign': 'center', 'padding': '20px', 'color': colors['primary_accent']}),

        html.Div(id='summary-card', children=[
            html.Div("Cargando resumen de datos...", style={'textAlign': 'center', 'padding': '50px'})
        ], style={'padding': '10px', 'backgroundColor': colors['background_divs'], 'borderRadius': '5px', 'margin': '10px'}),

        html.Div([
            html.Div([
                html.Label("Seleccionar Año:", style={'fontWeight': 'bold', 'color': colors['text']}),
                dcc.Dropdown(
                    id='filtro-ano',
                    options=[{'label': str(year), 'value': year} for year in sorted(df_full['year'].unique())],
                    value=df_full['year'].unique().max(),
                    clearable=False,
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.Label("Seleccionar Tipo de Tienda:", style={'fontWeight': 'bold', 'color': colors['text']}),
                dcc.Dropdown(
                    id='filtro-tipo-tienda',
                    options=[{'label': s_type, 'value': s_type} for s_type in sorted(df_full['store_type'].unique())],
                    value=list(df_full['store_type'].unique()),
                    multi=True,
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right', 'verticalAlign': 'top'})
        ], style={'padding': '20px', 'backgroundColor': colors['background_divs'], 'borderRadius': '5px', 'margin': '10px'}),

        dcc.Tabs(id="tabs-graficas", value='tab-general', children=[
            dcc.Tab(label='Análisis General de Ventas', value='tab-general', children=[
                dcc.Graph(id='grafica-familia-cluster'),
                dcc.Graph(id='grafica-ciudades'),
                dcc.Graph(id='grafica-clusters'),
                dcc.Graph(id='grafica-ventas-por-locale'),
                dcc.Graph(id='grafica-promociones')
            ]),
            dcc.Tab(label='Análisis Temporal', value='tab-temporal', children=[
                dcc.Graph(id='grafica-ventas-mensuales'),
                dcc.Graph(id='grafica-ventas-diarias'),
                dcc.Graph(id='grafica-oil-sales')
            ]),
            dcc.Tab(label='Análisis por Feriados', value='tab-feriados', children=[dcc.Graph(id='grafica-feriados-tienda')]),
            # --- NUEVA PESTAÑA: MAPA DE UBICACIONES ---
            dcc.Tab(label='Mapa de Ubicaciones', value='tab-mapa', children=[
                html.Div([
                    html.H3("Ubicaciones de Tiendas en el Mapa", style={'textAlign': 'center', 'color': colors['text'], 'paddingTop': '20px'}),
                    dcc.Graph(id='mapa-ubicaciones')
                ], style={'padding': '10px', 'backgroundColor': colors['background_divs'], 'borderRadius': '5px', 'margin': '10px'})
            ]),
            # --- NUEVA PESTAÑA: PREDICCIÓN DE VENTAS (ML) ---
            dcc.Tab(label='Predicción de Ventas (ML)', value='tab-ml-prediction', children=[
                html.Div([
                    html.H3("Histograma de Predicción de Ventas para los Próximos Meses", style={'textAlign': 'center', 'color': colors['text'], 'paddingTop': '20px'}),
                    html.Div([
                        html.Label("Número de Meses a Pronosticar:", style={'fontWeight': 'bold', 'color': colors['text']}),
                        dcc.Slider(
                            id='num-meses-slider',
                            min=1,
                            max=12,
                            step=1,
                            value=6, # Por defecto 6 meses
                            marks={i: str(i) for i in range(1, 13)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '80%', 'margin': '20px auto'}),
                    dcc.Graph(id='histograma-prediccion-ventas')
                ], style={'padding': '10px', 'backgroundColor': colors['background_divs'], 'borderRadius': '5px', 'margin': '10px'})
            ]),
            # --- NUEVA PESTAÑA: ANÁLISIS POR TIENDA INDIVIDUAL ---
            dcc.Tab(label='Análisis por Tienda', value='tab-store-analysis', children=[
                html.Div([
                    html.H3("Análisis Detallado por Tienda Individual", style={'textAlign': 'center', 'color': colors['text'], 'paddingTop': '20px'}),
                    html.Div([
                        html.Label("Seleccionar Tienda:", style={'fontWeight': 'bold', 'color': colors['text']}),
                        dcc.Dropdown(
                            id='filtro-tienda-individual',
                            options=[{'label': str(store_nbr), 'value': store_nbr} for store_nbr in sorted(df_full['store_nbr'].unique())],
                            value=df_full['store_nbr'].unique()[0], # Selecciona la primera tienda por defecto
                            clearable=False,
                        ),
                    ], style={'width': '48%', 'margin': '20px auto 10px auto'}),
                    dcc.Graph(id='grafica-tienda-tendencia'),
                    dcc.Graph(id='grafica-tienda-familias'),
                    dcc.Graph(id='grafica-tienda-comparativa')
                ], style={'padding': '10px', 'backgroundColor': colors['background_divs'], 'borderRadius': '5px', 'margin': '10px'})
            ]),
            # --- NUEVA PESTAÑA: DESCARGA DE DATOS ---
            dcc.Tab(label='Descarga de Datos', value='tab-descarga', children=[
                html.Div([
                    html.H3("Datos Filtrados para Descarga", style={'textAlign': 'center', 'color': colors['text'], 'paddingTop': '20px'}),
                    html.Button("Descargar Datos en Excel", id="boton-descargar-excel", n_clicks=0,
                                style={'backgroundColor': colors['primary_accent'], 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer',
                                       'margin': '10px auto 20px auto', 'display': 'block', 'fontSize': '16px'}),
                    dcc.Download(id="download-excel"),
                    dash_table.DataTable(
                        id='tabla-datos',
                        columns=[],
                        data=[],
                        page_size=10,
                        sort_action='native',
                        filter_action='native',
                        style_table={'overflowX': 'auto', 'margin': '20px auto', 'width': '95%', 'borderRadius': '5px'},
                        style_header={
                            'backgroundColor': colors['background_divs'],
                            'color': 'white',
                            'fontWeight': 'bold',
                            'border': '1px solid ' + colors['grid']
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': colors['background_divs']
                            },
                            {
                                'if': {'row_index': 'even'},
                                'backgroundColor': colors['background']
                            }
                        ],
                        style_cell={
                            'fontFamily': 'Arial, sans-serif',
                            'color': colors['text'],
                            'textAlign': 'left',
                            'backgroundColor': colors['background_divs'],
                            'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
                            'whiteSpace': 'normal',
                            'border': '1px solid ' + colors['grid']
                        },
                        export_format='xlsx',
                        export_headers='display',
                        fill_width=True
                    )
                ], style={'padding': '10px', 'backgroundColor': colors['background_divs'], 'borderRadius': '5px', 'margin': '10px'})
            ]),
        ], colors={
            "border": colors['grid'],
            "primary": colors['primary_accent'],
            "background": colors['background_divs']
        })
    ])

    # ==============================================================================
    # 6. CALLBACKS (La lógica interactiva)
    # ==============================================================================
    @app.callback(
        [Output('summary-card', 'children'),
         Output('grafica-familia-cluster', 'figure'),
         Output('grafica-ventas-mensuales', 'figure'),
         Output('grafica-ventas-diarias', 'figure'),
         Output('grafica-feriados-tienda', 'figure'),
         Output('grafica-ciudades', 'figure'),
         Output('grafica-clusters', 'figure'),
         Output('grafica-oil-sales', 'figure'),
         Output('grafica-ventas-por-locale', 'figure'),
         Output('grafica-promociones', 'figure'),
         Output('mapa-ubicaciones', 'figure'),
         Output('histograma-prediccion-ventas', 'figure'),
         Output('grafica-tienda-tendencia', 'figure'), # Nuevo Output
         Output('grafica-tienda-familias', 'figure'), # Nuevo Output
         Output('grafica-tienda-comparativa', 'figure'), # Nuevo Output
         Output('tabla-datos', 'data'),
         Output('tabla-datos', 'columns')],
        [Input('filtro-ano', 'value'),
         Input('filtro-tipo-tienda', 'value'),
         Input('num-meses-slider', 'value'),
         Input('filtro-tienda-individual', 'value')] # Nuevo Input
    )
    def actualizar_graficas(ano_seleccionado, tipo_tienda_seleccionado, num_meses_pronostico, tienda_individual_seleccionada):

        if not isinstance(tipo_tienda_seleccionado, list):
            tipo_tienda_seleccionado = [tipo_tienda_seleccionado]

        dff = df_full[(df_full['year'] == ano_seleccionado) & (df_full['store_type'].isin(tipo_tienda_seleccionado))].copy()

        # Inicializar figuras vacías para el retorno en caso de dff.empty
        empty_figure = go.Figure()
        empty_figure.update_layout(template="plotly_dark", paper_bgcolor=colors['background_divs'], plot_bgcolor=colors['background_divs'])

        if dff.empty:
            empty_summary_content = html.Div(
                "No hay datos disponibles para la selección actual.",
                style={'textAlign': 'center', 'padding': '50px', 'color': colors['text']}
            )
            return (empty_summary_content,
                    empty_figure, empty_figure, empty_figure, empty_figure,
                    empty_figure, empty_figure, empty_figure, empty_figure, empty_figure,
                    empty_figure, # Figura vacía para el mapa
                    empty_figure, # Figura vacía para el histograma de predicción
                    empty_figure, empty_figure, empty_figure, # Figuras vacías para análisis de tienda
                    [], []
                   )


        stores_num = dff['store_nbr'].nunique()
        type_store_num = dff['store_type'].nunique()
        product_num = dff['family'].nunique()
        cities_num = dff['city'].nunique()
        state_num = dff['state'].nunique()
        cluster_num = dff['cluster'].nunique()
        total_sales = dff['sales'].sum()
        total_transactions = dff['transactions'].sum()

        first_date = dff["date"].min().strftime("%Y-%m-%d")
        last_date = dff["date"].max().strftime("%Y-%m-%d")

        summary_metrics_data = [
            ("Tiendas", stores_num),
            ("Ciudades", cities_num),
            ("Estados", state_num),
            ("Tipos de Tienda", type_store_num),
            ("Productos", product_num),
            ("Clusters", cluster_num),
            ("Total Ventas", f"{total_sales:,.2f}"),
            ("Total Transacciones", f"{total_transactions:,.0f}")
        ]

        summary_children = []
        for label, value in summary_metrics_data:
            summary_children.append(
                html.Div([
                    html.P(label, style={'margin': '0', 'fontSize': '16px', 'color': colors['text']}),
                    html.P(html.B(str(value)), style={'margin': '0', 'fontSize': '24px', 'color': colors['primary_accent']})
                ], style={'textAlign': 'center', 'minWidth': '80px', 'margin': '5px'})
            )

        updated_summary_content = html.Div([
            html.Div("Resumen de Datos", style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold', 'paddingBottom': '15px', 'color': colors['text']}),
            html.Div(summary_children, style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'flex-start', 'flexWrap': 'wrap', 'padding': '5px', 'gap': '10px'}),
            html.Div(f"Rango de datos: desde {first_date} hasta {last_date}", style={'textAlign': 'center', 'fontSize': '14px', 'paddingTop': '15px', 'color': colors['text']})
        ], style={'height': '250px', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'})


        df_fa_sa = dff.groupby('family').agg({"sales": "mean"}).reset_index().sort_values(by='sales', ascending=False)[:10]
        df_st_sa = dff.groupby('store_type').agg({"sales": "mean"}).reset_index().sort_values(by='sales', ascending=False)
        fig1 = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]], column_widths=[0.65, 0.35], subplot_titles=("Top 10 Familias de Productos", "Ventas por Tipo de Tienda"))
        fig1.add_trace(go.Bar(x=df_fa_sa['sales'], y=df_fa_sa['family'], orientation='h', marker_color=colors['level2'], name='Ventas promedio'), row=1, col=1)
        fig1.add_trace(go.Pie(values=df_st_sa['sales'], labels=df_st_sa['store_type'], hole=0.7,
                              marker_colors=[colors[f'level{i % 5 + 1}'] for i in range(len(df_st_sa))],
                              name='Tipo de Tienda'), row=1, col=2)
        fig1.update_layout(title_text="Análisis de Ventas Promedio por Familia de Productos y Tipo de Tienda", template="plotly_dark", showlegend=False, paper_bgcolor=colors['background_divs'], plot_bgcolor=colors['background_divs'])
        fig1.update_yaxes(categoryorder='total ascending', row=1, col=1, title_text="Familia de Productos")
        fig1.update_xaxes(title_text="Ventas Promedio", row=1, col=1)

        df_m_sa = dff.groupby('month').agg({"sales": "mean"}).reset_index()
        fig2 = px.line(df_m_sa, x='month', y='sales', title=f"Evolución de Ventas Promedio Mensuales en {ano_seleccionado}", markers=True, labels={'month': 'Mes', 'sales': 'Ventas Promedio'})
        fig2.update_traces(line_color=colors['level1'], marker=dict(color=colors['level3'], size=8))
        fig2.update_xaxes(tickvals=list(range(1, 13)), ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], title_text="Mes")
        fig2.update_yaxes(title_text="Ventas Promedio")
        fig2.update_layout(template="plotly_dark", paper_bgcolor=colors['background_divs'], plot_bgcolor=colors['background_divs'])

        df_dw_sa = dff.groupby('day_of_week').agg({"sales": "mean"}).reset_index()
        dias_ordenados = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fig3 = px.bar(df_dw_sa, y='sales', x='day_of_week', text='sales', title=f"Ventas Promedio por Día de la Semana en {ano_seleccionado}", category_orders={'day_of_week': dias_ordenados}, labels={'day_of_week': 'Día de la Semana', 'sales': 'Ventas Promedio'}, color_discrete_sequence=[colors['level4']])
        fig3.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig3.update_layout(template="plotly_dark", uniformtext_minsize=8, paper_bgcolor=colors['background_divs'], plot_bgcolor=colors['background_divs'],
                           xaxis_title="Día de la Semana", yaxis_title="Ventas Promedio")

        df_st_ht = dff.groupby(['store_type', 'holiday_type']).agg({"sales": "mean"}).reset_index()
        fig4 = px.scatter(df_st_ht, x='store_type', y='holiday_type', size='sales', color='sales', title="Ventas Promedio: Tipo de Tienda vs Tipo de Feriado", labels={'Tipo de Tienda': 'Tipo de Tienda', 'holiday_type': 'Tipo de Feriado', 'sales': 'Ventas Promedio'}, size_max=40, color_continuous_scale=px.colors.sequential.Greens)
        fig4.update_layout(template="plotly_dark", paper_bgcolor=colors['background_divs'], plot_bgcolor=colors['background_divs'],
                           xaxis_title="Tipo de Tienda", yaxis_title="Tipo de Feriado")

        df_city_sales = dff.groupby('city').agg({"sales": "mean"}).reset_index().sort_values(by='sales', ascending=False)[:15]
        fig5 = px.bar(df_city_sales, x='sales', y='city', orientation='h', title=f"Top 15 Ciudades por Ventas Promedio en {ano_seleccionado}",
                      labels={'sales': 'Ventas Promedio', 'city': 'Ciudad'}, color_discrete_sequence=[colors['level5']])
        fig5.update_yaxes(categoryorder='total ascending')
        fig5.update_layout(template="plotly_dark", paper_bgcolor=colors['background_divs'], plot_bgcolor=colors['background_divs'],
                           xaxis_title="Ventas Promedio", yaxis_title="Ciudad")

        df_cluster_sales = dff.groupby('cluster').agg({"sales": "mean"}).reset_index().sort_values(by='sales', ascending=False)
        fig6 = px.bar(df_cluster_sales, x='cluster', y='sales', title=f"Ventas Promedio por Clúster en {ano_seleccionado}",
                      labels={'sales': 'Ventas Promedio', 'cluster': 'Clúster'}, color_discrete_sequence=[colors['level1']])
        fig6.update_layout(template="plotly_dark", paper_bgcolor=colors['background_divs'], plot_bgcolor=colors['background_divs'],
                           xaxis_title="Clúster", yaxis_title="Ventas Promedio")

        df_oil_sales = dff.groupby('date').agg(
            {"sales": "mean", "dcoilwtico": "mean"}
        ).reset_index().sort_values(by='date')

        fig7 = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        fig7.add_trace(go.Scatter(x=df_oil_sales['date'], y=df_oil_sales['sales'], name='Ventas Promedio', line=dict(color=colors['primary_accent'])), secondary_y=False,)
        # CAMBIO DE COLOR AQUÍ: Precio Petróleo (USD) a amarillo neón
        fig7.add_trace(go.Scatter(x=df_oil_sales['date'], y=df_oil_sales['dcoilwtico'], name='Precio Petróleo (USD)', line=dict(color=colors['neon_yellow'])), secondary_y=True,)
        fig7.update_layout(
            title_text=f"Ventas Promedio Diarias vs. Precio del Petróleo en {ano_seleccionado}",
            template="plotly_dark",
            paper_bgcolor=colors['background_divs'],
            plot_bgcolor=colors['background_divs'],
            legend_title_text="Leyenda"
        )
        fig7.update_xaxes(title_text="Fecha")
        fig7.update_yaxes(title_text="Ventas Promedio", secondary_y=False)
        fig7.update_yaxes(title_text="Precio Petróleo (USD)", secondary_y=True)

        df_locale_sales = dff.groupby('locale_name').agg({"sales": "mean"}).reset_index().sort_values(by='sales', ascending=False)[:15]
        fig8 = px.bar(df_locale_sales, x='sales', y='locale_name', orientation='h',
                      title=f"Top 15 Localidades por Ventas Promedio en {ano_seleccionado}",
                      labels={'sales': 'Ventas Promedio', 'locale_name': 'Localidad'},
                      color_discrete_sequence=[colors['level3']])
        fig8.update_yaxes(categoryorder='total ascending')
        fig8.update_layout(template="plotly_dark", paper_bgcolor=colors['background_divs'], plot_bgcolor=colors['background_divs'],
                           xaxis_title="Ventas Promedio", yaxis_title="Localidad")

        df_promo_sales = dff.groupby('onpromotion').agg({"sales": "sum"}).reset_index()
        df_promo_sales['onpromotion_label'] = df_promo_sales['onpromotion'].apply(lambda x: 'En Promoción' if x == 1 else 'Sin Promoción')
        fig9 = px.bar(df_promo_sales, x='onpromotion_label', y='sales',
                      title=f"Ventas Totales: En Promoción vs. Sin Promoción en {ano_seleccionado}",
                      labels={'sales': 'Ventas Totales', 'onpromotion_label': 'Estado de Promoción'},
                      color_discrete_sequence=[colors['primary_accent'], colors['level4']])
        fig9.update_layout(template="plotly_dark", paper_bgcolor=colors['background_divs'], plot_bgcolor=colors['background_divs'],
                           xaxis_title="Estado de Promoción", yaxis_title="Ventas Totales")

        # --- Generar el mapa ---
        df_map = dff.groupby(['city', 'state', 'latitude', 'longitude', 'store_type']).agg(
            sales_mean=('sales', 'mean'),
            store_count=('store_nbr', 'nunique')
        ).reset_index()

        df_map.dropna(subset=['latitude', 'longitude'], inplace=True)

        fig_map = px.scatter_mapbox(
            df_map,
            lat="latitude",
            lon="longitude",
            color="store_type", # Color por tipo de tienda
            size="sales_mean", # Tamaño por ventas promedio
            hover_name="city",
            hover_data={"state": True, "sales_mean": ":.2f", "store_count": True, "latitude": False, "longitude": False},
            zoom=5.5, # Ajustar el zoom inicial para Ecuador
            center={"lat": -1.831239, "lon": -78.183406}, # Centro aproximado de Ecuador
            mapbox_style="carto-positron", # Estilo oscuro para el mapa (gris tenue)
            title=f"Ubicación de Tiendas y Ventas Promedio en {ano_seleccionado}",
            color_discrete_sequence=[colors[f'level{i % 5 + 1}'] for i in range(len(df_map['store_type'].unique()))] # Usar tus colores level
        )
        fig_map.update_layout(
            margin={"r":0,"t":50,"l":0,"b":0},
            paper_bgcolor=colors['background_divs'],
            plot_bgcolor=colors['background_divs'],
            font_color=colors['text'] # Color del texto blanco, para que resalte sobre el fondo oscuro
        )
        # =======================================================================================================
        # IMPORTANTE: Para que los mapas base de Mapbox se carguen, necesitas un token de acceso.
        # Regístrate en mapbox.com, obtén tu token público y DESCOMENTA la siguiente línea, reemplazando "TU_TOKEN_DE_MAPOBOX_AQUI"
        # por tu token real. Sin este token, el mapa base no se mostrará.
        # =======================================================================================================
        # px.set_mapbox_access_token("TU_TOKEN_DE_MAPOBOX_AQUI")


        # --- Generar el histograma de predicción de ventas ---
        fig_prediccion_ventas = empty_figure # Figura vacía por defecto
        if m is not None:
            future = m.make_future_dataframe(periods=num_meses_pronostico * 30, freq='D') # Días para el pronóstico
            forecast = m.predict(future)
            
            # Filtrar pronóstico solo para el futuro (después de la última fecha histórica)
            last_historical_date = df_prophet['ds'].max()
            forecast_future = forecast[forecast['ds'] > last_historical_date].copy()
            
            # Para un histograma de "predicción de ventas para los próximos meses",
            # podemos agrupar las predicciones diarias en mensuales o simplemente
            # tomar la distribución de las predicciones diarias futuras.
            # Vamos a crear un histograma de la distribución de 'yhat' (predicción)
            # para todo el periodo futuro.
            if not forecast_future.empty:
                fig_prediccion_ventas = px.histogram(
                    forecast_future, 
                    x='yhat', 
                    nbins=20, # Número de bins para el histograma
                    title=f"Distribución de Ventas Diarias Predichas (Próximos {num_meses_pronostico} meses)",
                    labels={'yhat': 'Ventas Diarias Predichas'},
                    color_discrete_sequence=[colors['primary_accent']]
                )
                fig_prediccion_ventas.update_layout(
                    template="plotly_dark",
                    paper_bgcolor=colors['background_divs'],
                    plot_bgcolor=colors['background_divs'],
                    xaxis_title="Ventas Diarias Predichas",
                    yaxis_title="Frecuencia"
                )
            else:
                fig_prediccion_ventas = go.Figure().update_layout(
                    title="No hay datos futuros para la predicción de ventas.",
                    template="plotly_dark",
                    paper_bgcolor=colors['background_divs'],
                    plot_bgcolor=colors['background_divs']
                )

        # --- Gráficos para el análisis de tienda individual ---
        fig_tienda_tendencia = empty_figure
        fig_tienda_familias = empty_figure
        fig_tienda_comparativa = empty_figure

        if tienda_individual_seleccionada is not None:
            df_tienda = df_full[df_full['store_nbr'] == tienda_individual_seleccionada].copy()

            if not df_tienda.empty:
                # 1. Tendencia de Ventas y Transacciones por Tienda
                df_tienda_diario = df_tienda.groupby('date').agg({'sales': 'sum', 'transactions': 'sum'}).reset_index()
                fig_tienda_tendencia = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
                fig_tienda_tendencia.add_trace(
                    go.Scatter(x=df_tienda_diario['date'], y=df_tienda_diario['sales'], mode='lines', name='Ventas', line=dict(color=colors['neon_yellow'])),
                    secondary_y=False,
                )
                # CAMBIO DE COLOR AQUÍ: Transacciones a azul brillante
                fig_tienda_tendencia.add_trace(
                    go.Scatter(x=df_tienda_diario['date'], y=df_tienda_diario['transactions'], mode='lines', name='Transacciones', line=dict(color=colors['level1'])),
                    secondary_y=True,
                )
                fig_tienda_tendencia.update_layout(
                    title_text=f"Tendencia Diaria de Ventas y Transacciones para Tienda {tienda_individual_seleccionada} en {ano_seleccionado}",
                    template="plotly_dark",
                    paper_bgcolor=colors['background_divs'],
                    plot_bgcolor=colors['background_divs'],
                    legend_title_text="Métrica"
                )
                fig_tienda_tendencia.update_xaxes(title_text="Fecha")
                fig_tienda_tendencia.update_yaxes(title_text="Ventas", secondary_y=False)
                fig_tienda_tendencia.update_yaxes(title_text="Transacciones", secondary_y=True)

                # 2. Ventas por Familia de Productos por Tienda
                df_tienda_familias = df_tienda.groupby('family')['sales'].sum().reset_index().sort_values(by='sales', ascending=False)
                fig_tienda_familias = px.bar(
                    df_tienda_familias,
                    x='sales',
                    y='family',
                    orientation='h',
                    title=f"Ventas por Familia de Productos para Tienda {tienda_individual_seleccionada} en {ano_seleccionado}",
                    labels={'sales': 'Ventas Totales', 'family': 'Familia de Productos'},
                    color_discrete_sequence=[colors['level2']]
                )
                fig_tienda_familias.update_layout(
                    template="plotly_dark",
                    paper_bgcolor=colors['background_divs'],
                    plot_bgcolor=colors['background_divs'],
                    yaxis={'categoryorder':'total ascending'}
                )

                # 3. Comparación de Ventas de Tienda vs Promedios
                tienda_sales_avg = df_tienda['sales'].mean()
                tienda_info = df_tienda.iloc[0] # Obtener información de la primera fila para tipo y cluster
                
                type_sales_avg = df_full[
                    (df_full['store_type'] == tienda_info['store_type'])
                ]['sales'].mean()
                
                cluster_sales_avg = df_full[
                    (df_full['cluster'] == tienda_info['cluster'])
                ]['sales'].mean()

                comparison_data = pd.DataFrame({
                    'Categoría': [f'Tienda {tienda_individual_seleccionada}', f'Promedio Tipo {tienda_info["store_type"]}', f'Promedio Clúster {tienda_info["cluster"]}'],
                    'Ventas Promedio': [tienda_sales_avg, type_sales_avg, cluster_sales_avg]
                })

                fig_tienda_comparativa = px.bar(
                    comparison_data,
                    x='Categoría',
                    y='Ventas Promedio',
                    title=f"Comparación de Ventas Promedio: Tienda {tienda_individual_seleccionada}",
                    labels={'Ventas Promedio': 'Ventas Promedio', 'Categoría': 'Categoría de Comparación'},
                    color='Categoría',
                    color_discrete_map={
                        f'Tienda {tienda_individual_seleccionada}': colors['primary_accent'],
                        f'Promedio Tipo {tienda_info["store_type"]}': colors['level4'],
                        f'Promedio Clúster {tienda_info["cluster"]}': colors['level5']
                    }
                )
                fig_tienda_comparativa.update_layout(
                    template="plotly_dark",
                    paper_bgcolor=colors['background_divs'],
                    plot_bgcolor=colors['background_divs']
                )


        table_data = dff.to_dict('records')
        table_columns = [{"name": i, "id": i} for i in dff.columns]


        return (updated_summary_content, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig_map,
                fig_prediccion_ventas, fig_tienda_tendencia, fig_tienda_familias, fig_tienda_comparativa,
                table_data, table_columns)

    @app.callback(
        Output("download-excel", "data"),
        Input("boton-descargar-excel", "n_clicks"),
        State('filtro-ano', 'value'),
        State('filtro-tipo-tienda', 'value'),
        prevent_initial_call=True
    )
    def download_excel(n_clicks, ano_seleccionado, tipo_tienda_seleccionado):
        if n_clicks > 0:
            if not isinstance(tipo_tienda_seleccionado, list):
                tipo_tienda_seleccionado = [tipo_tienda_seleccionado]

            df_filtered_for_download = df_full[
                (df_full['year'] == ano_seleccionado) &
                (df_full['store_type'].isin(tipo_tienda_seleccionado))
            ].copy()

            if df_filtered_for_download.empty:
                return None

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_filtered_for_download.to_excel(writer, index=False, sheet_name='Datos de Ventas')
            excel_buffer.seek(0)

            return dcc.send_bytes(excel_buffer.read, filename=f"ventas_filtradas_{ano_seleccionado}.xlsx")
        return None

    if __name__ == '__main__':
        app.run_server(debug=True, port=8155)
else:
    print("La aplicación Dash no pudo iniciarse debido a errores en la carga de datos. Por favor, verifica los archivos CSV y los mensajes de error anteriores.")
