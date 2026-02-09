"""
MICMAC PRO - Análisis Estructural con Conversor Integrado
Matriz de Impactos Cruzados - Multiplicación Aplicada a una Clasificación

Autor: JETLEX Strategic Consulting / Martín PRATTO CHIARELLA
Basado en el método de Michel Godet (1990)
Versión: 5.1 - Descarga gráficos + Informe + Nuevos análisis estratégicos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import re
import base64
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="MICMAC PRO - Análisis Estructural",
    page_icon="🎯",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .download-btn {
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES DE DESCARGA DE GRÁFICOS
# ============================================================

def get_plotly_download_buttons(fig, nombre_base, key_suffix=""):
    """Genera botones de descarga para un gráfico Plotly"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Descargar como HTML interactivo
        html_buffer = BytesIO()
        html_content = fig.to_html(include_plotlyjs=True, full_html=True)
        html_buffer.write(html_content.encode())
        html_buffer.seek(0)
        
        st.download_button(
            label="📥 Descargar HTML interactivo",
            data=html_buffer,
            file_name=f"{nombre_base}.html",
            mime="text/html",
            key=f"html_{nombre_base}_{key_suffix}"
        )
    
    with col2:
        # Descargar como PNG
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            st.download_button(
                label="📥 Descargar PNG",
                data=img_bytes,
                file_name=f"{nombre_base}.png",
                mime="image/png",
                key=f"png_{nombre_base}_{key_suffix}"
            )
        except Exception as e:
            st.info("💡 Para descargar PNG, instala: `pip install kaleido`")

# ============================================================
# FUNCIONES DE EXTRACCIÓN DE CÓDIGOS
# ============================================================

def extraer_codigo(nombre_variable):
    """Extrae el código corto de una variable"""
    if pd.isna(nombre_variable):
        return "VAR"
    
    nombre = str(nombre_variable).strip()
    
    if re.match(r'^[A-Za-z_]+[A-Za-z0-9_]*$', nombre) and len(nombre) < 15:
        return nombre.upper()
    
    match = re.match(r'^([A-Za-z]+\d+)', nombre)
    if match:
        return match.group(1).upper()
    
    match = re.match(r'^(V\d+)', nombre, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return nombre[:8].upper() if len(nombre) >= 8 else nombre.upper()

def generar_codigos_y_mapeo(nombres_variables):
    """Genera códigos únicos para cada variable"""
    codigos = []
    mapeo = {}
    codigos_usados = set()
    
    for i, nombre in enumerate(nombres_variables):
        codigo = extraer_codigo(nombre)
        
        codigo_original = codigo
        contador = 1
        while codigo in codigos_usados:
            codigo = f"{codigo_original}_{contador}"
            contador += 1
        
        codigos_usados.add(codigo)
        codigos.append(codigo)
        mapeo[codigo] = nombre
    
    return codigos, mapeo

# ============================================================
# FUNCIONES DE CÁLCULO MICMAC
# ============================================================

def calcular_midi(M, alpha=0.5, K=3):
    """Calcula la Matriz de Influencias Directas e Indirectas (MIDI)"""
    M = np.array(M, dtype=float)
    n = M.shape[0]
    
    if n == 0:
        return M
    
    np.fill_diagonal(M, 0)
    
    MIDI = M.copy()
    M_power = M.copy()
    
    for k in range(2, K + 1):
        M_power = np.dot(M_power, M)
        MIDI += (alpha ** (k - 1)) * M_power
    
    return MIDI

def calcular_motricidad_dependencia(MIDI):
    """Calcula motricidad y dependencia de cada variable"""
    motricidad = MIDI.sum(axis=1)
    dependencia = MIDI.sum(axis=0)
    return motricidad, dependencia

def clasificar_variables(motricidad, dependencia):
    """Clasifica variables en 4 cuadrantes según metodología Godet"""
    med_mot = np.median(motricidad)
    med_dep = np.median(dependencia)
    
    clasificacion = []
    for mot, dep in zip(motricidad, dependencia):
        if mot >= med_mot and dep < med_dep:
            clasificacion.append("Determinantes")
        elif mot >= med_mot and dep >= med_dep:
            clasificacion.append("Clave")
        elif mot < med_dep and dep >= med_dep:
            clasificacion.append("Variables resultado")
        else:
            clasificacion.append("Autónomas")
    
    return clasificacion, med_mot, med_dep

def detectar_convergencia(M, K_max=10, tolerancia=0.01):
    """Detecta el K óptimo donde el ranking se estabiliza"""
    M = np.array(M, dtype=float)
    
    if M.shape[0] == 0:
        return 3
    
    ranking_anterior = None
    
    for K in range(2, K_max + 1):
        try:
            MIDI = calcular_midi(M, alpha=0.5, K=K)
            motricidad, _ = calcular_motricidad_dependencia(MIDI)
            ranking_actual = np.argsort(motricidad)[::-1]
            
            if ranking_anterior is not None and len(ranking_anterior) == len(ranking_actual):
                correlacion = np.corrcoef(ranking_anterior, ranking_actual)[0, 1]
                if not np.isnan(correlacion) and correlacion > (1 - tolerancia):
                    return K
            
            ranking_anterior = ranking_actual
        except:
            return 3
    
    return K_max

# ============================================================
# FUNCIONES DE ANÁLISIS ADICIONALES
# ============================================================

def calcular_estabilidad_ranking(M, alpha=0.5, K_max=5):
    """Calcula cómo cambia el ranking con diferentes valores de K"""
    rankings = {}
    for K in range(1, K_max + 1):
        if K == 1:
            MIDI = M.copy()
        else:
            MIDI = calcular_midi(M, alpha=alpha, K=K)
        mot, _ = calcular_motricidad_dependencia(MIDI)
        rankings[f"K={K}"] = np.argsort(mot)[::-1]
    return rankings

def identificar_relaciones_fuertes(M, nombres, umbral=2):
    """Identifica las relaciones más fuertes de la matriz"""
    relaciones = []
    n = len(nombres)
    for i in range(n):
        for j in range(n):
            if i != j and M[i, j] >= umbral:
                relaciones.append({
                    'Origen': nombres[i],
                    'Destino': nombres[j],
                    'Intensidad': M[i, j]
                })
    return sorted(relaciones, key=lambda x: x['Intensidad'], reverse=True)

def calcular_indice_inestabilidad(motricidad, dependencia):
    """Calcula índice de inestabilidad para cada variable"""
    # Variables con alta motricidad Y alta dependencia son más inestables
    return motricidad * dependencia

# ============================================================
# PROCESADOR ROBUSTO DE ARCHIVOS EXCEL
# ============================================================

def es_celda_excluir(valor):
    """Detecta si una celda contiene valores a excluir"""
    if pd.isna(valor):
        return False
    valor_str = str(valor).upper().strip()
    palabras_excluir = ['SUMA', 'TOTAL', 'SUM', 'PROMEDIO', 'AVERAGE', 'MEAN', 'Σ', 'MOTRI', 'DEPEN']
    return any(palabra in valor_str for palabra in palabras_excluir)

def es_valor_diagonal(valor):
    """Detecta si un valor es marcador de diagonal"""
    if pd.isna(valor):
        return False
    valor_str = str(valor).upper().strip()
    return valor_str in ['X', '-', 'N/A', 'NA', 'DIAG', '*']

def convertir_valor_numerico(valor):
    """Convierte un valor a numérico"""
    if pd.isna(valor):
        return 0.0
    if es_valor_diagonal(valor):
        return 0.0
    try:
        return float(valor)
    except (ValueError, TypeError):
        return 0.0

def detectar_inicio_matriz(df):
    """Detecta automáticamente dónde empieza la matriz de datos"""
    for i in range(min(20, len(df))):
        fila = df.iloc[i, 1:min(10, len(df.columns))]
        
        valores_validos = 0
        for val in fila:
            if pd.isna(val):
                continue
            if es_valor_diagonal(val):
                valores_validos += 1
            else:
                try:
                    float(val)
                    valores_validos += 1
                except:
                    pass
        
        if valores_validos >= len(fila) * 0.5:
            return i
    
    return 0

def procesar_archivo_excel(uploaded_file, nombre_hoja=None):
    """Procesa archivo Excel de forma robusta"""
    try:
        xl = pd.ExcelFile(uploaded_file)
        hojas_disponibles = xl.sheet_names
        
        if nombre_hoja and nombre_hoja in hojas_disponibles:
            hoja_usar = nombre_hoja
        else:
            hoja_usar = hojas_disponibles[0]
            for hoja in hojas_disponibles:
                if 'MID' in hoja.upper() or 'MATRIZ' in hoja.upper():
                    hoja_usar = hoja
                    break
        
        df = pd.read_excel(xl, sheet_name=hoja_usar, header=None)
        
        if df.empty:
            return None, None, None, "El archivo está vacío"
        
        fila_inicio = detectar_inicio_matriz(df)
        
        if fila_inicio > 0:
            headers = df.iloc[fila_inicio - 1, :].tolist()
        else:
            headers = df.iloc[0, :].tolist()
            fila_inicio = 1
        
        columnas_validas = []
        for j in range(1, len(df.columns)):
            header = headers[j] if j < len(headers) else None
            if not es_celda_excluir(header):
                columnas_validas.append(j)
        
        filas_validas = []
        nombres_variables = []
        
        for i in range(fila_inicio, len(df)):
            nombre_fila = df.iloc[i, 0]
            
            if es_celda_excluir(nombre_fila):
                continue
            
            fila_datos = df.iloc[i, columnas_validas[:5]]
            valores_numericos = 0
            for val in fila_datos:
                if es_valor_diagonal(val) or (not pd.isna(val) and isinstance(val, (int, float))):
                    valores_numericos += 1
                else:
                    try:
                        float(val)
                        valores_numericos += 1
                    except:
                        pass
            
            if valores_numericos >= 2:
                filas_validas.append(i)
                nombres_variables.append(str(nombre_fila).strip() if pd.notna(nombre_fila) else f"V{len(nombres_variables)+1}")
        
        if len(filas_validas) == 0:
            return None, None, None, "No se encontraron filas con datos válidos"
        
        n = min(len(filas_validas), len(columnas_validas))
        filas_validas = filas_validas[:n]
        columnas_validas = columnas_validas[:n]
        nombres_variables = nombres_variables[:n]
        
        matriz = np.zeros((n, n))
        for i, fila_idx in enumerate(filas_validas):
            for j, col_idx in enumerate(columnas_validas):
                valor = df.iloc[fila_idx, col_idx]
                matriz[i, j] = convertir_valor_numerico(valor)
        
        np.fill_diagonal(matriz, 0)
        
        df_matriz = pd.DataFrame(matriz, index=nombres_variables, columns=nombres_variables)
        
        return df_matriz, nombres_variables, hojas_disponibles, f"✅ Matriz {n}x{n} de hoja '{hoja_usar}' procesada correctamente"
            
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

# ============================================================
# GENERACIÓN DE GRÁFICOS ADICIONALES
# ============================================================

def crear_grafico_red_influencias(M, nombres, codigos, umbral=2, usar_codigos=True):
    """Crea un gráfico de red de influencias fuertes"""
    import networkx as nx
    
    etiquetas = codigos if usar_codigos else nombres
    
    # Crear grafo dirigido
    G = nx.DiGraph()
    
    # Agregar nodos
    for i, nombre in enumerate(etiquetas):
        G.add_node(nombre)
    
    # Agregar aristas (solo relaciones >= umbral)
    for i in range(len(M)):
        for j in range(len(M)):
            if i != j and M[i, j] >= umbral:
                G.add_edge(etiquetas[i], etiquetas[j], weight=M[i, j])
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Crear traces para Plotly
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Influencias'
    )
    
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    # Calcular tamaño por grado de entrada
    node_sizes = [15 + G.in_degree(node) * 5 for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='top center',
        marker=dict(
            size=node_sizes,
            color='#1f77b4',
            line=dict(width=2, color='white')
        ),
        name='Variables'
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Red de Influencias Fuertes (≥{umbral})',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        ))
    
    return fig

def crear_grafico_estabilidad(M, nombres, codigos, alpha=0.5, usar_codigos=True):
    """Gráfico de estabilidad del ranking por K"""
    etiquetas = codigos if usar_codigos else nombres
    
    # Calcular rankings para diferentes K
    data = []
    for K in range(1, 6):
        if K == 1:
            MIDI = M.copy()
        else:
            MIDI = calcular_midi(M, alpha=alpha, K=K)
        mot, _ = calcular_motricidad_dependencia(MIDI)
        
        for i, (nombre, m) in enumerate(zip(etiquetas, mot)):
            data.append({
                'K': K,
                'Variable': nombre,
                'Motricidad': m,
                'Ranking': len(mot) - np.argsort(np.argsort(mot))[i]
            })
    
    df = pd.DataFrame(data)
    
    # Gráfico de líneas para top 10
    top_vars = df[df['K'] == 5].nlargest(10, 'Motricidad')['Variable'].tolist()
    df_top = df[df['Variable'].isin(top_vars)]
    
    fig = px.line(df_top, x='K', y='Ranking', color='Variable',
        title='Estabilidad del Ranking (Top 10) según Profundidad K',
        labels={'K': 'Profundidad K', 'Ranking': 'Posición en Ranking'})
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=500)
    
    return fig

def crear_grafico_distribucion_cuadrantes(clasificacion):
    """Gráfico de distribución por cuadrantes"""
    conteo = pd.Series(clasificacion).value_counts()
    
    colores = {
        'Determinantes': '#FF4444',
        'Clave': '#1166CC',
        'Variables resultado': '#66BBFF',
        'Autónomas': '#FF9944'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=conteo.index,
        values=conteo.values,
        hole=0.4,
        marker_colors=[colores.get(c, '#999') for c in conteo.index],
        textinfo='label+percent+value'
    )])
    
    fig.update_layout(
        title='Distribución de Variables por Cuadrante',
        height=400
    )
    
    return fig

def crear_grafico_relaciones_fuertes(M, nombres, codigos, top_n=20, usar_codigos=True):
    """Gráfico de barras de las relaciones más fuertes"""
    etiquetas = codigos if usar_codigos else nombres
    
    relaciones = []
    for i in range(len(M)):
        for j in range(len(M)):
            if i != j and M[i, j] > 0:
                relaciones.append({
                    'Par': f"{etiquetas[i]} → {etiquetas[j]}",
                    'Intensidad': M[i, j]
                })
    
    df = pd.DataFrame(relaciones).nlargest(top_n, 'Intensidad')
    
    fig = px.bar(df, x='Intensidad', y='Par', orientation='h',
        title=f'Top {top_n} Relaciones de Influencia Más Fuertes',
        color='Intensidad',
        color_continuous_scale='Reds')
    
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    
    return fig

def crear_grafico_radar_variables_clave(df_resultados, top_n=8):
    """Gráfico radar de las variables más estratégicas"""
    df_top = df_resultados.nlargest(top_n, 'Motricidad')
    
    # Normalizar valores
    max_mot = df_resultados['Motricidad'].max()
    max_dep = df_resultados['Dependencia'].max()
    
    fig = go.Figure()
    
    for _, row in df_top.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Motricidad']/max_mot, row['Dependencia']/max_dep],
            theta=['Motricidad', 'Dependencia'],
            fill='toself',
            name=row['Código']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f'Perfil de las {top_n} Variables Más Motrices',
        height=500
    )
    
    return fig

def crear_grafico_inestabilidad(df_resultados):
    """Gráfico de índice de inestabilidad"""
    df = df_resultados.copy()
    df['Inestabilidad'] = df['Motricidad'] * df['Dependencia']
    df = df.nlargest(15, 'Inestabilidad')
    
    fig = px.bar(df, x='Código', y='Inestabilidad',
        title='Índice de Inestabilidad (Variables que amplifican cambios)',
        color='Clasificación',
        color_discrete_map={
            'Determinantes': '#FF4444',
            'Clave': '#1166CC',
            'Variables resultado': '#66BBFF',
            'Autónomas': '#FF9944'
        })
    
    fig.update_layout(height=400)
    
    return fig

# ============================================================
# GENERACIÓN DE INFORME
# ============================================================

def generar_informe_excel(res, nombres, codigos, M, nombre_proyecto):
    """Genera informe completo en Excel con múltiples hojas"""
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # 1. Resumen Ejecutivo
        resumen = pd.DataFrame({
            'Parámetro': [
                'Fecha de análisis',
                'Nombre del proyecto',
                'Total de variables',
                'Alpha (α)',
                'K (profundidad)',
                'Variables Determinantes',
                'Variables Clave',
                'Variables Resultado',
                'Variables Autónomas',
                'Densidad de la matriz (%)'
            ],
            'Valor': [
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                nombre_proyecto,
                len(nombres),
                res['alpha'],
                res['K'],
                sum(c == 'Determinantes' for c in res['clasificacion']),
                sum(c == 'Clave' for c in res['clasificacion']),
                sum(c == 'Variables resultado' for c in res['clasificacion']),
                sum(c == 'Autónomas' for c in res['clasificacion']),
                round((M != 0).sum() / M.size * 100, 1)
            ]
        })
        resumen.to_excel(writer, sheet_name='Resumen Ejecutivo', index=False)
        
        # 2. Ranking completo
        res['df_resultados'].to_excel(writer, sheet_name='Ranking Variables', index=False)
        
        # 3. Variables por cuadrante
        for cuadrante in ['Determinantes', 'Clave', 'Variables resultado', 'Autónomas']:
            df_cuad = res['df_resultados'][res['df_resultados']['Clasificación'] == cuadrante]
            if len(df_cuad) > 0:
                df_cuad.to_excel(writer, sheet_name=cuadrante[:30], index=False)
        
        # 4. Matriz MIDI
        pd.DataFrame(res['MIDI'], index=codigos, columns=codigos).to_excel(writer, sheet_name='Matriz MIDI')
        
        # 5. Relaciones fuertes
        relaciones = identificar_relaciones_fuertes(M, codigos, umbral=2)
        if relaciones:
            pd.DataFrame(relaciones).to_excel(writer, sheet_name='Relaciones Fuertes', index=False)
        
        # 6. Tabla de códigos
        pd.DataFrame({'Código': codigos, 'Variable': nombres}).to_excel(writer, sheet_name='Diccionario', index=False)
        
        # 7. Matriz original
        pd.DataFrame(M, index=codigos, columns=codigos).to_excel(writer, sheet_name='Matriz MID Original')
        
        # 8. Análisis estratégico
        df_estrategico = res['df_resultados'].copy()
        df_estrategico['Valor_Estratégico'] = df_estrategico['Motricidad'] + df_estrategico['Dependencia']
        df_estrategico['Índice_Inestabilidad'] = df_estrategico['Motricidad'] * df_estrategico['Dependencia']
        df_estrategico['Distancia_Eje'] = abs(df_estrategico['Motricidad'] - df_estrategico['Dependencia'])
        df_estrategico = df_estrategico.sort_values('Valor_Estratégico', ascending=False)
        df_estrategico.to_excel(writer, sheet_name='Análisis Estratégico', index=False)
    
    buffer.seek(0)
    return buffer

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

st.markdown('<div class="main-header">🎯 MICMAC PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis Estructural con Conversor Integrado</div>', unsafe_allow_html=True)

# Inicializar session state
for key in ['matriz_procesada', 'nombres_variables', 'codigos_variables', 'mapeo_codigos', 'resultados', 'hojas_excel', 'M_original']:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("1. Cargar Matriz")
    uploaded_file = st.file_uploader(
        "Subir archivo Excel",
        type=['xlsx', 'xls'],
        help="Soporta archivos con múltiples hojas"
    )
    
    hoja_seleccionada = None
    if uploaded_file is not None:
        try:
            xl_temp = pd.ExcelFile(uploaded_file)
            st.session_state.hojas_excel = xl_temp.sheet_names
            uploaded_file.seek(0)
        except:
            st.session_state.hojas_excel = None
        
        if st.session_state.hojas_excel and len(st.session_state.hojas_excel) > 1:
            default_idx = 0
            for i, hoja in enumerate(st.session_state.hojas_excel):
                if 'MID' in hoja.upper() or 'MATRIZ' in hoja.upper():
                    default_idx = i
                    break
            
            hoja_seleccionada = st.selectbox(
                "📑 Seleccionar hoja:",
                st.session_state.hojas_excel,
                index=default_idx
            )
    
    st.divider()
    
    st.subheader("2. Parámetros MICMAC")
    alpha = st.slider("α (Alpha)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    K_auto = st.checkbox("K automático", value=True)
    if not K_auto:
        K_manual = st.slider("K - Profundidad", min_value=2, max_value=10, value=3)
    else:
        K_manual = 3
    
    st.divider()
    
    st.subheader("3. Visualización")
    mostrar_etiquetas = st.checkbox("Mostrar etiquetas", value=True)
    usar_codigos = st.checkbox("Usar códigos cortos", value=True)
    tamaño_fuente = st.slider("Tamaño fuente", min_value=8, max_value=16, value=10)

# Pestañas principales
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Datos",
    "📊 Análisis MICMAC",
    "📈 Subsistemas",
    "🎯 Eje Estratégico",
    "🔬 Análisis Avanzado",
    "📑 Informe",
    "📥 Exportar"
])

# ============================================================
# TAB 1: DATOS
# ============================================================
with tab1:
    st.header("📋 Carga y Visualización de Datos")
    
    if uploaded_file is not None:
        df_procesado, nombres, hojas, mensaje = procesar_archivo_excel(uploaded_file, hoja_seleccionada)
        
        if df_procesado is not None:
            st.success(mensaje)
            
            codigos, mapeo = generar_codigos_y_mapeo(nombres)
            
            st.session_state.matriz_procesada = df_procesado
            st.session_state.nombres_variables = nombres
            st.session_state.codigos_variables = codigos
            st.session_state.mapeo_codigos = mapeo
            st.session_state.M_original = df_procesado.values.astype(float)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Variables", len(nombres))
            col2.metric("Tamaño", f"{len(nombres)}x{len(nombres)}")
            
            M_temp = df_procesado.values.astype(float)
            densidad = (M_temp != 0).sum() / M_temp.size * 100
            col3.metric("Densidad", f"{densidad:.1f}%")
            
            st.subheader("🏷️ Tabla de Variables")
            st.dataframe(pd.DataFrame({'Código': codigos, 'Variable': nombres}), use_container_width=True, height=300)
            
            st.subheader("📊 Vista previa de la matriz")
            st.dataframe(df_procesado, use_container_width=True, height=400)
        else:
            st.error(f"❌ {mensaje}")
    else:
        st.info("👆 Sube un archivo Excel con tu matriz de influencias")

# ============================================================
# TAB 2: ANÁLISIS MICMAC
# ============================================================
with tab2:
    st.header("📊 Análisis MICMAC")
    
    if st.session_state.matriz_procesada is not None:
        df = st.session_state.matriz_procesada
        nombres = st.session_state.nombres_variables
        codigos = st.session_state.codigos_variables
        
        M = df.values.astype(float)
        n = M.shape[0]
        
        st.info(f"📐 Matriz: {n}x{n} variables")
        
        if K_auto:
            K_usado = detectar_convergencia(M)
            st.success(f"🔍 K óptimo: **{K_usado}**")
        else:
            K_usado = K_manual
        
        MIDI = calcular_midi(M, alpha=alpha, K=K_usado)
        motricidad, dependencia = calcular_motricidad_dependencia(MIDI)
        clasificacion, med_mot, med_dep = clasificar_variables(motricidad, dependencia)
        
        df_resultados = pd.DataFrame({
            'Código': codigos,
            'Variable': nombres,
            'Motricidad': np.round(motricidad, 2),
            'Dependencia': np.round(dependencia, 2),
            'Clasificación': clasificacion
        })
        df_resultados['Ranking'] = df_resultados['Motricidad'].rank(ascending=False).astype(int)
        df_resultados = df_resultados.sort_values('Motricidad', ascending=False)
        
        st.session_state.resultados = {
            'df_resultados': df_resultados,
            'MIDI': MIDI,
            'motricidad': motricidad,
            'dependencia': dependencia,
            'clasificacion': clasificacion,
            'med_mot': med_mot,
            'med_dep': med_dep,
            'alpha': alpha,
            'K': K_usado
        }
        
        # Métricas
        st.subheader("📈 Resumen")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", len(nombres))
        col2.metric("Determinantes", sum(c == 'Determinantes' for c in clasificacion))
        col3.metric("Clave", sum(c == 'Clave' for c in clasificacion))
        col4.metric("Resultado", sum(c == 'Variables resultado' for c in clasificacion))
        
        # Tabla
        st.subheader("🏆 Ranking de Variables")
        
        def color_clasif(val):
            colors = {
                'Determinantes': 'background-color: #ffcccc',
                'Clave': 'background-color: #cce5ff',
                'Variables resultado': 'background-color: #cceeff',
                'Autónomas': 'background-color: #fff3cd'
            }
            return colors.get(val, '')
        
        st.dataframe(
            df_resultados.style.applymap(color_clasif, subset=['Clasificación']),
            use_container_width=True, height=400
        )
        
        # Heatmap MIDI
        st.subheader("🔢 Matriz MIDI")
        etiquetas = codigos if usar_codigos else nombres
        fig_midi = go.Figure(data=go.Heatmap(
            z=MIDI, x=etiquetas, y=etiquetas, colorscale='Blues'
        ))
        fig_midi.update_layout(height=600, title=f"MIDI (α={alpha}, K={K_usado})")
        st.plotly_chart(fig_midi, use_container_width=True)
        get_plotly_download_buttons(fig_midi, "matriz_midi", "tab2")
        
    else:
        st.warning("⚠️ Primero carga una matriz en 'Datos'")

# ============================================================
# TAB 3: SUBSISTEMAS
# ============================================================
with tab3:
    st.header("📈 Gráfico de Subsistemas")
    
    if st.session_state.resultados is not None:
        res = st.session_state.resultados
        df_res = res['df_resultados'].copy()
        
        st.markdown("""
        <div class="info-box">
        🔴 <b>Determinantes:</b> Palancas de acción<br>
        🔵 <b>Clave:</b> Nudo del sistema<br>
        💧 <b>Resultado:</b> Indicadores<br>
        🟠 <b>Autónomas:</b> Excluidas
        </div>
        """, unsafe_allow_html=True)
        
        color_map = {
            'Determinantes': '#FF4444',
            'Clave': '#1166CC',
            'Variables resultado': '#66BBFF',
            'Autónomas': '#FF9944'
        }
        
        fig = go.Figure()
        
        for clasif, color in color_map.items():
            mask = df_res['Clasificación'] == clasif
            df_temp = df_res[mask]
            if len(df_temp) > 0:
                etiquetas_temp = df_temp['Código'].tolist() if usar_codigos else df_temp['Variable'].tolist()
                hover_text = [f"<b>{c}</b><br>{v}" for c, v in zip(df_temp['Código'], df_temp['Variable'])]
                
                fig.add_trace(go.Scatter(
                    x=df_temp['Dependencia'],
                    y=df_temp['Motricidad'],
                    mode='markers+text' if mostrar_etiquetas else 'markers',
                    name=clasif,
                    text=etiquetas_temp if mostrar_etiquetas else None,
                    textposition='top center',
                    textfont=dict(size=tamaño_fuente),
                    marker=dict(size=12, color=color, line=dict(width=1, color='black')),
                    hovertemplate="%{customdata}<br>Mot: %{y:.1f}<br>Dep: %{x:.1f}<extra></extra>",
                    customdata=hover_text
                ))
        
        fig.add_hline(y=res['med_mot'], line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=res['med_dep'], line_dash="dash", line_color="gray", opacity=0.5)
        
        max_mot = max(res['motricidad']) * 1.1
        max_dep = max(res['dependencia']) * 1.1
        
        fig.add_annotation(x=res['med_dep']*0.3, y=max_mot*0.9, text="🔴 DETERMINANTES", showarrow=False, font=dict(color='red', size=14))
        fig.add_annotation(x=max_dep*0.8, y=max_mot*0.9, text="🔵 CLAVE", showarrow=False, font=dict(color='blue', size=14))
        fig.add_annotation(x=max_dep*0.8, y=res['med_mot']*0.3, text="💧 RESULTADO", showarrow=False, font=dict(color='#66BBFF', size=14))
        fig.add_annotation(x=res['med_dep']*0.3, y=res['med_mot']*0.3, text="🟠 AUTÓNOMAS", showarrow=False, font=dict(color='orange', size=14))
        
        fig.update_layout(
            title=f"Subsistemas MICMAC (α={res['alpha']}, K={res['K']})",
            xaxis_title="Dependencia", yaxis_title="Motricidad", height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        get_plotly_download_buttons(fig, "subsistemas", "tab3")
        
        # Distribución pie chart
        st.subheader("📊 Distribución por Cuadrantes")
        fig_pie = crear_grafico_distribucion_cuadrantes(res['clasificacion'])
        st.plotly_chart(fig_pie, use_container_width=True)
        get_plotly_download_buttons(fig_pie, "distribucion_cuadrantes", "tab3_pie")
        
    else:
        st.warning("⚠️ Ejecuta primero el análisis")

# ============================================================
# TAB 4: EJE ESTRATÉGICO
# ============================================================
with tab4:
    st.header("🎯 Eje Estratégico")
    
    if st.session_state.resultados is not None:
        res = st.session_state.resultados
        df_res = res['df_resultados'].copy()
        
        st.markdown("""
        <div class="info-box">
        <b>Eje Estratégico:</b> La diagonal donde Motricidad = Dependencia.<br>
        Variables cerca de esta línea tienen <b>máximo valor estratégico</b>.
        </div>
        """, unsafe_allow_html=True)
        
        df_res['Valor_Estrategico'] = df_res['Motricidad'] + df_res['Dependencia']
        df_res['Distancia_Eje'] = np.abs(df_res['Motricidad'] - df_res['Dependencia'])
        
        etiquetas = df_res['Código'].tolist() if usar_codigos else df_res['Variable'].tolist()
        hover_text = [f"<b>{c}</b><br>{v}" for c, v in zip(df_res['Código'], df_res['Variable'])]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_res['Dependencia'], y=df_res['Motricidad'],
            mode='markers+text' if mostrar_etiquetas else 'markers',
            text=etiquetas if mostrar_etiquetas else None,
            textposition='top center',
            textfont=dict(size=tamaño_fuente),
            marker=dict(size=12, color=df_res['Valor_Estrategico'], colorscale='YlOrRd',
                showscale=True, colorbar=dict(title="Valor<br>Estratégico")),
            hovertemplate="%{customdata}<br>Mot: %{y:.1f}<br>Dep: %{x:.1f}<extra></extra>",
            customdata=hover_text
        ))
        
        max_val = max(max(res['motricidad']), max(res['dependencia'])) * 1.1
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', name='Eje Estratégico',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(title="Eje Estratégico", height=600, xaxis_title="Dependencia", yaxis_title="Motricidad")
        st.plotly_chart(fig, use_container_width=True)
        get_plotly_download_buttons(fig, "eje_estrategico", "tab4")
        
        st.subheader("🏆 Top 10 Variables Estratégicas")
        top10 = df_res.nlargest(10, 'Valor_Estrategico')[
            ['Código', 'Variable', 'Motricidad', 'Dependencia', 'Valor_Estrategico', 'Clasificación']
        ]
        st.dataframe(top10, use_container_width=True)
        
    else:
        st.warning("⚠️ Ejecuta primero el análisis")

# ============================================================
# TAB 5: ANÁLISIS AVANZADO
# ============================================================
with tab5:
    st.header("🔬 Análisis Avanzado de Inteligencia Estratégica")
    
    if st.session_state.resultados is not None and st.session_state.M_original is not None:
        res = st.session_state.resultados
        M = st.session_state.M_original
        nombres = st.session_state.nombres_variables
        codigos = st.session_state.codigos_variables
        
        # 1. Red de Influencias
        st.subheader("🕸️ Red de Influencias Fuertes")
        umbral_red = st.slider("Umbral de intensidad para mostrar conexión", min_value=1, max_value=3, value=2)
        
        try:
            fig_red = crear_grafico_red_influencias(M, nombres, codigos, umbral=umbral_red, usar_codigos=usar_codigos)
            st.plotly_chart(fig_red, use_container_width=True)
            get_plotly_download_buttons(fig_red, "red_influencias", "tab5_red")
        except Exception as e:
            st.warning(f"No se pudo generar el gráfico de red: {e}")
        
        st.divider()
        
        # 2. Estabilidad del Ranking
        st.subheader("📊 Estabilidad del Ranking según Profundidad K")
        fig_estab = crear_grafico_estabilidad(M, nombres, codigos, alpha=res['alpha'], usar_codigos=usar_codigos)
        st.plotly_chart(fig_estab, use_container_width=True)
        get_plotly_download_buttons(fig_estab, "estabilidad_ranking", "tab5_estab")
        
        st.divider()
        
        # 3. Relaciones Fuertes
        st.subheader("💪 Top Relaciones de Influencia")
        fig_rel = crear_grafico_relaciones_fuertes(M, nombres, codigos, top_n=20, usar_codigos=usar_codigos)
        st.plotly_chart(fig_rel, use_container_width=True)
        get_plotly_download_buttons(fig_rel, "relaciones_fuertes", "tab5_rel")
        
        st.divider()
        
        # 4. Índice de Inestabilidad
        st.subheader("⚠️ Índice de Inestabilidad")
        st.markdown("""
        Variables con alto índice de inestabilidad (Motricidad × Dependencia) son 
        **amplificadores de cambios** en el sistema. Requieren monitoreo constante.
        """)
        fig_inest = crear_grafico_inestabilidad(res['df_resultados'])
        st.plotly_chart(fig_inest, use_container_width=True)
        get_plotly_download_buttons(fig_inest, "indice_inestabilidad", "tab5_inest")
        
        st.divider()
        
        # 5. Radar de Variables Clave
        st.subheader("🎯 Perfil Radar de Variables Estratégicas")
        fig_radar = crear_grafico_radar_variables_clave(res['df_resultados'], top_n=8)
        st.plotly_chart(fig_radar, use_container_width=True)
        get_plotly_download_buttons(fig_radar, "radar_variables", "tab5_radar")
        
    else:
        st.warning("⚠️ Ejecuta primero el análisis")

# ============================================================
# TAB 6: INFORME
# ============================================================
with tab6:
    st.header("📑 Generación de Informe")
    
    if st.session_state.resultados is not None:
        res = st.session_state.resultados
        nombres = st.session_state.nombres_variables
        codigos = st.session_state.codigos_variables
        M = st.session_state.M_original
        
        st.markdown("""
        Genera un informe completo con todos los resultados del análisis MICMAC.
        """)
        
        nombre_proyecto = st.text_input("Nombre del proyecto", value="Analisis_MICMAC")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Informe Excel Completo")
            st.markdown("""
            Incluye:
            - Resumen ejecutivo
            - Ranking de variables
            - Variables por cuadrante
            - Matriz MIDI
            - Relaciones fuertes
            - Análisis estratégico
            """)
            
            if st.button("📥 Generar Informe Excel", type="primary"):
                buffer = generar_informe_excel(res, nombres, codigos, M, nombre_proyecto)
                st.download_button(
                    label="📥 Descargar Informe Excel",
                    data=buffer,
                    file_name=f"{nombre_proyecto}_informe_completo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("✅ Informe generado!")
        
        with col2:
            st.subheader("📋 Resumen del Análisis")
            st.markdown(f"""
            **Proyecto:** {nombre_proyecto}
            
            **Parámetros:**
            - Variables: {len(nombres)}
            - Alpha (α): {res['alpha']}
            - K: {res['K']}
            
            **Distribución:**
            - Determinantes: {sum(c == 'Determinantes' for c in res['clasificacion'])}
            - Clave: {sum(c == 'Clave' for c in res['clasificacion'])}
            - Resultado: {sum(c == 'Variables resultado' for c in res['clasificacion'])}
            - Autónomas: {sum(c == 'Autónomas' for c in res['clasificacion'])}
            """)
        
    else:
        st.warning("⚠️ Ejecuta primero el análisis")

# ============================================================
# TAB 7: EXPORTAR
# ============================================================
with tab7:
    st.header("📥 Exportar Resultados")
    
    if st.session_state.resultados is not None:
        res = st.session_state.resultados
        codigos = st.session_state.codigos_variables
        nombres = st.session_state.nombres_variables
        
        st.subheader("Exportaciones Individuales")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Ranking CSV
            csv_ranking = res['df_resultados'].to_csv(index=False)
            st.download_button(
                "📥 Ranking (CSV)",
                csv_ranking,
                "ranking_variables.csv",
                "text/csv"
            )
        
        with col2:
            # Matriz MIDI CSV
            df_midi = pd.DataFrame(res['MIDI'], index=codigos, columns=codigos)
            csv_midi = df_midi.to_csv()
            st.download_button(
                "📥 Matriz MIDI (CSV)",
                csv_midi,
                "matriz_midi.csv",
                "text/csv"
            )
        
        with col3:
            # Diccionario
            df_dict = pd.DataFrame({'Código': codigos, 'Variable': nombres})
            csv_dict = df_dict.to_csv(index=False)
            st.download_button(
                "📥 Diccionario (CSV)",
                csv_dict,
                "diccionario_variables.csv",
                "text/csv"
            )
        
    else:
        st.warning("⚠️ Ejecuta primero el análisis")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<b>MICMAC PRO v5.1</b> | Metodología Michel Godet (1990) | JETLEX Strategic Consulting | MARTIN PRATTO CHIARELLA 2025
</div>
""", unsafe_allow_html=True)
