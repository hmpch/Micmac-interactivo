"""
MICMAC PRO - Análisis Estructural con Conversor Integrado
Matriz de Impactos Cruzados - Multiplicación Aplicada a una Clasificación

Autor: JETLEX Strategic Consulting / Martín Pratto Chiarella
Basado en el método de Michel Godet (1990)
Versión: 5.0 - Soporte múltiples hojas Excel + detección automática de matriz
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import re

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
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES DE EXTRACCIÓN DE CÓDIGOS
# ============================================================

def extraer_codigo(nombre_variable):
    """Extrae el código corto de una variable"""
    if pd.isna(nombre_variable):
        return "VAR"
    
    nombre = str(nombre_variable).strip()
    
    # Si el nombre ya es un código corto (solo letras y números, <15 chars)
    if re.match(r'^[A-Za-z_]+[A-Za-z0-9_]*$', nombre) and len(nombre) < 15:
        return nombre.upper()
    
    # Buscar patrón de código al inicio: letras seguidas de números
    match = re.match(r'^([A-Za-z]+\d+)', nombre)
    if match:
        return match.group(1).upper()
    
    # Buscar patrón V1, V2, etc.
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
        elif mot < med_mot and dep >= med_dep:
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
# PROCESADOR ROBUSTO DE ARCHIVOS EXCEL
# ============================================================

def es_celda_excluir(valor):
    """Detecta si una celda contiene valores a excluir (SUMA, TOTAL, Σ, etc.)"""
    if pd.isna(valor):
        return False
    valor_str = str(valor).upper().strip()
    palabras_excluir = ['SUMA', 'TOTAL', 'SUM', 'PROMEDIO', 'AVERAGE', 'MEAN', 'Σ', 'MOTRI', 'DEPEN']
    return any(palabra in valor_str for palabra in palabras_excluir)

def es_valor_diagonal(valor):
    """Detecta si un valor es marcador de diagonal (X, -, etc.)"""
    if pd.isna(valor):
        return False
    valor_str = str(valor).upper().strip()
    return valor_str in ['X', '-', 'N/A', 'NA', 'DIAG', '*']

def convertir_valor_numerico(valor):
    """Convierte un valor a numérico, manejando casos especiales"""
    if pd.isna(valor):
        return 0.0
    if es_valor_diagonal(valor):
        return 0.0
    try:
        return float(valor)
    except (ValueError, TypeError):
        return 0.0

def detectar_inicio_matriz(df):
    """
    Detecta automáticamente dónde empieza la matriz de datos.
    Busca la primera fila donde la mayoría de celdas son numéricas o 'X'.
    """
    for i in range(min(20, len(df))):  # Buscar en las primeras 20 filas
        fila = df.iloc[i, 1:min(10, len(df.columns))]  # Revisar primeras columnas (excluyendo col 0)
        
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
        
        # Si más del 50% son valores válidos, esta es probablemente la primera fila de datos
        if valores_validos >= len(fila) * 0.5:
            return i
    
    return 0  # Por defecto, empezar desde el inicio

def procesar_archivo_excel(uploaded_file, nombre_hoja=None):
    """
    Procesa archivo Excel de forma robusta:
    - Soporta múltiples hojas
    - Detecta automáticamente inicio de matriz
    - Maneja 'X' en diagonal
    - Excluye filas/columnas de totales
    """
    try:
        # Obtener lista de hojas
        xl = pd.ExcelFile(uploaded_file)
        hojas_disponibles = xl.sheet_names
        
        # Seleccionar hoja
        if nombre_hoja and nombre_hoja in hojas_disponibles:
            hoja_usar = nombre_hoja
        else:
            # Buscar hoja que contenga "MID" o "Matriz" en el nombre
            hoja_usar = hojas_disponibles[0]
            for hoja in hojas_disponibles:
                if 'MID' in hoja.upper() or 'MATRIZ' in hoja.upper():
                    hoja_usar = hoja
                    break
        
        # Leer hoja sin headers
        df = pd.read_excel(xl, sheet_name=hoja_usar, header=None)
        
        if df.empty:
            return None, None, None, "El archivo está vacío"
        
        # Detectar inicio de matriz
        fila_inicio = detectar_inicio_matriz(df)
        
        # La fila anterior al inicio de datos podría ser los headers
        if fila_inicio > 0:
            headers = df.iloc[fila_inicio - 1, :].tolist()
        else:
            headers = df.iloc[0, :].tolist()
            fila_inicio = 1
        
        # Identificar columnas válidas (excluir Σ, SUMA, etc.)
        columnas_validas = []
        for j in range(1, len(df.columns)):  # Empezar desde columna 1
            header = headers[j] if j < len(headers) else None
            if not es_celda_excluir(header):
                columnas_validas.append(j)
        
        # Identificar filas válidas
        filas_validas = []
        nombres_variables = []
        
        for i in range(fila_inicio, len(df)):
            nombre_fila = df.iloc[i, 0]
            
            # Excluir filas de totales
            if es_celda_excluir(nombre_fila):
                continue
            
            # Verificar que la fila tenga datos numéricos
            fila_datos = df.iloc[i, columnas_validas[:5]]  # Revisar primeras columnas
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
            
            if valores_numericos >= 2:  # Al menos 2 valores numéricos
                filas_validas.append(i)
                nombres_variables.append(str(nombre_fila).strip() if pd.notna(nombre_fila) else f"V{len(nombres_variables)+1}")
        
        if len(filas_validas) == 0:
            return None, None, None, "No se encontraron filas con datos válidos"
        
        # Hacer matriz cuadrada
        n = min(len(filas_validas), len(columnas_validas))
        filas_validas = filas_validas[:n]
        columnas_validas = columnas_validas[:n]
        nombres_variables = nombres_variables[:n]
        
        # Extraer y convertir matriz
        matriz = np.zeros((n, n))
        for i, fila_idx in enumerate(filas_validas):
            for j, col_idx in enumerate(columnas_validas):
                valor = df.iloc[fila_idx, col_idx]
                matriz[i, j] = convertir_valor_numerico(valor)
        
        # Diagonal a 0
        np.fill_diagonal(matriz, 0)
        
        # Crear DataFrame
        df_matriz = pd.DataFrame(matriz, index=nombres_variables, columns=nombres_variables)
        
        return df_matriz, nombres_variables, hojas_disponibles, f"✅ Matriz {n}x{n} de hoja '{hoja_usar}' procesada correctamente"
            
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

st.markdown('<div class="main-header">🎯 MICMAC PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis Estructural con Conversor Integrado</div>', unsafe_allow_html=True)

# Inicializar session state
if 'matriz_procesada' not in st.session_state:
    st.session_state.matriz_procesada = None
if 'nombres_variables' not in st.session_state:
    st.session_state.nombres_variables = None
if 'codigos_variables' not in st.session_state:
    st.session_state.codigos_variables = None
if 'mapeo_codigos' not in st.session_state:
    st.session_state.mapeo_codigos = None
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
if 'hojas_excel' not in st.session_state:
    st.session_state.hojas_excel = None

with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("1. Cargar Matriz")
    uploaded_file = st.file_uploader(
        "Subir archivo Excel",
        type=['xlsx', 'xls'],
        help="Soporta archivos con múltiples hojas. Detecta automáticamente la matriz."
    )
    
    # Selector de hoja (aparece después de cargar archivo)
    hoja_seleccionada = None
    if uploaded_file is not None:
        # Leer hojas disponibles
        try:
            xl_temp = pd.ExcelFile(uploaded_file)
            st.session_state.hojas_excel = xl_temp.sheet_names
            uploaded_file.seek(0)  # Reset file pointer
        except:
            st.session_state.hojas_excel = None
        
        if st.session_state.hojas_excel and len(st.session_state.hojas_excel) > 1:
            # Preseleccionar hoja con "MID" o "Matriz"
            default_idx = 0
            for i, hoja in enumerate(st.session_state.hojas_excel):
                if 'MID' in hoja.upper() or 'MATRIZ' in hoja.upper():
                    default_idx = i
                    break
            
            hoja_seleccionada = st.selectbox(
                "📑 Seleccionar hoja:",
                st.session_state.hojas_excel,
                index=default_idx,
                help="Selecciona la hoja que contiene la matriz MID"
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
    usar_codigos = st.checkbox("Usar códigos cortos", value=True,
        help="Muestra códigos como TENS_GEO en lugar del nombre completo")
    tamaño_fuente = st.slider("Tamaño fuente", min_value=8, max_value=16, value=10)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Datos",
    "📊 Análisis MICMAC",
    "📈 Subsistemas",
    "🎯 Eje Estratégico",
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
            
            # Generar códigos
            codigos, mapeo = generar_codigos_y_mapeo(nombres)
            
            st.session_state.matriz_procesada = df_procesado
            st.session_state.nombres_variables = nombres
            st.session_state.codigos_variables = codigos
            st.session_state.mapeo_codigos = mapeo
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Variables", len(nombres))
            col2.metric("Tamaño", f"{len(nombres)}x{len(nombres)}")
            
            M_temp = df_procesado.values.astype(float)
            densidad = (M_temp != 0).sum() / M_temp.size * 100
            col3.metric("Densidad", f"{densidad:.1f}%")
            
            # Tabla de códigos
            st.subheader("🏷️ Tabla de Variables")
            df_codigos = pd.DataFrame({
                'Código': codigos,
                'Variable': nombres
            })
            st.dataframe(df_codigos, use_container_width=True, height=300)
            
            st.subheader("📊 Vista previa de la matriz")
            st.dataframe(df_procesado, use_container_width=True, height=400)
            
        else:
            st.error(f"❌ {mensaje}")
    else:
        st.info("👆 Sube un archivo Excel con tu matriz de influencias")
        st.markdown("""
        **Formatos soportados:**
        - Archivos con múltiples hojas (selecciona la hoja con la matriz)
        - Matrices con encabezados descriptivos (se detectan automáticamente)
        - Diagonal marcada con 'X' (se convierte a 0)
        - Filas/columnas de totales (se excluyen automáticamente)
        """)

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
            use_container_width=True,
            height=400
        )
        
        # Heatmap
        st.subheader("🔢 Matriz MIDI")
        etiquetas = codigos if usar_codigos else nombres
        fig_midi = go.Figure(data=go.Heatmap(
            z=MIDI, x=etiquetas, y=etiquetas, colorscale='Blues'
        ))
        fig_midi.update_layout(height=600, title=f"MIDI (α={alpha}, K={K_usado})")
        st.plotly_chart(fig_midi, use_container_width=True)
        
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
        codigos = st.session_state.codigos_variables
        nombres = st.session_state.nombres_variables
        
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
        
        # Líneas de umbral
        fig.add_hline(y=res['med_mot'], line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=res['med_dep'], line_dash="dash", line_color="gray", opacity=0.5)
        
        # Etiquetas cuadrantes
        max_mot = max(res['motricidad']) * 1.1
        max_dep = max(res['dependencia']) * 1.1
        
        fig.add_annotation(x=res['med_dep']*0.3, y=max_mot*0.9, text="🔴 DETERMINANTES", showarrow=False, font=dict(color='red', size=14))
        fig.add_annotation(x=max_dep*0.8, y=max_mot*0.9, text="🔵 CLAVE", showarrow=False, font=dict(color='blue', size=14))
        fig.add_annotation(x=max_dep*0.8, y=res['med_mot']*0.3, text="💧 RESULTADO", showarrow=False, font=dict(color='#66BBFF', size=14))
        fig.add_annotation(x=res['med_dep']*0.3, y=res['med_mot']*0.3, text="🟠 AUTÓNOMAS", showarrow=False, font=dict(color='orange', size=14))
        
        fig.update_layout(
            title=f"Subsistemas MICMAC (α={res['alpha']}, K={res['K']})",
            xaxis_title="Dependencia",
            yaxis_title="Motricidad",
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de referencia
        with st.expander("📋 Ver tabla de referencia", expanded=False):
            st.dataframe(pd.DataFrame({'Código': codigos, 'Variable': nombres}), use_container_width=True)
        
        # Resumen por cuadrante
        st.subheader("📊 Distribución por Cuadrantes")
        resumen = df_res.groupby('Clasificación').agg({
            'Variable': 'count',
            'Motricidad': 'mean',
            'Dependencia': 'mean'
        }).round(2)
        resumen.columns = ['Cantidad', 'Motricidad Media', 'Dependencia Media']
        st.dataframe(resumen, use_container_width=True)
        
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
        codigos = st.session_state.codigos_variables
        nombres = st.session_state.nombres_variables
        
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
            x=df_res['Dependencia'],
            y=df_res['Motricidad'],
            mode='markers+text' if mostrar_etiquetas else 'markers',
            text=etiquetas if mostrar_etiquetas else None,
            textposition='top center',
            textfont=dict(size=tamaño_fuente),
            marker=dict(
                size=12,
                color=df_res['Valor_Estrategico'],
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Valor<br>Estratégico")
            ),
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
        
        with st.expander("📋 Ver tabla de referencia", expanded=False):
            st.dataframe(pd.DataFrame({'Código': codigos, 'Variable': nombres}), use_container_width=True)
        
        st.subheader("🏆 Top 10 Variables Estratégicas")
        top10 = df_res.nlargest(10, 'Valor_Estrategico')[
            ['Código', 'Variable', 'Motricidad', 'Dependencia', 'Valor_Estrategico', 'Clasificación']
        ]
        top10.columns = ['Código', 'Variable', 'Motricidad', 'Dependencia', 'Valor Estratégico', 'Clasificación']
        st.dataframe(top10, use_container_width=True)
        
    else:
        st.warning("⚠️ Ejecuta primero el análisis")

# ============================================================
# TAB 5: EXPORTAR
# ============================================================
with tab5:
    st.header("📥 Exportar Resultados")
    
    if st.session_state.resultados is not None:
        res = st.session_state.resultados
        codigos = st.session_state.codigos_variables
        nombres = st.session_state.nombres_variables
        
        nombre = st.text_input("Nombre del proyecto", value="micmac_analisis")
        
        if st.button("📥 Generar Excel", type="primary"):
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                res['df_resultados'].to_excel(writer, sheet_name='Resultados', index=False)
                
                pd.DataFrame(res['MIDI'], index=codigos, columns=codigos).to_excel(writer, sheet_name='MIDI')
                
                pd.DataFrame({'Código': codigos, 'Variable': nombres}).to_excel(writer, sheet_name='Codigos', index=False)
                
                pd.DataFrame({
                    'Parámetro': ['Alpha', 'K', 'Variables', 'Determinantes', 'Clave', 'Resultado', 'Autónomas'],
                    'Valor': [res['alpha'], res['K'], len(res['df_resultados']),
                        sum(c == 'Determinantes' for c in res['clasificacion']),
                        sum(c == 'Clave' for c in res['clasificacion']),
                        sum(c == 'Variables resultado' for c in res['clasificacion']),
                        sum(c == 'Autónomas' for c in res['clasificacion'])]
                }).to_excel(writer, sheet_name='Parametros', index=False)
                
                if st.session_state.matriz_procesada is not None:
                    st.session_state.matriz_procesada.to_excel(writer, sheet_name='Matriz_Original')
            
            buffer.seek(0)
            st.download_button("📥 Descargar Excel", buffer, f"{nombre}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success("✅ Excel generado!")
    else:
        st.warning("⚠️ Ejecuta primero el análisis")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<b>MICMAC PRO v5.0</b> | Metodología Michel Godet (1990) | JETLEX Strategic Consulting | Martin Pratto Chiarella 2025
</div>
""", unsafe_allow_html=True)
