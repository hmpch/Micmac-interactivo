"""
MICMAC PRO - Analisis Estructural con Carga Robusta de Archivos
Aplicacion completa para analisis prospectivo de matrices de influencias

Autor: JETLEX Strategic Consulting / Martin Pratto Chiarella
Basado en la metodologia de Michel Godet (LIPSOR)
Version: 2.1 - Carga robusta de archivos + nomenclatura oficial
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="MICMAC PRO - Analisis Estructural", page_icon="🔄", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #2c3e50; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.2rem; color: #7f8c8d; text-align: center; margin-bottom: 2rem;}
    .info-box {background-color: #e7f3ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin: 1rem 0;}
    .warning-box {background-color: #fff3e0; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ff9800; margin: 1rem 0;}
    .success-box {background-color: #e8f5e9; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #4caf50; margin: 1rem 0;}
    .error-box {background-color: #ffebee; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f44336; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCION DE CARGA ROBUSTA DE ARCHIVOS
# ============================================================================

def cargar_matriz_robusta(archivo, hoja=0):
    """
    Carga una matriz de forma robusta, detectando automaticamente el formato.
    
    Formatos soportados:
    1. Matriz simple (primera columna = nombres, resto = datos)
    2. Matriz con encabezados en fila 1 y columna 1
    3. Matriz con columnas/filas de SUMA al final
    4. Matriz con metadatos adicionales
    
    Returns:
        df_matriz: DataFrame con la matriz cuadrada
        variables: Lista de nombres de variables
        mensaje: Mensaje descriptivo del proceso
    """
    try:
        # Leer el archivo sin procesar
        if archivo.name.endswith('.csv'):
            df_raw = pd.read_csv(archivo, header=None)
        else:
            df_raw = pd.read_excel(archivo, sheet_name=hoja, header=None)
        
        # Resetear posicion del archivo
        archivo.seek(0)
        
        # Detectar formato
        formato_info = detectar_formato_matriz(df_raw)
        
        if formato_info['tipo'] == 'simple_con_indice':
            # Formato: primera columna es indice, primera fila es encabezado
            return procesar_formato_simple(archivo, hoja)
        
        elif formato_info['tipo'] == 'matriz_con_encabezados':
            # Formato: encabezados en fila 0 y columna 0
            return procesar_formato_encabezados(df_raw, formato_info)
        
        elif formato_info['tipo'] == 'matriz_con_metadata':
            # Formato: tiene columnas de metadata antes de los datos
            return procesar_formato_metadata(df_raw, formato_info)
        
        else:
            # Intentar carga basica
            return procesar_formato_basico(archivo, hoja)
            
    except Exception as e:
        return None, None, f"Error al cargar: {str(e)}"


def detectar_formato_matriz(df):
    """
    Detecta el formato de la matriz analizando su estructura.
    """
    info = {
        'tipo': 'desconocido',
        'fila_inicio': 0,
        'col_inicio': 0,
        'tiene_suma': False,
        'n_variables': 0
    }
    
    # Buscar columna/fila SUMA
    for i, col in enumerate(df.iloc[0]):
        if pd.notna(col) and 'SUMA' in str(col).upper():
            info['tiene_suma'] = True
            info['col_suma'] = i
            break
    
    for i, val in enumerate(df.iloc[:, 0] if len(df.columns) > 0 else []):
        if pd.notna(val) and 'SUMA' in str(val).upper():
            info['tiene_suma'] = True
            info['fila_suma'] = i
            break
    
    # Detectar si la celda [0,0] es vacia (indica encabezados en fila/columna)
    if pd.isna(df.iloc[0, 0]) or str(df.iloc[0, 0]).strip() == '':
        # Verificar si la primera fila tiene nombres de variables
        primera_fila = df.iloc[0, 1:].dropna()
        primera_columna = df.iloc[1:, 0].dropna()
        
        if len(primera_fila) > 2 and len(primera_columna) > 2:
            info['tipo'] = 'matriz_con_encabezados'
            info['fila_inicio'] = 1
            info['col_inicio'] = 1
            return info
    
    # Detectar si hay columnas de metadata (N, Nombre Largo, Codigo)
    primera_fila = df.iloc[0].astype(str)
    if any(x.upper() in ['N', 'NOMBRE', 'CODIGO', 'COD', 'ID'] for x in primera_fila[:5] if pd.notna(x)):
        info['tipo'] = 'matriz_con_metadata'
        # Encontrar donde empiezan los datos numericos
        for i in range(min(10, len(df.columns))):
            if df.iloc[1:, i].apply(lambda x: pd.notna(x) and isinstance(x, (int, float))).any():
                # Verificar si las columnas siguientes tambien son numericas
                if i > 0:
                    info['col_inicio'] = i
                    break
        return info
    
    # Formato simple (primera columna = indice)
    info['tipo'] = 'simple_con_indice'
    return info


def procesar_formato_simple(archivo, hoja):
    """Procesa formato simple con primera columna como indice."""
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo, index_col=0)
        else:
            df = pd.read_excel(archivo, sheet_name=hoja, index_col=0)
        
        # Limpiar nombres de columnas e indice
        df.columns = [str(c).strip() for c in df.columns]
        df.index = [str(i).strip() for i in df.index]
        
        # Eliminar columnas/filas de SUMA
        cols_eliminar = [c for c in df.columns if 'SUMA' in c.upper() or 'TOTAL' in c.upper() or 'SUM' in c.upper()]
        filas_eliminar = [i for i in df.index if 'SUMA' in str(i).upper() or 'TOTAL' in str(i).upper() or 'SUM' in str(i).upper()]
        
        df = df.drop(columns=cols_eliminar, errors='ignore')
        df = df.drop(index=filas_eliminar, errors='ignore')
        
        # Asegurar matriz cuadrada
        n = min(df.shape[0], df.shape[1])
        df = df.iloc[:n, :n]
        
        # Convertir a numerico y reemplazar NaN con 0
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        
        # Validar rango 0-3
        df = df.clip(0, 3)
        
        # Diagonal a 0
        np.fill_diagonal(df.values, 0)
        
        variables = list(df.index)
        return df, variables, f"✅ Matriz cargada: {len(variables)} variables"
        
    except Exception as e:
        return None, None, f"Error formato simple: {str(e)}"


def procesar_formato_encabezados(df_raw, info):
    """Procesa formato con encabezados en primera fila y columna."""
    try:
        # Obtener nombres de variables de la primera fila (excluyendo celda 0,0)
        variables = []
        for val in df_raw.iloc[0, 1:]:
            if pd.notna(val):
                nombre = str(val).strip()
                if 'SUMA' not in nombre.upper() and 'TOTAL' not in nombre.upper():
                    variables.append(nombre)
        
        n_vars = len(variables)
        
        # Extraer datos (desde fila 1, columna 1)
        datos = df_raw.iloc[1:n_vars+1, 1:n_vars+1].copy()
        
        # Convertir a numerico
        datos = datos.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        
        # Crear DataFrame
        df = pd.DataFrame(datos.values, index=variables, columns=variables)
        
        # Validar rango 0-3
        df = df.clip(0, 3)
        
        # Diagonal a 0
        np.fill_diagonal(df.values, 0)
        
        return df, variables, f"✅ Matriz cargada: {len(variables)} variables (formato con encabezados)"
        
    except Exception as e:
        return None, None, f"Error formato encabezados: {str(e)}"


def procesar_formato_metadata(df_raw, info):
    """Procesa formato con columnas de metadata (N, Nombre, Codigo)."""
    try:
        col_inicio = info.get('col_inicio', 3)
        
        # Buscar columna con codigos de variables
        col_codigo = None
        for i in range(min(5, len(df_raw.columns))):
            valores = df_raw.iloc[1:, i].dropna().astype(str)
            # Si los valores parecen codigos (cortos, alfanumericos)
            if len(valores) > 0 and valores.str.len().mean() < 50:
                col_codigo = i
        
        if col_codigo is None:
            col_codigo = min(2, len(df_raw.columns) - 1)
        
        # Extraer variables
        variables = []
        for val in df_raw.iloc[1:, col_codigo]:
            if pd.notna(val):
                nombre = str(val).strip()
                if 'SUMA' not in nombre.upper() and 'TOTAL' not in nombre.upper():
                    variables.append(nombre)
        
        n_vars = len(variables)
        
        # Extraer datos
        datos = df_raw.iloc[1:n_vars+1, col_inicio:col_inicio+n_vars].copy()
        datos = datos.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        
        df = pd.DataFrame(datos.values, index=variables, columns=variables)
        df = df.clip(0, 3)
        np.fill_diagonal(df.values, 0)
        
        return df, variables, f"✅ Matriz cargada: {len(variables)} variables (formato con metadata)"
        
    except Exception as e:
        return None, None, f"Error formato metadata: {str(e)}"


def procesar_formato_basico(archivo, hoja):
    """Intento basico de carga."""
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo, sheet_name=hoja)
        
        # Eliminar columnas no numericas excepto la primera
        cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(cols_numericas) > 0:
            # Usar primera columna como indice si no es numerica
            if df.iloc[:, 0].dtype == object:
                df = df.set_index(df.columns[0])
            
            # Filtrar solo columnas numericas
            df = df[cols_numericas]
            
            # Hacer cuadrada
            n = min(df.shape[0], df.shape[1])
            df = df.iloc[:n, :n]
            
            df = df.fillna(0).astype(int).clip(0, 3)
            np.fill_diagonal(df.values, 0)
            
            return df, list(df.index), f"✅ Matriz cargada: {len(df)} variables (formato basico)"
        
        return None, None, "No se encontraron datos numericos"
        
    except Exception as e:
        return None, None, f"Error formato basico: {str(e)}"


# ============================================================================
# FUNCIONES MICMAC
# ============================================================================

def calcular_midi(mid, alpha=0.8, k=2):
    """Calcula MIDI: M + alpha*M^2 + alpha^2*M^3 + ..."""
    midi = mid.copy().astype(float)
    mid_power = mid.copy().astype(float)
    for i in range(2, k + 1):
        mid_power = np.dot(mid_power, mid)
        midi += (alpha ** (i-1)) * mid_power
    return midi

def analizar_estabilidad(mid, max_k=10):
    """Analiza cuando el ranking deja de cambiar"""
    rankings = []
    cambios = []
    midi_prev = mid.copy().astype(float)
    motricidad_prev = midi_prev.sum(axis=1)
    ranking_prev = np.argsort(-motricidad_prev)
    rankings.append(ranking_prev.copy())
    mid_power = mid.copy().astype(float)
    
    for k in range(2, max_k + 1):
        mid_power = np.dot(mid_power, mid)
        midi_actual = midi_prev + mid_power
        motricidad_actual = midi_actual.sum(axis=1)
        ranking_actual = np.argsort(-motricidad_actual)
        rankings.append(ranking_actual.copy())
        n_cambios = np.sum(ranking_actual != ranking_prev)
        cambios.append(n_cambios)
        if n_cambios == 0:
            return k, rankings, cambios
        ranking_prev = ranking_actual.copy()
        midi_prev = midi_actual.copy()
    return max_k, rankings, cambios

def clasificar_variables(influencias, dependencias):
    """Clasificacion segun Godet: Clave, Determinante, Resultado, Autonoma"""
    med_inf = np.median(influencias)
    med_dep = np.median(dependencias)
    clasificacion = []
    for i in range(len(influencias)):
        if influencias[i] >= med_inf and dependencias[i] >= med_dep:
            clasificacion.append("Clave")
        elif influencias[i] >= med_inf and dependencias[i] < med_dep:
            clasificacion.append("Determinante")
        elif influencias[i] < med_inf and dependencias[i] >= med_dep:
            clasificacion.append("Resultado")
        else:
            clasificacion.append("Autonoma")
    return clasificacion

# ============================================================================
# INTERFAZ
# ============================================================================

st.markdown('<div class="main-header">🔄 MICMAC PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analisis Estructural Prospectivo - Metodologia Godet</div>', unsafe_allow_html=True)

# Session state
if 'matriz' not in st.session_state:
    st.session_state.matriz = None
if 'variables' not in st.session_state:
    st.session_state.variables = None

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Metodologia", "📥 Cargar Matriz", "📊 Analisis MICMAC", "📈 Estabilidad", "💾 Exportar"
])

# TAB 1: METODOLOGIA
with tab1:
    st.header("📘 Metodologia MICMAC")
    
    st.markdown("""
    <div class="info-box">
    <h3>¿Que es MICMAC?</h3>
    <p><strong>MICMAC</strong> (Matriz de Impactos Cruzados - Multiplicacion Aplicada para una Clasificacion) 
    es una herramienta desarrollada por <strong>Michel Godet</strong> en el LIPSOR para identificar 
    las variables clave de un sistema.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📊 Clasificacion de Variables")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="warning-box">
        <h4>🔴 VARIABLES CLAVE</h4>
        <p><strong>Alta motricidad + Alta dependencia</strong></p>
        <p>Son los "retos del sistema". Muy influyentes pero tambien muy influidas.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #9e9e9e; margin: 1rem 0;">
        <h4>⚪ VARIABLES RESULTADO</h4>
        <p><strong>Baja motricidad + Alta dependencia</strong></p>
        <p>Variables de salida, resultados del sistema.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="success-box">
        <h4>🟢 VARIABLES DETERMINANTES</h4>
        <p><strong>Alta motricidad + Baja dependencia</strong></p>
        <p>Variables de entrada, muy influyentes. Palancas del sistema.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #fff8e1; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; margin: 1rem 0;">
        <h4>🟡 VARIABLES AUTONOMAS</h4>
        <p><strong>Baja motricidad + Baja dependencia</strong></p>
        <p>Variables poco conectadas al sistema.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("📐 Escala de Influencia Directa (MID)")
    st.markdown("""
    | Valor | Significado |
    |-------|-------------|
    | **0** | Sin influencia directa |
    | **1** | Influencia debil |
    | **2** | Influencia media |
    | **3** | Influencia fuerte |
    """)

# TAB 2: CARGAR MATRIZ
with tab2:
    st.header("📥 Cargar Matriz")
    
    st.markdown("""
    <div class="info-box">
    <h4>📁 Formatos Soportados</h4>
    <p>La aplicacion detecta automaticamente el formato de tu archivo:</p>
    <ul>
    <li><strong>Formato simple:</strong> Primera columna = nombres, resto = datos</li>
    <li><strong>Formato con encabezados:</strong> Primera fila y columna = nombres de variables</li>
    <li><strong>Formato con metadata:</strong> Columnas adicionales (N, Nombre, Codigo) antes de los datos</li>
    <li><strong>Columnas SUMA:</strong> Se detectan y eliminan automaticamente</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        archivo = st.file_uploader("Sube tu archivo Excel o CSV", type=['xlsx', 'xls', 'csv'], key="file_upload")
    
    with col2:
        if archivo and archivo.name.endswith(('.xlsx', '.xls')):
            try:
                xl = pd.ExcelFile(archivo)
                hojas = xl.sheet_names
                archivo.seek(0)
                if len(hojas) > 1:
                    hoja_seleccionada = st.selectbox("Selecciona hoja:", hojas)
                else:
                    hoja_seleccionada = hojas[0]
            except:
                hoja_seleccionada = 0
        else:
            hoja_seleccionada = 0
    
    if archivo:
        if st.button("🔄 Cargar y Procesar", type="primary"):
            with st.spinner("Procesando archivo..."):
                df, variables, mensaje = cargar_matriz_robusta(archivo, hoja_seleccionada)
                
                if df is not None:
                    st.session_state.matriz = df
                    st.session_state.variables = variables
                    st.success(mensaje)
                    
                    st.subheader("📋 Vista Previa de la Matriz")
                    st.dataframe(df, use_container_width=True)
                    
                    # Estadisticas
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Variables", len(variables))
                    col2.metric("Celdas con valor", int((df.values > 0).sum()))
                    col3.metric("Densidad", f"{(df.values > 0).sum() / (len(df)**2 - len(df)) * 100:.1f}%")
                    col4.metric("Valor maximo", int(df.values.max()))
                else:
                    st.error(mensaje)
    
    # Opcion de crear matriz manual
    st.divider()
    st.subheader("✏️ O crear matriz manualmente")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        n_vars = st.number_input("Numero de variables", min_value=3, max_value=50, value=10)
        if st.button("Crear matriz vacia"):
            nombres = [f"V{i+1}" for i in range(n_vars)]
            df_vacia = pd.DataFrame(np.zeros((n_vars, n_vars), dtype=int), index=nombres, columns=nombres)
            st.session_state.matriz = df_vacia
            st.session_state.variables = nombres
            st.success(f"Matriz {n_vars}x{n_vars} creada")
            st.rerun()

# TAB 3: ANALISIS
with tab3:
    st.header("📊 Analisis MICMAC")
    
    if st.session_state.matriz is not None:
        matriz = st.session_state.matriz
        variables = st.session_state.variables
        
        with st.sidebar:
            st.header("⚙️ Parametros")
            alpha = st.slider("α (atenuacion)", 0.0, 1.0, 0.8, 0.1, help="Factor de atenuacion para influencias indirectas")
            k_potencia = st.slider("K (potencia)", 1, 10, 3, help="Numero de iteraciones")
            calcular = st.button("🚀 Calcular MICMAC", type="primary")
        
        if calcular or 'midi' in st.session_state:
            if calcular:
                midi = calcular_midi(matriz.values, alpha, k_potencia)
                influencias = midi.sum(axis=1)
                dependencias = midi.sum(axis=0)
                clasificacion = clasificar_variables(influencias, dependencias)
                valor_estrategico = influencias + dependencias
                
                st.session_state.midi = pd.DataFrame(midi, index=variables, columns=variables)
                st.session_state.influencias = influencias
                st.session_state.dependencias = dependencias
                st.session_state.clasificacion = clasificacion
                st.session_state.valor_estrategico = valor_estrategico
                st.session_state.alpha_usado = alpha
                st.session_state.k_usado = k_potencia
                st.success("✅ Analisis completado")
            
            subtab1, subtab2, subtab3 = st.tabs(["📊 MIDI", "🎯 Plano de Subsistemas", "📈 Rankings"])
            
            with subtab1:
                st.subheader("Matriz MIDI")
                st.dataframe(st.session_state.midi.round(2), use_container_width=True)
                
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.midi.values, x=variables, y=variables,
                    colorscale='Blues', colorbar=dict(title="Influencia")
                ))
                fig.update_layout(title="Heatmap MIDI", height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab2:
                st.subheader("Plano de Subsistemas")
                
                influencias = st.session_state.influencias
                dependencias = st.session_state.dependencias
                clasificacion = st.session_state.clasificacion
                
                color_map = {'Clave': '#e74c3c', 'Determinante': '#27ae60', 'Resultado': '#95a5a6', 'Autonoma': '#f39c12'}
                
                fig = go.Figure()
                for clasif in color_map.keys():
                    mask = [c == clasif for c in clasificacion]
                    if any(mask):
                        fig.add_trace(go.Scatter(
                            x=[d for d, m in zip(dependencias, mask) if m],
                            y=[i for i, m in zip(influencias, mask) if m],
                            mode='markers+text', name=clasif,
                            text=[v for v, m in zip(variables, mask) if m],
                            textposition='top center',
                            marker=dict(size=12, color=color_map[clasif])
                        ))
                
                med_inf, med_dep = np.median(influencias), np.median(dependencias)
                fig.add_hline(y=med_inf, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=med_dep, line_dash="dash", line_color="gray", opacity=0.5)
                
                # Anotaciones de cuadrantes
                fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text="DETERMINANTES", showarrow=False, font=dict(size=11, color="#27ae60"))
                fig.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper", text="CLAVE", showarrow=False, font=dict(size=11, color="#e74c3c"))
                fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text="RESULTADO", showarrow=False, font=dict(size=11, color="#95a5a6"))
                fig.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper", text="AUTONOMAS", showarrow=False, font=dict(size=11, color="#f39c12"))
                
                fig.update_layout(
                    title=f"Plano MICMAC (α={st.session_state.alpha_usado}, K={st.session_state.k_usado})",
                    xaxis_title="Dependencia", yaxis_title="Motricidad", height=700
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🔴 Clave", sum(1 for c in clasificacion if c == 'Clave'))
                col2.metric("🟢 Determinante", sum(1 for c in clasificacion if c == 'Determinante'))
                col3.metric("⚪ Resultado", sum(1 for c in clasificacion if c == 'Resultado'))
                col4.metric("🟡 Autonoma", sum(1 for c in clasificacion if c == 'Autonoma'))
            
            with subtab3:
                st.subheader("Rankings de Variables")
                df_ranking = pd.DataFrame({
                    'Variable': variables, 
                    'Motricidad': st.session_state.influencias,
                    'Dependencia': st.session_state.dependencias,
                    'Valor_Estrategico': st.session_state.valor_estrategico,
                    'Clasificacion': st.session_state.clasificacion
                }).sort_values('Motricidad', ascending=False)
                st.dataframe(df_ranking.round(2), use_container_width=True)
                
                # Top 10 variables clave
                df_clave = df_ranking[df_ranking['Clasificacion'] == 'Clave'].head(10)
                if len(df_clave) > 0:
                    st.subheader("🔴 Top Variables Clave")
                    st.dataframe(df_clave.round(2), use_container_width=True)
    else:
        st.warning("⚠️ Carga una matriz primero en la pestaña 'Cargar Matriz'")

# TAB 4: ESTABILIDAD
with tab4:
    st.header("📈 Analisis de Estabilidad")
    
    if st.session_state.matriz is not None:
        st.markdown("""
        <div class="info-box">
        <p>El analisis de estabilidad determina en que iteracion (K) el ranking de variables deja de cambiar.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Analizar Estabilidad", type="primary"):
            k_estable, rankings, cambios = analizar_estabilidad(st.session_state.matriz.values)
            
            if k_estable < 10:
                st.success(f"✅ El ranking se estabiliza en K = {k_estable}")
            else:
                st.warning(f"⚠️ El ranking no se estabilizo completamente en K = {k_estable}")
            
            fig = go.Figure(go.Scatter(
                x=list(range(2, len(cambios)+2)), y=cambios,
                mode='lines+markers', marker=dict(size=10, color='#3498db'),
                line=dict(width=2)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Estable")
            fig.update_layout(
                title="Cambios en el Ranking por Iteracion",
                xaxis_title="Iteracion (K)", yaxis_title="Numero de Cambios",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Carga una matriz primero")

# TAB 5: EXPORTAR
with tab5:
    st.header("💾 Exportar Resultados")
    
    if 'midi' in st.session_state:
        nombre = st.text_input("Nombre del proyecto", "analisis_micmac")
        
        if st.button("📥 Generar Excel", type="primary"):
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.matriz.to_excel(writer, sheet_name='Matriz_Original')
                st.session_state.midi.round(2).to_excel(writer, sheet_name='MIDI')
                
                df_ranking = pd.DataFrame({
                    'Variable': st.session_state.variables,
                    'Motricidad': st.session_state.influencias,
                    'Dependencia': st.session_state.dependencias,
                    'Valor_Estrategico': st.session_state.valor_estrategico,
                    'Clasificacion': st.session_state.clasificacion
                }).sort_values('Valor_Estrategico', ascending=False)
                df_ranking.round(2).to_excel(writer, sheet_name='Rankings', index=False)
            
            buffer.seek(0)
            st.download_button("📥 Descargar Excel", buffer, f"{nombre}_micmac.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success("✅ Excel generado!")
    else:
        st.warning("⚠️ Completa el analisis primero")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<p><strong>MICMAC PRO v2.1</strong> - Carga Robusta + Metodologia Godet</p>
<p><strong>JETLEX Strategic Consulting</strong> - Martin Pratto Chiarella - 2025</p>
</div>
""", unsafe_allow_html=True)
