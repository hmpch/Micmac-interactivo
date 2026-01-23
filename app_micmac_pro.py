"""
MICMAC PRO - Análisis Estructural con Conversor Integrado
Matriz de Impactos Cruzados - Multiplicación Aplicada a una Clasificación

Autor: JETLEX Strategic Consulting / Martín Pratto Chiarella
Basado en el método de Michel Godet (1990)
Versión: 4.1 - Corrección terminológica (Variables Clave)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

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
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES DE CÁLCULO MICMAC
# ============================================================

def calcular_midi(M, alpha=0.5, K=3):
    """
    Calcula la Matriz de Influencias Directas e Indirectas (MIDI)
    
    Fórmula: MIDI = M + α·M² + α²·M³ + ... + α^(K-1)·M^K
    
    Parámetros:
    - M: Matriz de influencias directas
    - alpha: Factor de atenuación (0 < α ≤ 1)
    - K: Profundidad máxima de análisis
    
    Retorna:
    - MIDI: Matriz de influencias totales
    """
    n = M.shape[0]
    M = M.astype(float)
    
    # Normalizar diagonal a 0
    np.fill_diagonal(M, 0)
    
    # Inicializar MIDI con influencias directas
    MIDI = M.copy()
    M_power = M.copy()
    
    # Sumar influencias indirectas con atenuación
    for k in range(2, K + 1):
        M_power = np.dot(M_power, M)
        MIDI += (alpha ** (k - 1)) * M_power
    
    return MIDI

def calcular_motricidad_dependencia(MIDI):
    """
    Calcula motricidad y dependencia de cada variable
    
    - Motricidad: suma de influencias que ejerce (filas)
    - Dependencia: suma de influencias que recibe (columnas)
    """
    motricidad = MIDI.sum(axis=1)
    dependencia = MIDI.sum(axis=0)
    return motricidad, dependencia

def clasificar_variables(motricidad, dependencia):
    """
    Clasifica variables en 4 cuadrantes según metodología MICMAC de Godet
    
    Cuadrantes:
    - Determinantes: Alta motricidad, Baja dependencia (PALANCAS)
    - Clave: Alta motricidad, Alta dependencia (NUDO DEL SISTEMA)
    - Variables resultado: Baja motricidad, Alta dependencia (INDICADORES)
    - Autónomas: Baja motricidad, Baja dependencia (EXCLUIDAS)
    """
    # Umbrales basados en medianas
    med_mot = np.median(motricidad)
    med_dep = np.median(dependencia)
    
    clasificacion = []
    for mot, dep in zip(motricidad, dependencia):
        if mot >= med_mot and dep < med_dep:
            clasificacion.append("Determinantes")
        elif mot >= med_mot and dep >= med_dep:
            clasificacion.append("Clave")  # CORREGIDO: antes era "Crítico/inestable"
        elif mot < med_mot and dep >= med_dep:
            clasificacion.append("Variables resultado")
        else:
            clasificacion.append("Autónomas")
    
    return clasificacion, med_mot, med_dep

def detectar_convergencia(M, K_max=10, tolerancia=0.01):
    """
    Detecta el K óptimo donde el ranking de variables se estabiliza
    """
    n = M.shape[0]
    ranking_anterior = None
    
    for K in range(2, K_max + 1):
        MIDI = calcular_midi(M, alpha=0.5, K=K)
        motricidad, _ = calcular_motricidad_dependencia(MIDI)
        ranking_actual = np.argsort(motricidad)[::-1]
        
        if ranking_anterior is not None:
            # Calcular correlación de Spearman
            correlacion = np.corrcoef(ranking_anterior, ranking_actual)[0, 1]
            if correlacion > (1 - tolerancia):
                return K
        
        ranking_anterior = ranking_actual
    
    return K_max

# ============================================================
# FUNCIONES DE CONVERSIÓN DE MATRICES
# ============================================================

def detectar_formato_matriz(df):
    """
    Detecta si la matriz tiene columnas de metadata (Tipo, Nombre, Código)
    """
    columnas = df.columns.tolist()
    primera_col = df.iloc[:, 0].astype(str)
    
    # Buscar patrones de metadata
    tiene_tipos = any(col.lower() in ['tipo', 'type', 'categoria', 'category'] for col in columnas[:3])
    tiene_nombres = any(col.lower() in ['nombre', 'name', 'variable', 'descripcion'] for col in columnas[:3])
    tiene_codigos = any(col.lower() in ['codigo', 'code', 'cod', 'id'] for col in columnas[:3])
    
    # Detectar si primera columna tiene códigos tipo P1, E2, S3, etc.
    patron_codigo = primera_col.str.match(r'^[A-Z]+\d+$', na=False).any()
    
    return {
        'tiene_metadata': tiene_tipos or tiene_nombres or tiene_codigos,
        'tiene_codigos_patron': patron_codigo,
        'n_columnas_metadata': sum([tiene_tipos, tiene_nombres, tiene_codigos])
    }

def convertir_matriz_con_metadata(df):
    """
    Convierte matriz con metadata al formato MICMAC estándar
    """
    # Detectar columnas numéricas (la matriz real)
    columnas_numericas = []
    for col in df.columns:
        try:
            if df[col].dtype in ['int64', 'float64'] or pd.to_numeric(df[col], errors='coerce').notna().sum() > len(df) * 0.5:
                columnas_numericas.append(col)
        except:
            pass
    
    # Detectar columna de códigos
    col_codigo = None
    for col in df.columns[:5]:
        if df[col].astype(str).str.match(r'^[A-Z]+\d+$', na=False).sum() > len(df) * 0.5:
            col_codigo = col
            break
    
    if col_codigo is None:
        # Usar primera columna como nombres
        col_codigo = df.columns[0]
    
    # Extraer códigos/nombres de variables
    nombres_variables = df[col_codigo].astype(str).tolist()
    
    # Extraer matriz numérica
    matriz_datos = df[columnas_numericas].values.astype(float)
    
    # Crear DataFrame limpio
    df_limpio = pd.DataFrame(
        matriz_datos,
        index=nombres_variables,
        columns=nombres_variables[:len(columnas_numericas)]
    )
    
    return df_limpio, nombres_variables

def procesar_archivo_excel(uploaded_file):
    """
    Procesa archivo Excel y detecta automáticamente el formato
    """
    try:
        df = pd.read_excel(uploaded_file, header=0)
        
        # Detectar formato
        formato = detectar_formato_matriz(df)
        
        if formato['tiene_metadata']:
            df_procesado, nombres = convertir_matriz_con_metadata(df)
            return df_procesado, nombres, "Matriz con metadata detectada y convertida"
        else:
            # Asumir formato estándar
            df.set_index(df.columns[0], inplace=True)
            nombres = df.index.tolist()
            return df, nombres, "Matriz en formato estándar"
            
    except Exception as e:
        return None, None, f"Error al procesar archivo: {str(e)}"

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

# Header principal
st.markdown('<div class="main-header">🎯 MICMAC PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis Estructural con Conversor Integrado</div>', unsafe_allow_html=True)

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("1. Cargar Matriz")
    uploaded_file = st.file_uploader(
        "Subir archivo Excel",
        type=['xlsx', 'xls'],
        help="Acepta matrices con o sin metadata (Tipo, Nombre, Código)"
    )
    
    st.divider()
    
    st.subheader("2. Parámetros MICMAC")
    
    alpha = st.slider(
        "α (Alpha) - Atenuación",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Factor de atenuación para influencias indirectas. Recomendado: 0.5"
    )
    
    K_auto = st.checkbox("K automático (detectar convergencia)", value=True)
    
    if not K_auto:
        K_max = st.slider(
            "K - Profundidad de análisis",
            min_value=2,
            max_value=10,
            value=3,
            help="Número de iteraciones para calcular influencias indirectas"
        )
    else:
        K_max = None
    
    st.divider()
    
    st.subheader("3. Visualización")
    mostrar_etiquetas = st.checkbox("Mostrar etiquetas en gráficos", value=True)
    tamaño_fuente = st.slider("Tamaño de fuente", min_value=8, max_value=16, value=10)

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Datos",
    "📊 Análisis MICMAC",
    "📈 Subsistemas",
    "🎯 Eje Estratégico",
    "📥 Exportar"
])

# Variables globales para almacenar resultados
if 'matriz_procesada' not in st.session_state:
    st.session_state.matriz_procesada = None
if 'resultados' not in st.session_state:
    st.session_state.resultados = None

# ============================================================
# TAB 1: DATOS
# ============================================================
with tab1:
    st.header("📋 Carga y Visualización de Datos")
    
    if uploaded_file is not None:
        df_procesado, nombres, mensaje = procesar_archivo_excel(uploaded_file)
        
        if df_procesado is not None:
            st.success(f"✅ {mensaje}")
            
            # Guardar en session state
            st.session_state.matriz_procesada = df_procesado
            st.session_state.nombres_variables = nombres
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Variables", len(nombres))
            col2.metric("Celdas", df_procesado.size)
            col3.metric("Densidad", f"{(df_procesado.values != 0).sum() / df_procesado.size * 100:.1f}%")
            
            st.subheader("Vista previa de la matriz")
            st.dataframe(df_procesado, use_container_width=True, height=400)
            
            # Estadísticas básicas
            st.subheader("📊 Estadísticas de la matriz")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Distribución de valores:**")
                valores = df_procesado.values.flatten()
                valores = valores[~np.isnan(valores)]
                
                fig_hist = px.histogram(
                    x=valores,
                    nbins=20,
                    title="Distribución de influencias",
                    labels={'x': 'Valor de influencia', 'y': 'Frecuencia'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.write("**Resumen estadístico:**")
                stats = {
                    'Mínimo': np.nanmin(valores),
                    'Máximo': np.nanmax(valores),
                    'Media': np.nanmean(valores),
                    'Mediana': np.nanmedian(valores),
                    'Desv. Estándar': np.nanstd(valores)
                }
                st.dataframe(pd.DataFrame([stats]).T, use_container_width=True)
        else:
            st.error(mensaje)
    else:
        st.markdown("""
        <div class="info-box">
        <h3>📁 Formatos Aceptados</h3>
        <p><strong>Formato 1 - Matriz estándar:</strong></p>
        <pre>
| Variable | Var1 | Var2 | Var3 |
|----------|------|------|------|
| Var1     | 0    | 3    | 1    |
| Var2     | 2    | 0    | 2    |
| Var3     | 1    | 1    | 0    |
        </pre>
        <p><strong>Formato 2 - Con metadata:</strong></p>
        <pre>
| Tipo      | Nombre              | Código | P1 | E1 | S1 |
|-----------|---------------------|--------|----|----|----| 
| Políticas | Regulación ambiental| P1     | 0  | 2  | 1  |
| Económicas| Inversión digital   | E1     | 1  | 0  | 2  |
| Sociales  | Aceptación social   | S1     | 2  | 1  | 0  |
        </pre>
        <p>MICMAC PRO detecta automáticamente el formato y extrae la matriz.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2: ANÁLISIS MICMAC
# ============================================================
with tab2:
    st.header("📊 Análisis MICMAC")
    
    if st.session_state.matriz_procesada is not None:
        df = st.session_state.matriz_procesada
        nombres = st.session_state.nombres_variables
        M = df.values.astype(float)
        
        # Normalizar diagonal
        np.fill_diagonal(M, 0)
        
        # Detectar K óptimo si es automático
        if K_auto:
            K_usado = detectar_convergencia(M)
            st.info(f"🔍 K óptimo detectado: **{K_usado}** (convergencia del ranking)")
        else:
            K_usado = K_max
        
        # Calcular MIDI
        MIDI = calcular_midi(M, alpha=alpha, K=K_usado)
        
        # Calcular motricidad y dependencia
        motricidad, dependencia = calcular_motricidad_dependencia(MIDI)
        
        # Clasificar variables
        clasificacion, med_mot, med_dep = clasificar_variables(motricidad, dependencia)
        
        # Crear DataFrame de resultados
        df_resultados = pd.DataFrame({
            'Variable': nombres[:len(motricidad)],
            'Motricidad': motricidad,
            'Dependencia': dependencia,
            'Clasificación': clasificacion
        })
        df_resultados['Ranking_Mot'] = df_resultados['Motricidad'].rank(ascending=False).astype(int)
        df_resultados = df_resultados.sort_values('Motricidad', ascending=False)
        
        # Guardar resultados
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
        
        # Métricas resumen
        st.subheader("📈 Resumen del Análisis")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Variables", len(nombres))
        col2.metric("Determinantes", sum(c == 'Determinantes' for c in clasificacion))
        col3.metric("Variables Clave", sum(c == 'Clave' for c in clasificacion))  # CORREGIDO
        col4.metric("Variables Resultado", sum(c == 'Variables resultado' for c in clasificacion))
        
        # Tabla de resultados
        st.subheader("🏆 Ranking de Variables por Motricidad")
        
        # Aplicar colores según clasificación
        def color_clasificacion(val):
            colors = {
                'Determinantes': 'background-color: #ffcccc',
                'Clave': 'background-color: #cce5ff',  # CORREGIDO
                'Variables resultado': 'background-color: #cceeff',
                'Autónomas': 'background-color: #fff3cd'
            }
            return colors.get(val, '')
        
        st.dataframe(
            df_resultados.style.applymap(color_clasificacion, subset=['Clasificación']),
            use_container_width=True,
            height=400
        )
        
        # Matriz MIDI
        st.subheader("🔢 Matriz MIDI (Influencias Directas e Indirectas)")
        
        df_midi = pd.DataFrame(
            MIDI,
            index=nombres[:len(MIDI)],
            columns=nombres[:len(MIDI)]
        )
        
        fig_midi = go.Figure(data=go.Heatmap(
            z=MIDI,
            x=nombres[:len(MIDI)],
            y=nombres[:len(MIDI)],
            colorscale='Blues',
            showscale=True
        ))
        fig_midi.update_layout(
            title=f"Matriz MIDI (α={alpha}, K={K_usado})",
            xaxis_title="Variables (influenciadas)",
            yaxis_title="Variables (influyentes)",
            height=600
        )
        st.plotly_chart(fig_midi, use_container_width=True)
        
    else:
        st.warning("⚠️ Primero carga una matriz en la pestaña 'Datos'")

# ============================================================
# TAB 3: SUBSISTEMAS (GRÁFICO DE CUADRANTES)
# ============================================================
with tab3:
    st.header("📈 Gráfico de Subsistemas")
    
    if st.session_state.resultados is not None:
        res = st.session_state.resultados
        df_res = res['df_resultados']
        
        st.markdown("""
        <div class="info-box">
        <strong>Interpretación de los cuadrantes:</strong>
        <ul>
            <li><strong>🔴 Determinantes:</strong> Alta motricidad, baja dependencia → PALANCAS DE ACCIÓN</li>
            <li><strong>🔵 Clave:</strong> Alta motricidad, alta dependencia → NUDO DEL SISTEMA (vigilar)</li>
            <li><strong>💧 Variables resultado:</strong> Baja motricidad, alta dependencia → INDICADORES</li>
            <li><strong>🟠 Autónomas:</strong> Baja motricidad, baja dependencia → EXCLUIDAS</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Mapa de colores - CORREGIDO
        color_map = {
            'Determinantes': '#FF4444',
            'Clave': '#1166CC',  # CORREGIDO: antes era 'Crítico/inestable'
            'Variables resultado': '#66BBFF',
            'Autónomas': '#FF9944'
        }
        
        # Crear gráfico de dispersión
        fig_subsistemas = go.Figure()
        
        for clasif, color in color_map.items():
            df_temp = df_res[df_res['Clasificación'] == clasif]
            if len(df_temp) > 0:
                fig_subsistemas.add_trace(go.Scatter(
                    x=df_temp['Dependencia'],
                    y=df_temp['Motricidad'],
                    mode='markers+text' if mostrar_etiquetas else 'markers',
                    name=clasif,
                    text=df_temp['Variable'] if mostrar_etiquetas else None,
                    textposition='top center',
                    textfont=dict(size=tamaño_fuente),
                    marker=dict(
                        size=12,
                        color=color,
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate="<b>%{text}</b><br>Motricidad: %{y:.2f}<br>Dependencia: %{x:.2f}<extra></extra>"
                ))
        
        # Líneas de umbrales (medianas)
        fig_subsistemas.add_hline(
            y=res['med_mot'],
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="Mediana Motricidad"
        )
        fig_subsistemas.add_vline(
            x=res['med_dep'],
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="Mediana Dependencia"
        )
        
        # Etiquetas de cuadrantes - CORREGIDO
        max_mot = max(res['motricidad']) * 1.1
        max_dep = max(res['dependencia']) * 1.1
        
        fig_subsistemas.add_annotation(
            x=res['med_dep'] * 0.3,
            y=max_mot * 0.9,
            text="🔴 DETERMINANTES<br>(Palancas)",
            showarrow=False,
            font=dict(size=12, color='red')
        )
        fig_subsistemas.add_annotation(
            x=max_dep * 0.8,
            y=max_mot * 0.9,
            text="🔵 VARIABLES CLAVE<br>(Nudo del sistema)",  # CORREGIDO
            showarrow=False,
            font=dict(size=12, color='blue')
        )
        fig_subsistemas.add_annotation(
            x=max_dep * 0.8,
            y=res['med_mot'] * 0.3,
            text="💧 RESULTADO<br>(Indicadores)",
            showarrow=False,
            font=dict(size=12, color='#66BBFF')
        )
        fig_subsistemas.add_annotation(
            x=res['med_dep'] * 0.3,
            y=res['med_mot'] * 0.3,
            text="🟠 AUTÓNOMAS<br>(Excluidas)",
            showarrow=False,
            font=dict(size=12, color='orange')
        )
        
        fig_subsistemas.update_layout(
            title=f"Plano de Subsistemas MICMAC (α={res['alpha']}, K={res['K']})",
            xaxis_title="Dependencia (suma de influencias recibidas)",
            yaxis_title="Motricidad (suma de influencias ejercidas)",
            height=700,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig_subsistemas, use_container_width=True)
        
        # Tabla resumen por cuadrante
        st.subheader("📊 Distribución por Cuadrantes")
        
        resumen_cuadrantes = df_res.groupby('Clasificación').agg({
            'Variable': 'count',
            'Motricidad': 'mean',
            'Dependencia': 'mean'
        }).round(2)
        resumen_cuadrantes.columns = ['N° Variables', 'Motricidad Media', 'Dependencia Media']
        resumen_cuadrantes['Porcentaje'] = (resumen_cuadrantes['N° Variables'] / len(df_res) * 100).round(1).astype(str) + '%'
        
        st.dataframe(resumen_cuadrantes, use_container_width=True)
        
    else:
        st.warning("⚠️ Primero ejecuta el análisis en la pestaña 'Análisis MICMAC'")

# ============================================================
# TAB 4: EJE ESTRATÉGICO
# ============================================================
with tab4:
    st.header("🎯 Eje Estratégico")
    
    if st.session_state.resultados is not None:
        res = st.session_state.resultados
        df_res = res['df_resultados']
        
        st.markdown("""
        <div class="info-box">
        <strong>¿Qué es el Eje Estratégico?</strong>
        <p>La diagonal donde Motricidad = Dependencia representa las variables con <strong>máximo valor estratégico</strong>.
        Variables cerca de esta línea participan intensamente en los circuitos de retroalimentación del sistema.</p>
        <p><strong>Distancia al eje:</strong> menor distancia = mayor importancia estratégica.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calcular distancia al eje estratégico
        df_res['Distancia_Eje'] = np.abs(df_res['Motricidad'] - df_res['Dependencia'])
        df_res['Valor_Estrategico'] = df_res['Motricidad'] + df_res['Dependencia']
        
        # Gráfico con eje estratégico
        fig_eje = go.Figure()
        
        # Colorear por distancia al eje (valor estratégico)
        fig_eje.add_trace(go.Scatter(
            x=df_res['Dependencia'],
            y=df_res['Motricidad'],
            mode='markers+text' if mostrar_etiquetas else 'markers',
            text=df_res['Variable'] if mostrar_etiquetas else None,
            textposition='top center',
            textfont=dict(size=tamaño_fuente),
            marker=dict(
                size=12,
                color=df_res['Valor_Estrategico'],
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Valor<br>Estratégico")
            ),
            hovertemplate="<b>%{text}</b><br>Motricidad: %{y:.2f}<br>Dependencia: %{x:.2f}<br>Valor Estratégico: %{marker.color:.2f}<extra></extra>"
        ))
        
        # Línea del eje estratégico (diagonal)
        max_val = max(max(res['motricidad']), max(res['dependencia'])) * 1.1
        fig_eje.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Eje Estratégico',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_eje.update_layout(
            title="Eje Estratégico - Variables por Valor Estratégico",
            xaxis_title="Dependencia",
            yaxis_title="Motricidad",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig_eje, use_container_width=True)
        
        # Top 10 variables más estratégicas
        st.subheader("🏆 Top 10 Variables Más Estratégicas")
        
        top_estrategicas = df_res.nlargest(10, 'Valor_Estrategico')[
            ['Variable', 'Motricidad', 'Dependencia', 'Valor_Estrategico', 'Distancia_Eje', 'Clasificación']
        ]
        top_estrategicas.columns = ['Variable', 'Motricidad', 'Dependencia', 'Valor Estratégico', 'Distancia al Eje', 'Clasificación']
        
        st.dataframe(
            top_estrategicas.style.background_gradient(subset=['Valor Estratégico'], cmap='YlOrRd'),
            use_container_width=True
        )
        
    else:
        st.warning("⚠️ Primero ejecuta el análisis en la pestaña 'Análisis MICMAC'")

# ============================================================
# TAB 5: EXPORTAR
# ============================================================
with tab5:
    st.header("📥 Exportar Resultados")
    
    if st.session_state.resultados is not None:
        res = st.session_state.resultados
        df_res = res['df_resultados']
        
        nombre_proyecto = st.text_input("Nombre del proyecto", value="analisis_micmac")
        
        if st.button("📥 Generar Excel Completo", type="primary"):
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Hoja 1: Resultados principales
                df_res.to_excel(writer, sheet_name='Resultados', index=False)
                
                # Hoja 2: Matriz MIDI
                df_midi = pd.DataFrame(
                    res['MIDI'],
                    index=st.session_state.nombres_variables[:len(res['MIDI'])],
                    columns=st.session_state.nombres_variables[:len(res['MIDI'])]
                )
                df_midi.to_excel(writer, sheet_name='Matriz_MIDI')
                
                # Hoja 3: Parámetros
                params = pd.DataFrame({
                    'Parámetro': ['Alpha (α)', 'K (profundidad)', 'N° Variables', 'Determinantes', 'Clave', 'Resultado', 'Autónomas'],
                    'Valor': [
                        res['alpha'],
                        res['K'],
                        len(df_res),
                        sum(c == 'Determinantes' for c in res['clasificacion']),
                        sum(c == 'Clave' for c in res['clasificacion']),  # CORREGIDO
                        sum(c == 'Variables resultado' for c in res['clasificacion']),
                        sum(c == 'Autónomas' for c in res['clasificacion'])
                    ]
                })
                params.to_excel(writer, sheet_name='Parametros', index=False)
                
                # Hoja 4: Matriz original
                if st.session_state.matriz_procesada is not None:
                    st.session_state.matriz_procesada.to_excel(writer, sheet_name='Matriz_Original')
            
            buffer.seek(0)
            
            st.download_button(
                label="📥 Descargar Excel",
                data=buffer,
                file_name=f"{nombre_proyecto}_micmac_pro.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success("✅ Excel generado correctamente!")
        
        # Vista previa de lo que se exportará
        st.subheader("📋 Vista previa de datos a exportar")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Hojas incluidas:**")
            st.write("1. Resultados (ranking y clasificación)")
            st.write("2. Matriz MIDI")
            st.write("3. Parámetros del análisis")
            st.write("4. Matriz original")
        
        with col2:
            st.write("**Resumen del análisis:**")
            st.write(f"- α = {res['alpha']}")
            st.write(f"- K = {res['K']}")
            st.write(f"- Variables: {len(df_res)}")
            
    else:
        st.warning("⚠️ Primero ejecuta el análisis para poder exportar")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>MICMAC PRO</strong> - Análisis Estructural con Conversor Integrado</p>
    <p>Basado en la metodología de Michel Godet (1990)</p>
    <p>Desarrollado por <strong>JETLEX Strategic Consulting</strong></p>
    <p>Martín Pratto Chiarella - 2025</p>
</div>
""", unsafe_allow_html=True)
