"""
MICMAC PRO - Análisis Estructural con Conversor Integrado
Matriz de Impactos Cruzados - Multiplicación Aplicada a una Clasificación

Autor: JETLEX Strategic Consulting / Martín Pratto Chiarella
Basado en el método de Michel Godet (1990)
Versión: 4.4 - Procesador robusto de matrices Excel
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES DE CÁLCULO MICMAC
# ============================================================

def calcular_midi(M, alpha=0.5, K=3):
    """
    Calcula la Matriz de Influencias Directas e Indirectas (MIDI)
    """
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

def es_columna_excluir(nombre):
    """Detecta si una columna debe ser excluida (SUMA, TOTAL, etc.)"""
    if pd.isna(nombre):
        return True
    nombre_str = str(nombre).upper().strip()
    palabras_excluir = ['SUMA', 'TOTAL', 'SUM', 'PROMEDIO', 'AVERAGE', 'MEAN']
    return any(palabra in nombre_str for palabra in palabras_excluir)

def es_fila_valida(fila_datos):
    """Verifica si una fila contiene datos numéricos válidos"""
    valores_numericos = pd.to_numeric(fila_datos, errors='coerce')
    # Una fila es válida si al menos 30% de sus valores son numéricos
    return valores_numericos.notna().sum() > len(fila_datos) * 0.3

def procesar_archivo_excel(uploaded_file):
    """
    Procesa archivo Excel de forma robusta:
    1. Detecta y extrae nombres de variables
    2. Excluye columnas/filas de totales (SUMA, TOTAL, etc.)
    3. Genera matriz cuadrada limpia
    """
    try:
        # Leer archivo
        df = pd.read_excel(uploaded_file, header=0)
        
        if df.empty:
            return None, None, "El archivo está vacío"
        
        # --- PASO 1: Identificar columna de nombres ---
        col_nombres = df.columns[0]
        
        # --- PASO 2: Identificar columnas de datos (excluir SUMA, TOTAL, etc.) ---
        columnas_datos = []
        for col in df.columns[1:]:
            if not es_columna_excluir(col):
                columnas_datos.append(col)
        
        if len(columnas_datos) == 0:
            return None, None, "No se encontraron columnas de datos válidas"
        
        # --- PASO 3: Identificar filas válidas (con datos numéricos) ---
        filas_validas_idx = []
        nombres_variables = []
        
        for i in range(len(df)):
            nombre_fila = df.iloc[i, 0]
            
            # Excluir filas de totales
            if es_columna_excluir(nombre_fila):
                continue
            
            # Verificar si la fila tiene datos numéricos
            fila_datos = df.loc[df.index[i], columnas_datos]
            if es_fila_valida(fila_datos):
                filas_validas_idx.append(i)
                nombres_variables.append(str(nombre_fila).strip() if pd.notna(nombre_fila) else f"Var_{i}")
        
        if len(filas_validas_idx) == 0:
            return None, None, "No se encontraron filas con datos numéricos válidos"
        
        # --- PASO 4: Extraer submatriz ---
        df_matriz = df.iloc[filas_validas_idx][columnas_datos].copy()
        
        # Convertir todo a numérico
        for col in df_matriz.columns:
            df_matriz[col] = pd.to_numeric(df_matriz[col], errors='coerce').fillna(0)
        
        # --- PASO 5: Hacer matriz cuadrada ---
        n_filas = len(df_matriz)
        n_cols = len(df_matriz.columns)
        n = min(n_filas, n_cols)
        
        # Recortar a matriz cuadrada
        df_matriz = df_matriz.iloc[:n, :n]
        nombres_variables = nombres_variables[:n]
        
        # Asignar nombres
        df_matriz.columns = nombres_variables
        df_matriz.index = nombres_variables
        
        # Diagonal a 0
        M = df_matriz.values.astype(float)
        np.fill_diagonal(M, 0)
        df_matriz = pd.DataFrame(M, index=nombres_variables, columns=nombres_variables)
        
        return df_matriz, nombres_variables, f"✅ Matriz {n}x{n} procesada correctamente"
            
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

st.markdown('<div class="main-header">🎯 MICMAC PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis Estructural con Conversor Integrado</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("1. Cargar Matriz")
    uploaded_file = st.file_uploader(
        "Subir archivo Excel",
        type=['xlsx', 'xls'],
        help="Primera columna = nombres de variables. Se excluyen automáticamente filas/columnas de SUMA/TOTAL."
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
    tamaño_fuente = st.slider("Tamaño fuente", min_value=8, max_value=16, value=10)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Datos",
    "📊 Análisis MICMAC",
    "📈 Subsistemas",
    "🎯 Eje Estratégico",
    "📥 Exportar"
])

# Inicializar session state
if 'matriz_procesada' not in st.session_state:
    st.session_state.matriz_procesada = None
if 'nombres_variables' not in st.session_state:
    st.session_state.nombres_variables = None
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
            st.success(mensaje)
            
            st.session_state.matriz_procesada = df_procesado
            st.session_state.nombres_variables = nombres
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Variables", len(nombres))
            col2.metric("Tamaño", f"{len(nombres)}x{len(nombres)}")
            
            M_temp = df_procesado.values.astype(float)
            densidad = (M_temp != 0).sum() / M_temp.size * 100
            col3.metric("Densidad", f"{densidad:.1f}%")
            
            st.subheader("Vista previa de la matriz")
            st.dataframe(df_procesado, use_container_width=True, height=400)
            
            # Estadísticas
            st.subheader("📊 Estadísticas")
            col1, col2 = st.columns(2)
            
            with col1:
                valores = M_temp.flatten()
                valores_sin_cero = valores[valores != 0]
                
                if len(valores_sin_cero) > 0:
                    fig_hist = px.histogram(x=valores_sin_cero, nbins=20, 
                        title="Distribución de influencias (sin ceros)")
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.write("**Resumen:**")
                st.write(f"- Mínimo: {M_temp.min()}")
                st.write(f"- Máximo: {M_temp.max()}")
                st.write(f"- Media: {M_temp.mean():.2f}")
                st.write(f"- Mediana: {np.median(M_temp):.2f}")
        else:
            st.error(f"❌ {mensaje}")
    else:
        st.info("👆 Sube un archivo Excel con tu matriz de influencias")
        st.markdown("""
        **Formato esperado:**
        - Primera columna: nombres de las variables
        - Primera fila (header): nombres de las variables  
        - Filas/columnas con "SUMA" o "TOTAL" se excluyen automáticamente
        """)

# ============================================================
# TAB 2: ANÁLISIS MICMAC
# ============================================================
with tab2:
    st.header("📊 Análisis MICMAC")
    
    if st.session_state.matriz_procesada is not None:
        df = st.session_state.matriz_procesada
        nombres = st.session_state.nombres_variables
        
        M = df.values.astype(float)
        n = M.shape[0]
        
        st.info(f"📐 Matriz: {n}x{n} variables")
        
        # Detectar K
        if K_auto:
            K_usado = detectar_convergencia(M)
            st.success(f"🔍 K óptimo: **{K_usado}**")
        else:
            K_usado = K_manual
        
        # Calcular
        MIDI = calcular_midi(M, alpha=alpha, K=K_usado)
        motricidad, dependencia = calcular_motricidad_dependencia(MIDI)
        clasificacion, med_mot, med_dep = clasificar_variables(motricidad, dependencia)
        
        # DataFrame de resultados
        df_resultados = pd.DataFrame({
            'Variable': nombres,
            'Motricidad': np.round(motricidad, 2),
            'Dependencia': np.round(dependencia, 2),
            'Clasificación': clasificacion
        })
        df_resultados['Ranking'] = df_resultados['Motricidad'].rank(ascending=False).astype(int)
        df_resultados = df_resultados.sort_values('Motricidad', ascending=False)
        
        # Guardar
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
        fig_midi = go.Figure(data=go.Heatmap(
            z=MIDI, x=nombres, y=nombres, colorscale='Blues'
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
        
        st.markdown("""
        <div class="info-box">
        🔴 <b>Determinantes:</b> Palancas de acción (alta motricidad, baja dependencia)<br>
        🔵 <b>Clave:</b> Nudo del sistema (alta motricidad, alta dependencia)<br>
        💧 <b>Resultado:</b> Indicadores (baja motricidad, alta dependencia)<br>
        🟠 <b>Autónomas:</b> Excluidas (baja motricidad, baja dependencia)
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
            df_temp = df_res[df_res['Clasificación'] == clasif]
            if len(df_temp) > 0:
                fig.add_trace(go.Scatter(
                    x=df_temp['Dependencia'],
                    y=df_temp['Motricidad'],
                    mode='markers+text' if mostrar_etiquetas else 'markers',
                    name=clasif,
                    text=df_temp['Variable'] if mostrar_etiquetas else None,
                    textposition='top center',
                    textfont=dict(size=tamaño_fuente),
                    marker=dict(size=12, color=color, line=dict(width=1, color='black')),
                    hovertemplate="<b>%{text}</b><br>Mot: %{y:.1f}<br>Dep: %{x:.1f}<extra></extra>"
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
        
        st.markdown("""
        <div class="info-box">
        <b>Eje Estratégico:</b> La diagonal donde Motricidad = Dependencia.<br>
        Variables cerca de esta línea tienen <b>máximo valor estratégico</b> porque participan
        intensamente en los circuitos de retroalimentación del sistema.
        </div>
        """, unsafe_allow_html=True)
        
        df_res['Valor_Estrategico'] = df_res['Motricidad'] + df_res['Dependencia']
        df_res['Distancia_Eje'] = np.abs(df_res['Motricidad'] - df_res['Dependencia'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
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
            hovertemplate="<b>%{text}</b><br>Mot: %{y:.1f}<br>Dep: %{x:.1f}<extra></extra>"
        ))
        
        # Diagonal
        max_val = max(max(res['motricidad']), max(res['dependencia'])) * 1.1
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', name='Eje Estratégico',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(title="Eje Estratégico", height=600, xaxis_title="Dependencia", yaxis_title="Motricidad")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🏆 Top 10 Variables Estratégicas")
        top10 = df_res.nlargest(10, 'Valor_Estrategico')[
            ['Variable', 'Motricidad', 'Dependencia', 'Valor_Estrategico', 'Distancia_Eje', 'Clasificación']
        ]
        top10.columns = ['Variable', 'Motricidad', 'Dependencia', 'Valor Estratégico', 'Distancia al Eje', 'Clasificación']
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
        
        nombre = st.text_input("Nombre del proyecto", value="micmac_analisis")
        
        if st.button("📥 Generar Excel", type="primary"):
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                res['df_resultados'].to_excel(writer, sheet_name='Resultados', index=False)
                
                nombres = st.session_state.nombres_variables
                pd.DataFrame(res['MIDI'], index=nombres, columns=nombres).to_excel(writer, sheet_name='MIDI')
                
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
<b>MICMAC PRO v4.4</b> | Metodología Michel Godet (1990) | JETLEX Strategic Consulting | Martin Pratto Chiarella 2026
</div>
""", unsafe_allow_html=True)
