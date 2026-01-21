"""
MICMAC PRO - Análisis Estructural con Conversor Integrado
Aplicación completa para análisis prospectivo de matrices de influencias
Incluye conversor de matrices con metadata a formato MICMAC

Autor: JETLEX Strategic Consulting / Martín Pratto Chiarella
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

# Configuración de la página
st.set_page_config(
    page_title="MICMAC PRO - Análisis Estructural",
    page_icon="🔄",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
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

# Funciones del conversor
def procesar_matriz_con_metadata(df):
    """Procesa matriz Excel con 3 columnas de metadata"""
    try:
        # Extraer variables desde columna 2 (códigos)
        variables = []
        
        # Primera pasada: contar variables
        for i in range(1, len(df)):
            codigo = df.iloc[i, 2]
            if pd.isna(codigo) or 'SUMA' in str(codigo) or 'Influencia' in str(codigo):
                continue
            variables.append(codigo)
        
        # Segunda pasada: extraer matriz completa
        matriz = []
        for i in range(1, len(df)):
            codigo = df.iloc[i, 2]
            if pd.isna(codigo) or 'SUMA' in str(codigo) or 'Influencia' in str(codigo):
                continue
            
            valores = []
            for j in range(3, 3 + len(variables)):
                if j < len(df.columns):
                    val = df.iloc[i, j]
                    val = 0 if pd.isna(val) else int(val)
                    val = max(0, min(3, val))
                    valores.append(val)
                else:
                    valores.append(0)
            
            matriz.append(valores)
        
        # Crear DataFrame de salida
        df_salida = pd.DataFrame(matriz, index=variables, columns=variables)
        
        return df_salida, True, f"✅ Matriz procesada: {len(variables)} variables"
        
    except Exception as e:
        return None, False, f"❌ Error: {str(e)}"

# Funciones MICMAC
def calcular_midi(mid, alpha=0.8, k=2):
    """Calcula Matriz de Influencias Directas e Indirectas"""
    n = mid.shape[0]
    midi = mid.copy()
    mid_power = mid.copy()
    
    for i in range(2, k + 1):
        mid_power = np.dot(mid_power, mid)
        midi += (alpha ** (i-1)) * mid_power
    
    return midi

def clasificar_variables(influencias, dependencias):
    """Clasifica variables en cuadrantes MICMAC"""
    med_inf = np.median(influencias)
    med_dep = np.median(dependencias)
    
    clasificacion = []
    for i in range(len(influencias)):
        if influencias[i] >= med_inf and dependencias[i] < med_dep:
            clasificacion.append("Determinante")
        elif influencias[i] >= med_inf and dependencias[i] >= med_dep:
            clasificacion.append("Crítico")
        elif influencias[i] < med_inf and dependencias[i] >= med_dep:
            clasificacion.append("Resultado")
        else:
            clasificacion.append("Autónomo")
    
    return clasificacion

# Header
st.markdown('<div class="main-header">🔄 MICMAC PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis Estructural con Conversor Integrado</div>', unsafe_allow_html=True)

# Inicializar session state
if 'matriz' not in st.session_state:
    st.session_state.matriz = None
if 'variables' not in st.session_state:
    st.session_state.variables = None
if 'matriz_convertida' not in st.session_state:
    st.session_state.matriz_convertida = None

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs([
    "🔄 Conversor",
    "📥 Cargar Matriz", 
    "📊 Análisis MICMAC",
    "💾 Exportar"
])

# TAB 1: CONVERSOR
with tab1:
    st.header("🔄 Conversor de Matriz con Metadata")
    
    st.markdown("""
    <div class="info-box">
    <h4>¿Tu matriz tiene metadata (Tipo, Nombre, Código)?</h4>
    
    Si tu archivo Excel tiene esta estructura:
    
    | Tipo       | Variable                    | Código | P1 | P6 | ...
    |------------|------------------------------|--------|----|----|-----|
    | Políticas  | Descripción larga           | P1     | 0  | 2  | ...
    
    Usa este conversor para transformarla al formato MICMAC.
    </div>
    """, unsafe_allow_html=True)
    
    archivo_metadata = st.file_uploader(
        "Selecciona tu archivo Excel con metadata",
        type=['xlsx', 'xls'],
        key="file_metadata"
    )
    
    if archivo_metadata:
        try:
            df_original = pd.read_excel(archivo_metadata, header=None)
            st.success(f"✅ Archivo cargado: {df_original.shape[0]} filas × {df_original.shape[1]} columnas")
            
            with st.expander("👁️ Vista previa"):
                st.dataframe(df_original.head(10), use_container_width=True)
            
            if st.button("🔄 Convertir Matriz"):
                with st.spinner("Procesando..."):
                    matriz_convertida, exito, mensaje = procesar_matriz_con_metadata(df_original)
                    
                    if exito:
                        st.session_state.matriz_convertida = matriz_convertida
                        st.success(mensaje)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Variables", len(matriz_convertida))
                        with col2:
                            st.metric("Dimensión", f"{len(matriz_convertida)}×{len(matriz_convertida)}")
                        with col3:
                            st.metric("Total", int(matriz_convertida.values.sum()))
                        
                        st.dataframe(matriz_convertida, use_container_width=True)
                        
                        if st.button("✅ Usar en análisis MICMAC"):
                            st.session_state.matriz = matriz_convertida
                            st.session_state.variables = list(matriz_convertida.index)
                            st.success("✅ Matriz lista. Ve a 'Análisis MICMAC'")
                        
                        # Descarga
                        buffer = BytesIO()
                        matriz_convertida.to_excel(buffer, index=True, header=True)
                        buffer.seek(0)
                        
                        st.download_button(
                            "📥 Descargar Excel convertido",
                            buffer,
                            f"matriz_convertida_{len(matriz_convertida)}vars.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.error(mensaje)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# TAB 2: CARGAR MATRIZ
with tab2:
    st.header("📥 Cargar Matriz Simple")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        archivo_simple = st.file_uploader(
            "Excel formato simple",
            type=['xlsx', 'xls', 'csv'],
            key="file_simple"
        )
    
    with col2:
        if st.session_state.matriz_convertida is not None:
            if st.button("📋 Usar matriz del conversor"):
                st.session_state.matriz = st.session_state.matriz_convertida
                st.session_state.variables = list(st.session_state.matriz_convertida.index)
                st.success("✅ Matriz cargada")
    
    if archivo_simple:
        try:
            if archivo_simple.name.endswith('.csv'):
                df = pd.read_csv(archivo_simple, index_col=0)
            else:
                df = pd.read_excel(archivo_simple, index_col=0)
            
            # Validar cuadrada
            if df.shape[0] != df.shape[1]:
                n = min(df.shape[0], df.shape[1])
                df = df.iloc[:n, :n]
                st.warning("⚠️ Matriz ajustada para ser cuadrada")
            
            st.session_state.matriz = df
            st.session_state.variables = list(df.index)
            
            st.success(f"✅ Matriz cargada: {len(df)} variables")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Variables", len(df))
            with col2:
                st.metric("Total", int(df.values.sum()))
            with col3:
                st.metric("Promedio", f"{df.values.mean():.2f}")
            
            st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# TAB 3: ANÁLISIS
with tab3:
    st.header("📊 Análisis MICMAC")
    
    if st.session_state.matriz is not None:
        matriz = st.session_state.matriz
        variables = st.session_state.variables
        
        # Parámetros
        with st.sidebar:
            st.header("⚙️ Parámetros")
            alpha = st.slider("α (atenuación)", 0.0, 1.0, 0.8, 0.1)
            k_potencia = st.slider("K (potencia)", 1, 5, 2)
            
            calcular = st.button("🚀 Calcular MICMAC")
        
        if calcular or 'midi' in st.session_state:
            if calcular:
                with st.spinner("Calculando..."):
                    # Calcular MIDI
                    midi = calcular_midi(matriz.values, alpha, k_potencia)
                    
                    # Influencias y dependencias
                    influencias = midi.sum(axis=1)
                    dependencias = midi.sum(axis=0)
                    
                    # Clasificar
                    clasificacion = clasificar_variables(influencias, dependencias)
                    
                    # Guardar
                    st.session_state.midi = pd.DataFrame(midi, index=variables, columns=variables)
                    st.session_state.influencias = influencias
                    st.session_state.dependencias = dependencias
                    st.session_state.clasificacion = clasificacion
                    
                    st.success("✅ Análisis completado")
            
            # Mostrar resultados
            subtab1, subtab2, subtab3 = st.tabs(["📊 MIDI", "🎯 Plano", "📈 Rankings"])
            
            with subtab1:
                st.subheader("Matriz MIDI")
                st.dataframe(st.session_state.midi, use_container_width=True)
                
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.midi.values,
                    x=variables,
                    y=variables,
                    colorscale='Blues'
                ))
                fig.update_layout(title="Heatmap MIDI", height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab2:
                st.subheader("Plano Influencia/Dependencia")
                
                influencias = st.session_state.influencias
                dependencias = st.session_state.dependencias
                clasificacion = st.session_state.clasificacion
                
                df_plot = pd.DataFrame({
                    'Variable': variables,
                    'Influencia': influencias,
                    'Dependencia': dependencias,
                    'Clasificación': clasificacion
                })
                
                color_map = {
                    'Determinante': '#e74c3c',
                    'Crítico': '#3498db',
                    'Resultado': '#95a5a6',
                    'Autónomo': '#f39c12'
                }
                
                fig = go.Figure()
                
                for clasif in color_map.keys():
                    df_temp = df_plot[df_plot['Clasificación'] == clasif]
                    fig.add_trace(go.Scatter(
                        x=df_temp['Dependencia'],
                        y=df_temp['Influencia'],
                        mode='markers+text',
                        name=clasif,
                        text=df_temp['Variable'],
                        textposition='top center',
                        marker=dict(size=12, color=color_map[clasif])
                    ))
                
                med_inf = np.median(influencias)
                med_dep = np.median(dependencias)
                
                fig.add_hline(y=med_inf, line_dash="dash", opacity=0.5)
                fig.add_vline(x=med_dep, line_dash="dash", opacity=0.5)
                
                fig.update_layout(
                    title=f"Plano MICMAC (α={alpha}, K={k_potencia})",
                    xaxis_title="Dependencia",
                    yaxis_title="Influencia",
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab3:
                st.subheader("Rankings")
                
                df_ranking = pd.DataFrame({
                    'Variable': variables,
                    'Influencia': influencias,
                    'Dependencia': dependencias,
                    'Clasificación': clasificacion
                }).sort_values('Influencia', ascending=False)
                
                st.dataframe(df_ranking, use_container_width=True, height=400)
                
                # Top 15
                top15 = df_ranking.head(15)
                
                fig = go.Figure(go.Bar(
                    x=top15['Influencia'],
                    y=top15['Variable'],
                    orientation='h',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Top 15 Variables",
                    xaxis_title="Influencia",
                    yaxis=dict(autorange="reversed"),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Carga una matriz primero")

# TAB 4: EXPORTAR
with tab4:
    st.header("💾 Exportar Resultados")
    
    if 'midi' in st.session_state:
        nombre = st.text_input("Nombre del proyecto", "analisis_micmac")
        
        if st.button("📥 Generar Excel"):
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.matriz.to_excel(writer, sheet_name='Matriz_Original')
                st.session_state.midi.to_excel(writer, sheet_name='MIDI')
                
                df_ranking = pd.DataFrame({
                    'Variable': st.session_state.variables,
                    'Influencia': st.session_state.influencias,
                    'Dependencia': st.session_state.dependencias,
                    'Clasificación': st.session_state.clasificacion
                }).sort_values('Influencia', ascending=False)
                
                df_ranking.to_excel(writer, sheet_name='Rankings', index=False)
            
            buffer.seek(0)
            
            st.download_button(
                "📥 Descargar Excel",
                buffer,
                f"{nombre}_micmac.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("⚠️ Completa el análisis primero")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<p><strong>MICMAC PRO</strong> - JETLEX Strategic Consulting</p>
<p>Martín Pratto Chiarella © 2025</p>
</div>
""", unsafe_allow_html=True)
