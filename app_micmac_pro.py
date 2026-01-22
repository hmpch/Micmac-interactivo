"""
MICMAC PRO - Analisis Estructural con Conversor Integrado
Aplicacion completa para analisis prospectivo de matrices de influencias

Autor: JETLEX Strategic Consulting / Martin Pratto Chiarella
Basado en la metodologia de Michel Godet (LIPSOR)
Version: 2.0 - Nomenclatura corregida segun metodologia oficial
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Configuracion de la pagina
st.set_page_config(
    page_title="MICMAC PRO - Analisis Estructural",
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
</style>
""", unsafe_allow_html=True)

# FUNCIONES DEL CONVERSOR
def procesar_matriz_con_metadata(df):
    """Procesa matriz Excel con 3 columnas de metadata"""
    try:
        variables = []
        for i in range(1, len(df)):
            codigo = df.iloc[i, 2]
            if pd.isna(codigo) or 'SUMA' in str(codigo) or 'Influencia' in str(codigo):
                continue
            variables.append(codigo)
        
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
        
        df_salida = pd.DataFrame(matriz, index=variables, columns=variables)
        return df_salida, True, f"Matriz procesada: {len(variables)} variables"
    except Exception as e:
        return None, False, f"Error: {str(e)}"

# FUNCIONES MICMAC
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
    """
    Clasifica variables segun Godet/LISA Institute:
    - CLAVE: Alta motricidad + Alta dependencia (superior derecha)
    - DETERMINANTE: Alta motricidad + Baja dependencia (superior izquierda)
    - RESULTADO: Baja motricidad + Alta dependencia (inferior derecha)
    - AUTONOMA: Baja motricidad + Baja dependencia (inferior izquierda)
    """
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

def calcular_valor_estrategico(influencias, dependencias):
    """Valor Estrategico = Motricidad + Dependencia (segun Godet)"""
    return influencias + dependencias

# Header
st.markdown('<div class="main-header">🔄 MICMAC PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analisis Estructural Prospectivo - Metodologia Godet</div>', unsafe_allow_html=True)

# Session state
if 'matriz' not in st.session_state:
    st.session_state.matriz = None
if 'variables' not in st.session_state:
    st.session_state.variables = None
if 'matriz_convertida' not in st.session_state:
    st.session_state.matriz_convertida = None

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📘 Metodologia", "🔄 Conversor", "📥 Cargar Matriz", 
    "📊 Analisis MICMAC", "📈 Estabilidad", "💾 Exportar"
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
    
    st.subheader("📊 Clasificacion de Variables (Subsistemas)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="warning-box">
        <h4>🔴 VARIABLES CLAVE (Superior Derecha)</h4>
        <p><strong>Alta motricidad + Alta dependencia</strong></p>
        <p>Son los "retos del sistema". Muy influyentes pero tambien muy influidas. 
        Son <strong>inestables</strong> y estrategicas.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h4>🟢 VARIABLES DETERMINANTES (Superior Izquierda)</h4>
        <p><strong>Alta motricidad + Baja dependencia</strong></p>
        <p>Son las "palancas del sistema". Determinan el funcionamiento. 
        Actuar sobre ellas genera efectos en cascada.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #9e9e9e; margin: 1rem 0;">
        <h4>⚪ VARIABLES RESULTADO (Inferior Derecha)</h4>
        <p><strong>Baja motricidad + Alta dependencia</strong></p>
        <p>Variables de salida. Indicadores de evolucion. No se actua directamente sobre ellas.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #fff8e1; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; margin: 1rem 0;">
        <h4>🟡 VARIABLES AUTONOMAS (Inferior Izquierda)</h4>
        <p><strong>Baja motricidad + Baja dependencia</strong></p>
        <p>Desconectadas del sistema. Tendencias pasadas o inercias. Menor prioridad.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("⚠️ Diferencia Heatmap vs Plano de Subsistemas")
    st.markdown("""
    <div class="warning-box">
    <p><strong>Heatmap:</strong> Muestra valores <strong>absolutos</strong> (motricidad total).</p>
    <p><strong>Plano:</strong> Clasifica segun posicion <strong>relativa a las medianas</strong>.</p>
    <p>Una variable puede tener alta motricidad absoluta pero estar debajo de la mediana si otras son mas motrices.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("📈 ¿Que mide la Estabilidad?")
    st.markdown("""
    <div class="info-box">
    <p>Segun Godet: <em>"El numero de veces necesario para que el ranking de variables ya no varie 
    aunque multipliquemos mas veces la matriz por si misma."</em></p>
    <ul>
    <li><strong>K≤3:</strong> Sistema con relaciones directas y claras</li>
    <li><strong>K=4-6:</strong> Influencias indirectas significativas</li>
    <li><strong>K>6:</strong> Sistema complejo o posibles inconsistencias</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# TAB 2: CONVERSOR
with tab2:
    st.header("🔄 Conversor de Matriz con Metadata")
    archivo_metadata = st.file_uploader("Excel con metadata (Tipo, Variable, Codigo)", type=['xlsx', 'xls'], key="file_metadata")
    
    if archivo_metadata:
        try:
            df_original = pd.read_excel(archivo_metadata, header=None)
            st.success(f"Archivo cargado: {df_original.shape[0]} filas x {df_original.shape[1]} columnas")
            with st.expander("Vista previa"):
                st.dataframe(df_original.head(10))
            
            if st.button("🔄 Convertir Matriz", type="primary"):
                matriz_convertida, exito, mensaje = procesar_matriz_con_metadata(df_original)
                if exito:
                    st.session_state.matriz_convertida = matriz_convertida
                    st.success(mensaje)
                    st.dataframe(matriz_convertida)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Usar en analisis"):
                            st.session_state.matriz = matriz_convertida
                            st.session_state.variables = list(matriz_convertida.index)
                            st.success("Matriz lista")
                    with col2:
                        buffer = BytesIO()
                        matriz_convertida.to_excel(buffer)
                        buffer.seek(0)
                        st.download_button("📥 Descargar", buffer, "matriz_convertida.xlsx")
                else:
                    st.error(mensaje)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# TAB 3: CARGAR MATRIZ
with tab3:
    st.header("📥 Cargar Matriz Simple")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        archivo_simple = st.file_uploader("Excel/CSV formato simple", type=['xlsx', 'xls', 'csv'], key="file_simple")
    with col2:
        if st.session_state.matriz_convertida is not None:
            if st.button("📋 Usar matriz del conversor"):
                st.session_state.matriz = st.session_state.matriz_convertida
                st.session_state.variables = list(st.session_state.matriz_convertida.index)
                st.success("Matriz cargada")
    
    if archivo_simple:
        try:
            df = pd.read_csv(archivo_simple, index_col=0) if archivo_simple.name.endswith('.csv') else pd.read_excel(archivo_simple, index_col=0)
            if df.shape[0] != df.shape[1]:
                n = min(df.shape[0], df.shape[1])
                df = df.iloc[:n, :n]
            st.session_state.matriz = df
            st.session_state.variables = list(df.index)
            st.success(f"Matriz cargada: {len(df)} variables")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# TAB 4: ANALISIS
with tab4:
    st.header("📊 Analisis MICMAC")
    
    if st.session_state.matriz is not None:
        matriz = st.session_state.matriz
        variables = st.session_state.variables
        
        with st.sidebar:
            st.header("⚙️ Parametros")
            alpha = st.slider("α (atenuacion)", 0.0, 1.0, 0.8, 0.1)
            k_potencia = st.slider("K (potencia)", 1, 10, 3)
            calcular = st.button("🚀 Calcular MICMAC", type="primary")
        
        if calcular or 'midi' in st.session_state:
            if calcular:
                midi = calcular_midi(matriz.values, alpha, k_potencia)
                influencias = midi.sum(axis=1)
                dependencias = midi.sum(axis=0)
                clasificacion = clasificar_variables(influencias, dependencias)
                valor_estrategico = calcular_valor_estrategico(influencias, dependencias)
                
                st.session_state.midi = pd.DataFrame(midi, index=variables, columns=variables)
                st.session_state.influencias = influencias
                st.session_state.dependencias = dependencias
                st.session_state.clasificacion = clasificacion
                st.session_state.valor_estrategico = valor_estrategico
                st.session_state.alpha_usado = alpha
                st.session_state.k_usado = k_potencia
                st.success("Analisis completado")
            
            subtab1, subtab2, subtab3, subtab4 = st.tabs(["📊 MIDI", "🎯 Plano Subsistemas", "📈 Rankings", "🎯 Eje Estrategico"])
            
            with subtab1:
                st.subheader("Matriz MIDI")
                st.dataframe(st.session_state.midi.round(2))
                
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.midi.values, x=variables, y=variables,
                    colorscale='Blues', colorbar=dict(title="Influencia")
                ))
                fig.update_layout(title="Heatmap MIDI (valores absolutos)", height=600)
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
                st.subheader("Rankings")
                df_ranking = pd.DataFrame({
                    'Variable': variables, 'Motricidad': st.session_state.influencias,
                    'Dependencia': st.session_state.dependencias,
                    'Valor_Estrategico': st.session_state.valor_estrategico,
                    'Clasificacion': st.session_state.clasificacion
                }).sort_values('Motricidad', ascending=False)
                st.dataframe(df_ranking.round(2))
            
            with subtab4:
                st.subheader("Eje Estrategico")
                st.markdown("**Valor Estrategico = Motricidad + Dependencia**")
                df_est = pd.DataFrame({
                    'Variable': variables, 'Valor_Estrategico': st.session_state.valor_estrategico,
                    'Clasificacion': st.session_state.clasificacion
                }).sort_values('Valor_Estrategico', ascending=False)
                st.dataframe(df_est.round(2))
    else:
        st.warning("⚠️ Carga una matriz primero")

# TAB 5: ESTABILIDAD
with tab5:
    st.header("📈 Analisis de Estabilidad")
    
    if st.session_state.matriz is not None:
        if st.button("🔍 Analizar Estabilidad", type="primary"):
            k_estable, rankings, cambios = analizar_estabilidad(st.session_state.matriz.values)
            st.success(f"El ranking se estabiliza en K = {k_estable}")
            
            fig = go.Figure(go.Scatter(
                x=list(range(2, len(cambios)+2)), y=cambios,
                mode='lines+markers', marker=dict(size=10)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="green")
            fig.update_layout(title="Cambios en el ranking por iteracion", xaxis_title="K", yaxis_title="Cambios")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Carga una matriz primero")

# TAB 6: EXPORTAR
with tab6:
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
            st.download_button("📥 Descargar Excel", buffer, f"{nombre}_micmac.xlsx")
    else:
        st.warning("⚠️ Completa el analisis primero")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<p><strong>MICMAC PRO v2.0</strong> - Metodologia Godet/LISA Institute</p>
<p><strong>JETLEX Strategic Consulting</strong> - Martin Pratto Chiarella - 2025</p>
</div>
""", unsafe_allow_html=True)
