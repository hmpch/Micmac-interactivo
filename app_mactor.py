"""
MACTOR - Metodo de Analisis de Actores
Matriz de Alianzas y Conflictos: Tacticas, Objetivos y Recomendaciones

Autor: JETLEX Strategic Consulting / Martin Pratto Chiarella
Basado en el metodo de Michel Godet (LIPSOR)
Version: 2.0 - Corregido segun metodologia oficial (LISA Institute)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="MACTOR - Analisis de Actores", page_icon="🎭", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem;}
    .info-box {background-color: #e7f3ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin: 1rem 0;}
    .warning-box {background-color: #fff3e0; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ff9800; margin: 1rem 0;}
    .success-box {background-color: #e8f5e9; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #4caf50; margin: 1rem 0;}
    .example-box {background-color: #f3e5f5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #9c27b0; margin: 1rem 0;}
    .step-box {background-color: #fff8e1; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; margin: 0.5rem 0;}
    .method-box {background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2196f3; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# FUNCIONES DE CALCULO

def calcular_midi(mid, k=2):
    midi = mid.copy().astype(float)
    mid_power = mid.copy().astype(float)
    for i in range(2, k + 1):
        mid_power = np.dot(mid_power, mid)
        midi += mid_power
    np.fill_diagonal(midi, 0)
    return midi

def calcular_balance(midi):
    influencias = midi.sum(axis=1)
    dependencias = midi.sum(axis=0)
    balance = influencias - dependencias
    return influencias, dependencias, balance

def calcular_coeficiente_ri(influencias, dependencias):
    ri_star = np.zeros_like(influencias)
    for i in range(len(influencias)):
        suma = influencias[i] + dependencias[i]
        if suma > 0:
            ri_star[i] = (influencias[i] - dependencias[i]) / suma
    return ri_star

def calcular_convergencias_simples(mao_1):
    n_actores = mao_1.shape[0]
    convergencias = np.zeros((n_actores, n_actores))
    for i in range(n_actores):
        for j in range(n_actores):
            if i != j:
                conv = np.sum((mao_1[i] == mao_1[j]) & (mao_1[i] != 0))
                convergencias[i, j] = conv
    return convergencias

def calcular_divergencias_simples(mao_1):
    n_actores = mao_1.shape[0]
    divergencias = np.zeros((n_actores, n_actores))
    for i in range(n_actores):
        for j in range(n_actores):
            if i != j:
                div = np.sum((mao_1[i] == -mao_1[j]) & (mao_1[i] != 0) & (mao_1[j] != 0))
                divergencias[i, j] = div
    return divergencias

def calcular_convergencias_ponderadas(mao_1, mao_2):
    n_actores = mao_1.shape[0]
    n_objetivos = mao_1.shape[1]
    convergencias = np.zeros((n_actores, n_actores))
    for i in range(n_actores):
        for j in range(n_actores):
            if i != j:
                for k in range(n_objetivos):
                    if mao_1[i, k] == mao_1[j, k] and mao_1[i, k] != 0:
                        convergencias[i, j] += min(mao_2[i, k], mao_2[j, k])
    return convergencias

def calcular_divergencias_ponderadas(mao_1, mao_2):
    n_actores = mao_1.shape[0]
    n_objetivos = mao_1.shape[1]
    divergencias = np.zeros((n_actores, n_actores))
    for i in range(n_actores):
        for j in range(n_actores):
            if i != j:
                for k in range(n_objetivos):
                    if mao_1[i, k] == -mao_1[j, k] and mao_1[i, k] != 0 and mao_1[j, k] != 0:
                        divergencias[i, j] += min(mao_2[i, k], mao_2[j, k])
    return divergencias

def calcular_3mao(mao_1, mao_2, midi):
    n_actores = mao_1.shape[0]
    n_objetivos = mao_1.shape[1]
    influencias, dependencias, _ = calcular_balance(midi)
    ri_star = calcular_coeficiente_ri(influencias, dependencias)
    ri_norm = (ri_star + 1) / 2
    mao_3 = np.zeros((n_actores, n_objetivos))
    for i in range(n_actores):
        for j in range(n_objetivos):
            mao_3[i, j] = mao_1[i, j] * mao_2[i, j] * (1 + ri_norm[i])
    return mao_3

def calcular_movilizacion_objetivos(mao_1, mao_2):
    n_objetivos = mao_1.shape[1]
    mov_positiva = np.zeros(n_objetivos)
    mov_negativa = np.zeros(n_objetivos)
    for j in range(n_objetivos):
        for i in range(mao_1.shape[0]):
            if mao_1[i, j] == 1:
                mov_positiva[j] += mao_2[i, j]
            elif mao_1[i, j] == -1:
                mov_negativa[j] += mao_2[i, j]
    return mov_positiva, mov_negativa

def clasificar_actores(influencias, dependencias):
    med_inf = np.median(influencias)
    med_dep = np.median(dependencias)
    clasificacion = []
    for i in range(len(influencias)):
        if influencias[i] >= med_inf and dependencias[i] < med_dep:
            clasificacion.append("Dominante")
        elif influencias[i] >= med_inf and dependencias[i] >= med_dep:
            clasificacion.append("Enlace")
        elif influencias[i] < med_inf and dependencias[i] >= med_dep:
            clasificacion.append("Dominado")
        else:
            clasificacion.append("Autonomo")
    return clasificacion

# INICIALIZACION
if 'actores' not in st.session_state:
    st.session_state.actores = []
if 'objetivos' not in st.session_state:
    st.session_state.objetivos = []
if 'mao_1' not in st.session_state:
    st.session_state.mao_1 = None
if 'mao_2' not in st.session_state:
    st.session_state.mao_2 = None
if 'mid' not in st.session_state:
    st.session_state.mid = None

# HEADER
st.markdown('<div class="main-header">🎭 MACTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Método de Análisis de Juego de Actores<br><em>Michel Godet - LIPSOR</em></div>', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("1. Definir Actores")
    num_actores = st.number_input("Número de actores", min_value=2, max_value=20, value=5, step=1)
    
    if st.button("Configurar Actores", key="btn_actores"):
        st.session_state.actores = [f"Actor_{i+1}" for i in range(num_actores)]
        st.session_state.mid = None
        st.session_state.mao_1 = None
        st.session_state.mao_2 = None
    
    if st.session_state.actores:
        st.write("**Editar nombres:**")
        nuevos_actores = []
        for i, actor in enumerate(st.session_state.actores):
            nuevo_nombre = st.text_input(f"Actor {i+1}", value=actor, key=f"actor_name_{i}")
            nuevos_actores.append(nuevo_nombre)
        st.session_state.actores = nuevos_actores
    
    st.divider()
    
    st.subheader("2. Definir Objetivos")
    num_objetivos = st.number_input("Número de objetivos", min_value=2, max_value=20, value=5, step=1)
    
    if st.button("Configurar Objetivos", key="btn_objetivos"):
        st.session_state.objetivos = [f"Obj_{i+1}" for i in range(num_objetivos)]
        st.session_state.mao_1 = None
        st.session_state.mao_2 = None
    
    if st.session_state.objetivos:
        st.write("**Editar nombres:**")
        nuevos_objetivos = []
        for i, obj in enumerate(st.session_state.objetivos):
            nuevo_nombre = st.text_input(f"Objetivo {i+1}", value=obj, key=f"obj_name_{i}")
            nuevos_objetivos.append(nuevo_nombre)
        st.session_state.objetivos = nuevos_objetivos
    
    st.divider()
    
    st.subheader("3. Parámetros")
    k_potencia = st.slider("Potencia K (MIDI)", min_value=2, max_value=5, value=2)
    
    st.divider()
    st.markdown("""<div style="text-align: center; font-size: 0.8rem; color: #666;">
    <strong>JETLEX Strategic Consulting</strong><br>Martín Pratto Chiarella - 2025</div>""", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📘 Metodología", "📖 Guía de Uso", "📊 1MAO (Posición)", "🎯 2MAO (Prioridad)", 
    "🔗 MID (Fuerza)", "🔄 Cálculos", "💾 Exportar"
])

# TAB 1: METODOLOGIA
with tab1:
    st.header("📘 Metodología MACTOR")
    
    st.markdown("""
    <div class="method-box">
    <h3>🎭 ¿Qué es MACTOR?</h3>
    <p><strong>MACTOR</strong> (Matriz de Alianzas y Conflictos: Tácticas, Objetivos y Recomendaciones) 
    es un método de prospectiva desarrollado por <strong>Michel Godet</strong> que permite analizar 
    el juego de actores: identificar actores clave, analizar objetivos, alianzas y conflictos, 
    y evaluar el equilibrio de poder.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>📥 Matrices de ENTRADA</h4>
        <table>
        <tr><th>Matriz</th><th>Descripción</th><th>Escala</th></tr>
        <tr><td><strong>1MAO</strong></td><td>Posición del actor</td><td>-1, 0, +1</td></tr>
        <tr><td><strong>2MAO</strong></td><td>Prioridad/Intensidad</td><td>0 a 3</td></tr>
        <tr><td><strong>MID</strong></td><td>Medios de presión</td><td>0 a 3</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box">
        <h4>📤 Matrices CALCULADAS</h4>
        <table>
        <tr><th>Matriz</th><th>Descripción</th></tr>
        <tr><td><strong>MIDI</strong></td><td>Influencias Directas e Indirectas</td></tr>
        <tr><td><strong>CAA/DAA</strong></td><td>Convergencias/Divergencias</td></tr>
        <tr><td><strong>3MAO</strong></td><td>Posición valorada por fuerza</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.subheader("🎯 Clasificación de Actores")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="warning-box">
        <h4>🔴 DOMINANTES</h4>
        <p><strong>Alta Influencia + Baja Dependencia</strong></p>
        <p>Actores más poderosos. Pueden imponer su voluntad.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #9e9e9e; margin: 1rem 0;">
        <h4>⚪ DOMINADOS</h4>
        <p><strong>Baja Influencia + Alta Dependencia</strong></p>
        <p>Actores débiles que dependen de otros.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>🔵 DE ENLACE</h4>
        <p><strong>Alta Influencia + Alta Dependencia</strong></p>
        <p>Actores "bisagra" muy conectados.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="example-box">
        <h4>🟡 AUTÓNOMOS</h4>
        <p><strong>Baja Influencia + Baja Dependencia</strong></p>
        <p>Actores independientes, poco conectados.</p>
        </div>
        """, unsafe_allow_html=True)

# TAB 2: GUIA DE USO
with tab2:
    st.header("📖 Guía Completa de Uso")
    
    st.markdown("""<div class="info-box"><h3>🚀 Flujo de Trabajo</h3>
    <p>Sigue estos pasos en orden:</p></div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="step-box"><h4>📌 PASO 1: Configurar Actores y Objetivos</h4>
    <p>En el panel lateral: define número y nombres de actores y objetivos.</p></div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="step-box"><h4>📌 PASO 2: Completar 1MAO (Posición)</h4>
    <p><strong>+1</strong> = A favor | <strong>0</strong> = Neutral | <strong>-1</strong> = En contra</p></div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="step-box"><h4>📌 PASO 3: Completar 2MAO (Prioridad)</h4>
    <p><strong>0</strong> = Nulo | <strong>1</strong> = Débil | <strong>2</strong> = Medio | <strong>3</strong> = Fuerte</p></div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="step-box"><h4>📌 PASO 4: Completar MID (Relaciones de Fuerza)</h4>
    <p><strong>0</strong> = Nula | <strong>1</strong> = Débil | <strong>2</strong> = Media | <strong>3</strong> = Fuerte</p>
    <p><em>La diagonal es 0. La matriz NO es simétrica.</em></p></div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="step-box"><h4>📌 PASO 5: Calcular y Analizar</h4>
    <p>Presiona "Calcular Todo" en la pestaña Cálculos.</p></div>""", unsafe_allow_html=True)

# TAB 3: 1MAO
with tab3:
    st.header("📊 Matriz 1MAO - Posición de Actores")
    st.markdown("""<div class="info-box"><strong>Escala:</strong> <strong>-1</strong> (en contra) | <strong>0</strong> (neutral) | <strong>+1</strong> (a favor)</div>""", unsafe_allow_html=True)
    
    if st.session_state.actores and st.session_state.objetivos:
        if st.session_state.mao_1 is None:
            st.session_state.mao_1 = pd.DataFrame(
                np.zeros((len(st.session_state.actores), len(st.session_state.objetivos)), dtype=int),
                index=st.session_state.actores, columns=st.session_state.objetivos)
        
        mao_1_editada = st.data_editor(st.session_state.mao_1, use_container_width=True, key="mao1_editor")
        
        if st.button("💾 Guardar 1MAO", key="save_mao1"):
            st.session_state.mao_1 = mao_1_editada.clip(-1, 1).astype(int)
            st.success("✅ 1MAO guardada")
    else:
        st.warning("⚠️ Primero configura actores y objetivos en el panel lateral")

# TAB 4: 2MAO
with tab4:
    st.header("🎯 Matriz 2MAO - Prioridad/Intensidad")
    st.markdown("""<div class="info-box"><strong>Escala:</strong> <strong>0</strong> (nulo) | <strong>1</strong> (débil) | <strong>2</strong> (medio) | <strong>3</strong> (fuerte)</div>""", unsafe_allow_html=True)
    
    if st.session_state.actores and st.session_state.objetivos:
        if st.session_state.mao_2 is None:
            st.session_state.mao_2 = pd.DataFrame(
                np.zeros((len(st.session_state.actores), len(st.session_state.objetivos)), dtype=int),
                index=st.session_state.actores, columns=st.session_state.objetivos)
        
        mao_2_editada = st.data_editor(st.session_state.mao_2, use_container_width=True, key="mao2_editor")
        
        if st.button("💾 Guardar 2MAO", key="save_mao2"):
            st.session_state.mao_2 = mao_2_editada.clip(0, 3).astype(int)
            st.success("✅ 2MAO guardada")
    else:
        st.warning("⚠️ Primero configura actores y objetivos en el panel lateral")

# TAB 5: MID
with tab5:
    st.header("🔗 Matriz MID - Relaciones de Fuerza")
    st.markdown("""<div class="info-box"><strong>Escala:</strong> <strong>0</strong> (nula) | <strong>1</strong> (débil) | <strong>2</strong> (media) | <strong>3</strong> (fuerte)<br><em>Diagonal = 0. Matriz NO simétrica.</em></div>""", unsafe_allow_html=True)
    
    if st.session_state.actores:
        if st.session_state.mid is None:
            st.session_state.mid = pd.DataFrame(
                np.zeros((len(st.session_state.actores), len(st.session_state.actores)), dtype=int),
                index=st.session_state.actores, columns=st.session_state.actores)
        
        mid_editada = st.data_editor(st.session_state.mid, use_container_width=True, key="mid_editor")
        
        if st.button("💾 Guardar MID", key="save_mid"):
            mid_array = mid_editada.values.astype(int)
            np.fill_diagonal(mid_array, 0)
            st.session_state.mid = pd.DataFrame(mid_array.clip(0, 3), index=st.session_state.actores, columns=st.session_state.actores)
            st.success("✅ MID guardada (diagonal = 0)")
    else:
        st.warning("⚠️ Primero configura actores en el panel lateral")

# TAB 6: CALCULOS
with tab6:
    st.header("🔄 Cálculos y Análisis")
    
    matrices_completas = all([
        st.session_state.mao_1 is not None,
        st.session_state.mao_2 is not None,
        st.session_state.mid is not None and st.session_state.mid.values.sum() > 0
    ])
    
    if matrices_completas:
        if st.button("🚀 Calcular Todo", type="primary", key="calc_all"):
            with st.spinner("Calculando..."):
                mao_1_array = st.session_state.mao_1.values.astype(float)
                mao_2_array = st.session_state.mao_2.values.astype(float)
                mid_array = st.session_state.mid.values.astype(float)
                
                midi_array = calcular_midi(mid_array, k=k_potencia)
                st.session_state.midi = pd.DataFrame(midi_array, index=st.session_state.actores, columns=st.session_state.actores)
                
                influencias, dependencias, balance = calcular_balance(midi_array)
                st.session_state.influencias = influencias
                st.session_state.dependencias = dependencias
                st.session_state.balance = balance
                st.session_state.ri_star = calcular_coeficiente_ri(influencias, dependencias)
                st.session_state.clasificacion = clasificar_actores(influencias, dependencias)
                
                st.session_state.conv_simple = pd.DataFrame(calcular_convergencias_simples(mao_1_array), index=st.session_state.actores, columns=st.session_state.actores)
                st.session_state.div_simple = pd.DataFrame(calcular_divergencias_simples(mao_1_array), index=st.session_state.actores, columns=st.session_state.actores)
                st.session_state.conv_pond = pd.DataFrame(calcular_convergencias_ponderadas(mao_1_array, mao_2_array), index=st.session_state.actores, columns=st.session_state.actores)
                st.session_state.div_pond = pd.DataFrame(calcular_divergencias_ponderadas(mao_1_array, mao_2_array), index=st.session_state.actores, columns=st.session_state.actores)
                
                mao_3_array = calcular_3mao(mao_1_array, mao_2_array, midi_array)
                st.session_state.mao_3 = pd.DataFrame(mao_3_array, index=st.session_state.actores, columns=st.session_state.objetivos)
                
                mov_pos, mov_neg = calcular_movilizacion_objetivos(mao_1_array, mao_2_array)
                st.session_state.mov_positiva = mov_pos
                st.session_state.mov_negativa = mov_neg
                
                st.success("✅ Todos los cálculos completados!")
        
        if hasattr(st.session_state, 'midi') and st.session_state.midi is not None:
            subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs(["MIDI", "Plano Actores", "Convergencias", "Divergencias", "Movilización"])
            
            with subtab1:
                st.subheader("Matriz MIDI")
                st.dataframe(st.session_state.midi.round(2), use_container_width=True)
                
                df_resumen = pd.DataFrame({
                    'Actor': st.session_state.actores,
                    'Influencia': st.session_state.influencias.round(2),
                    'Dependencia': st.session_state.dependencias.round(2),
                    'Balance': st.session_state.balance.round(2),
                    'ri*': st.session_state.ri_star.round(3),
                    'Clasificación': st.session_state.clasificacion
                })
                st.subheader("📋 Resumen de Actores")
                st.dataframe(df_resumen, use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Influencia', x=df_resumen['Actor'], y=df_resumen['Influencia'], marker_color='#2ecc71'))
                fig.add_trace(go.Bar(name='Dependencia', x=df_resumen['Actor'], y=df_resumen['Dependencia'], marker_color='#e74c3c'))
                fig.update_layout(title="Influencia vs Dependencia", barmode='group', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab2:
                st.subheader("Plano de Actores")
                df_plano = pd.DataFrame({'Actor': st.session_state.actores, 'Influencia': st.session_state.influencias, 'Dependencia': st.session_state.dependencias, 'Clasificacion': st.session_state.clasificacion})
                color_map = {'Dominante': '#e74c3c', 'Enlace': '#3498db', 'Dominado': '#95a5a6', 'Autonomo': '#f39c12'}
                
                fig = go.Figure()
                for clasif in color_map.keys():
                    df_temp = df_plano[df_plano['Clasificacion'] == clasif]
                    if len(df_temp) > 0:
                        fig.add_trace(go.Scatter(x=df_temp['Dependencia'], y=df_temp['Influencia'], mode='markers+text', name=clasif, text=df_temp['Actor'], textposition='top center', marker=dict(size=15, color=color_map[clasif])))
                
                fig.add_hline(y=np.median(st.session_state.influencias), line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=np.median(st.session_state.dependencias), line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text="DOMINANTES", showarrow=False, font=dict(size=11, color="#e74c3c"))
                fig.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper", text="ENLACE", showarrow=False, font=dict(size=11, color="#3498db"))
                fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text="DOMINADOS", showarrow=False, font=dict(size=11, color="#95a5a6"))
                fig.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper", text="AUTÓNOMOS", showarrow=False, font=dict(size=11, color="#f39c12"))
                fig.update_layout(title="Clasificación de Actores", xaxis_title="Dependencia", yaxis_title="Influencia", height=550)
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab3:
                st.subheader("🤝 Convergencias")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Simples**")
                    st.dataframe(st.session_state.conv_simple.astype(int), use_container_width=True)
                with col2:
                    st.write("**Ponderadas**")
                    st.dataframe(st.session_state.conv_pond.round(1), use_container_width=True)
                
                fig = go.Figure(data=go.Heatmap(z=st.session_state.conv_pond.values, x=st.session_state.conv_pond.columns.tolist(), y=st.session_state.conv_pond.index.tolist(), colorscale='Greens'))
                fig.update_layout(title="Convergencias Ponderadas", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab4:
                st.subheader("⚔️ Divergencias")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Simples**")
                    st.dataframe(st.session_state.div_simple.astype(int), use_container_width=True)
                with col2:
                    st.write("**Ponderadas**")
                    st.dataframe(st.session_state.div_pond.round(1), use_container_width=True)
                
                fig = go.Figure(data=go.Heatmap(z=st.session_state.div_pond.values, x=st.session_state.div_pond.columns.tolist(), y=st.session_state.div_pond.index.tolist(), colorscale='Reds'))
                fig.update_layout(title="Divergencias Ponderadas", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab5:
                st.subheader("📊 Movilización sobre Objetivos")
                df_mov = pd.DataFrame({'Objetivo': st.session_state.objetivos, 'A Favor': st.session_state.mov_positiva, 'En Contra': st.session_state.mov_negativa, 'Balance': st.session_state.mov_positiva - st.session_state.mov_negativa})
                st.dataframe(df_mov, use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='A Favor', x=st.session_state.objetivos, y=st.session_state.mov_positiva, marker_color='#27ae60'))
                fig.add_trace(go.Bar(name='En Contra', x=st.session_state.objetivos, y=-st.session_state.mov_negativa, marker_color='#c0392b'))
                fig.update_layout(title="Movilización por Objetivo", barmode='relative', height=400)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""<div class="warning-box"><strong>⚠️ Matrices incompletas</strong><p>Completa 1MAO, 2MAO y MID antes de calcular.</p></div>""", unsafe_allow_html=True)

# TAB 7: EXPORTAR
with tab7:
    st.header("💾 Exportar Resultados")
    
    if hasattr(st.session_state, 'midi') and st.session_state.midi is not None:
        st.markdown("""<div class="success-box"><h4>✅ Análisis completado</h4><p>Exporta a Excel con múltiples hojas.</p></div>""", unsafe_allow_html=True)
        
        nombre_proyecto = st.text_input("Nombre del proyecto", value="analisis_mactor")
        
        if st.button("📥 Generar Excel", type="primary"):
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                if st.session_state.mao_1 is not None:
                    st.session_state.mao_1.to_excel(writer, sheet_name='1MAO')
                if st.session_state.mao_2 is not None:
                    st.session_state.mao_2.to_excel(writer, sheet_name='2MAO')
                if st.session_state.mid is not None:
                    st.session_state.mid.to_excel(writer, sheet_name='MID')
                st.session_state.midi.to_excel(writer, sheet_name='MIDI')
                st.session_state.conv_simple.to_excel(writer, sheet_name='CAA_Simple')
                st.session_state.conv_pond.to_excel(writer, sheet_name='CAA_Pond')
                st.session_state.div_simple.to_excel(writer, sheet_name='DAA_Simple')
                st.session_state.div_pond.to_excel(writer, sheet_name='DAA_Pond')
                st.session_state.mao_3.to_excel(writer, sheet_name='3MAO')
                pd.DataFrame({'Actor': st.session_state.actores, 'Influencia': st.session_state.influencias, 'Dependencia': st.session_state.dependencias, 'Balance': st.session_state.balance, 'ri_star': st.session_state.ri_star, 'Clasificacion': st.session_state.clasificacion}).to_excel(writer, sheet_name='Resumen_Actores', index=False)
                pd.DataFrame({'Objetivo': st.session_state.objetivos, 'Mov_Favor': st.session_state.mov_positiva, 'Mov_Contra': st.session_state.mov_negativa, 'Balance': st.session_state.mov_positiva - st.session_state.mov_negativa}).to_excel(writer, sheet_name='Movilizacion', index=False)
            
            buffer.seek(0)
            st.download_button(label="📥 Descargar Excel", data=buffer, file_name=f"{nombre_proyecto}_mactor.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success("✅ Excel generado!")
    else:
        st.markdown("""<div class="warning-box">⚠️ Primero completa los cálculos.</div>""", unsafe_allow_html=True)

# FOOTER
st.divider()
st.markdown("""<div style="text-align: center; color: #666; padding: 1rem;">
<p><strong>🎭 MACTOR</strong> - Michel Godet (LIPSOR)</p>
<p>Escalas según metodología oficial LISA Institute</p>
<hr style="width: 50%; margin: 1rem auto;">
<p><strong>JETLEX Strategic Consulting</strong> | Martín Pratto Chiarella - 2025</p>
</div>""", unsafe_allow_html=True)
