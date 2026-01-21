"""
MACTOR - Metodo de Analisis de Actores
Matriz de Alianzas y Conflictos: Tacticas, Objetivos y Recomendaciones

Autor: JETLEX Strategic Consulting / Martin Ezequiel CUELLO
Basado en el metodo de Michel Godet
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Configuracion de la pagina
st.set_page_config(
    page_title="MACTOR - Analisis de Actores",
    page_icon="🎭",
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
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)

# Funciones de calculo MACTOR
def calcular_midi(mid, k=2):
    """Calcula la Matriz de Influencias Directas e Indirectas (MIDI)"""
    n = mid.shape[0]
    midi = mid.copy().astype(float)
    mid_power = mid.copy().astype(float)
    
    for i in range(2, k + 1):
        mid_power = np.dot(mid_power, mid)
        midi += mid_power
    
    # Normalizar diagonal a 0
    np.fill_diagonal(midi, 0)
    
    return midi

def calcular_balance(midi):
    """Calcula el balance de influencias (ri)"""
    influencias = midi.sum(axis=1)
    dependencias = midi.sum(axis=0)
    balance = influencias - dependencias
    return influencias, dependencias, balance

def calcular_3mao(mao, midi):
    """Calcula 3MAO - Posicion valorada de actores sobre objetivos"""
    return np.dot(midi, mao)

def calcular_convergencias(mao_1):
    """Calcula matriz de convergencias entre actores"""
    n_actores = mao_1.shape[0]
    convergencias = np.zeros((n_actores, n_actores))
    
    for i in range(n_actores):
        for j in range(n_actores):
            if i != j:
                # Contar objetivos donde ambos tienen misma posicion (+1 o -1)
                conv = np.sum((mao_1[i] == mao_1[j]) & (mao_1[i] != 0))
                convergencias[i, j] = conv
    
    return convergencias

def calcular_divergencias(mao_1):
    """Calcula matriz de divergencias entre actores"""
    n_actores = mao_1.shape[0]
    divergencias = np.zeros((n_actores, n_actores))
    
    for i in range(n_actores):
        for j in range(n_actores):
            if i != j:
                # Contar objetivos donde tienen posiciones opuestas
                div = np.sum((mao_1[i] == -mao_1[j]) & (mao_1[i] != 0) & (mao_1[j] != 0))
                divergencias[i, j] = div
    
    return divergencias

def clasificar_actores(influencias, dependencias):
    """Clasifica actores en cuadrantes"""
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

# Inicializar session state
if 'actores' not in st.session_state:
    st.session_state.actores = []
if 'objetivos' not in st.session_state:
    st.session_state.objetivos = []
if 'mao' not in st.session_state:
    st.session_state.mao = None
if 'mao_1' not in st.session_state:
    st.session_state.mao_1 = None
if 'mao_2' not in st.session_state:
    st.session_state.mao_2 = None
if 'mid' not in st.session_state:
    st.session_state.mid = None

# Header principal
st.markdown('<div class="main-header">🎭 MACTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Metodo de Analisis de Juego de Actores</div>', unsafe_allow_html=True)

# Sidebar para configuracion
with st.sidebar:
    st.header("⚙️ Configuracion")
    
    st.subheader("1. Definir Actores")
    num_actores = st.number_input("Numero de actores", min_value=2, max_value=30, value=5, step=1)
    
    if st.button("Configurar Actores", key="btn_actores"):
        st.session_state.actores = [f"Actor_{i+1}" for i in range(num_actores)]
        st.session_state.mid = None
        st.session_state.mao = None
        st.session_state.mao_1 = None
        st.session_state.mao_2 = None
    
    if st.session_state.actores:
        st.write("**Editar nombres de actores:**")
        nuevos_actores = []
        for i, actor in enumerate(st.session_state.actores):
            nuevo_nombre = st.text_input(
                f"Actor {i+1}",
                value=actor,
                key=f"actor_name_{i}"
            )
            nuevos_actores.append(nuevo_nombre)
        st.session_state.actores = nuevos_actores
    
    st.divider()
    
    st.subheader("2. Definir Objetivos")
    num_objetivos = st.number_input("Numero de objetivos", min_value=2, max_value=30, value=5, step=1)
    
    if st.button("Configurar Objetivos", key="btn_objetivos"):
        st.session_state.objetivos = [f"Objetivo_{i+1}" for i in range(num_objetivos)]
        st.session_state.mao = None
        st.session_state.mao_1 = None
        st.session_state.mao_2 = None
    
    if st.session_state.objetivos:
        st.write("**Editar nombres de objetivos:**")
        nuevos_objetivos = []
        for i, obj in enumerate(st.session_state.objetivos):
            nuevo_nombre = st.text_input(
                f"Objetivo {i+1}",
                value=obj,
                key=f"obj_name_{i}"
            )
            nuevos_objetivos.append(nuevo_nombre)
        st.session_state.objetivos = nuevos_objetivos
    
    st.divider()
    
    st.subheader("3. Parametros")
    k_potencia = st.slider("Potencia K (MIDI)", min_value=2, max_value=5, value=2)

# Tabs principales
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📘 Introduccion",
    "🎯 MAO",
    "📊 1MAO / 2MAO",
    "🔗 MID",
    "🔄 Calculos",
    "💾 Exportar"
])

# TAB 1: INTRODUCCION
with tab1:
    st.header("Introduccion al Metodo MACTOR")
    
    st.markdown("""
    <div class="info-box">
    <h3>¿Que es MACTOR?</h3>
    <p>MACTOR (Metodo de Analisis de Juego de Actores) es una herramienta de prospectiva 
    estrategica desarrollada por Michel Godet que permite analizar las relaciones de fuerza 
    entre actores y estudiar sus convergencias y divergencias frente a objetivos estrategicos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Matrices de Entrada")
        st.markdown("""
        | Matriz | Descripcion | Valores |
        |--------|-------------|---------|
        | **MAO** | Influencia de actores sobre objetivos | 0-4 |
        | **1MAO** | Posicion de actores frente a objetivos | -1, 0, +1 |
        | **2MAO** | Prioridad de objetivos para actores | 0-4 |
        | **MID** | Influencias directas entre actores | 0-4 |
        """)
    
    with col2:
        st.subheader("📊 Matrices Calculadas")
        st.markdown("""
        | Matriz | Descripcion |
        |--------|-------------|
        | **MIDI** | Influencias Directas e Indirectas |
        | **3MAO** | Posicion valorada (MIDI x MAO) |
        | **Convergencias** | Objetivos en comun entre actores |
        | **Divergencias** | Objetivos en conflicto entre actores |
        """)
    
    st.subheader("🎯 Flujo de Trabajo")
    st.markdown("""
    1. **Definir** actores y objetivos en el panel lateral
    2. **Completar** las matrices MAO, 1MAO, 2MAO y MID
    3. **Calcular** las matrices derivadas (MIDI, 3MAO, convergencias)
    4. **Analizar** los resultados y exportar
    """)

# TAB 2: MAO
with tab2:
    st.header("🎯 Matriz MAO - Actores x Objetivos")
    
    st.markdown("""
    <div class="info-box">
    <strong>Instrucciones:</strong> Evalua la influencia que cada actor tiene sobre cada objetivo.
    <br>Escala: <strong>0</strong> (nula) | <strong>1</strong> (debil) | <strong>2</strong> (media) | <strong>3</strong> (fuerte) | <strong>4</strong> (muy fuerte)
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.actores and st.session_state.objetivos:
        # Inicializar MAO si no existe
        if st.session_state.mao is None:
            st.session_state.mao = pd.DataFrame(
                np.zeros((len(st.session_state.actores), len(st.session_state.objetivos)), dtype=int),
                index=st.session_state.actores,
                columns=st.session_state.objetivos
            )
        
        # Editor de datos
        mao_editada = st.data_editor(
            st.session_state.mao,
            use_container_width=True,
            key="mao_editor"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("💾 Guardar MAO", key="save_mao"):
                st.session_state.mao = mao_editada.clip(0, 4).astype(int)
                st.success("✅ MAO guardada correctamente")
        
        # Visualizacion
        if st.session_state.mao is not None and st.session_state.mao.values.sum() > 0:
            st.subheader("📊 Visualizacion MAO")
            
            fig = go.Figure(data=go.Heatmap(
                z=st.session_state.mao.values,
                x=st.session_state.mao.columns.tolist(),
                y=st.session_state.mao.index.tolist(),
                colorscale='Blues',
                showscale=True
            ))
            fig.update_layout(
                title="Matriz MAO - Influencia de Actores sobre Objetivos",
                xaxis_title="Objetivos",
                yaxis_title="Actores",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Primero configura los actores y objetivos en el panel lateral")

# TAB 3: 1MAO / 2MAO
with tab3:
    st.header("📊 Matrices 1MAO y 2MAO")
    
    if st.session_state.actores and st.session_state.objetivos:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1MAO - Posicion")
            st.markdown("""
            <div class="info-box">
            <strong>Valores:</strong> <strong>-1</strong> (en contra) | <strong>0</strong> (neutral) | <strong>+1</strong> (a favor)
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.mao_1 is None:
                st.session_state.mao_1 = pd.DataFrame(
                    np.zeros((len(st.session_state.actores), len(st.session_state.objetivos)), dtype=int),
                    index=st.session_state.actores,
                    columns=st.session_state.objetivos
                )
            
            mao_1_editada = st.data_editor(
                st.session_state.mao_1,
                use_container_width=True,
                key="mao1_editor"
            )
            
            if st.button("💾 Guardar 1MAO", key="save_mao1"):
                st.session_state.mao_1 = mao_1_editada.clip(-1, 1).astype(int)
                st.success("✅ 1MAO guardada")
        
        with col2:
            st.subheader("2MAO - Prioridad")
            st.markdown("""
            <div class="info-box">
            <strong>Escala:</strong> <strong>0</strong> (nula) a <strong>4</strong> (muy alta prioridad)
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.mao_2 is None:
                st.session_state.mao_2 = pd.DataFrame(
                    np.zeros((len(st.session_state.actores), len(st.session_state.objetivos)), dtype=int),
                    index=st.session_state.actores,
                    columns=st.session_state.objetivos
                )
            
            mao_2_editada = st.data_editor(
                st.session_state.mao_2,
                use_container_width=True,
                key="mao2_editor"
            )
            
            if st.button("💾 Guardar 2MAO", key="save_mao2"):
                st.session_state.mao_2 = mao_2_editada.clip(0, 4).astype(int)
                st.success("✅ 2MAO guardada")
    else:
        st.warning("⚠️ Primero configura los actores y objetivos en el panel lateral")

# TAB 4: MID
with tab4:
    st.header("🔗 Matriz MID - Influencias Directas entre Actores")
    
    st.markdown("""
    <div class="info-box">
    <strong>Instrucciones:</strong> Evalua la influencia directa que cada actor (fila) ejerce sobre cada otro actor (columna).
    <br>Escala: <strong>0</strong> (nula) | <strong>1</strong> (debil) | <strong>2</strong> (media) | <strong>3</strong> (fuerte) | <strong>4</strong> (muy fuerte)
    <br><em>La diagonal se pone automaticamente a 0.</em>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.actores:
        if st.session_state.mid is None:
            st.session_state.mid = pd.DataFrame(
                np.zeros((len(st.session_state.actores), len(st.session_state.actores)), dtype=int),
                index=st.session_state.actores,
                columns=st.session_state.actores
            )
        
        mid_editada = st.data_editor(
            st.session_state.mid,
            use_container_width=True,
            key="mid_editor"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("💾 Guardar MID", key="save_mid"):
                # Asegurar diagonal = 0
                mid_array = mid_editada.values.astype(int)
                np.fill_diagonal(mid_array, 0)
                st.session_state.mid = pd.DataFrame(
                    mid_array.clip(0, 4),
                    index=st.session_state.actores,
                    columns=st.session_state.actores
                )
                st.success("✅ MID guardada (diagonal = 0)")
        
        # Visualizacion
        if st.session_state.mid is not None and st.session_state.mid.values.sum() > 0:
            st.subheader("📊 Visualizacion MID")
            
            fig = go.Figure(data=go.Heatmap(
                z=st.session_state.mid.values,
                x=st.session_state.mid.columns.tolist(),
                y=st.session_state.mid.index.tolist(),
                colorscale='Viridis',
                showscale=True
            ))
            fig.update_layout(
                title="Matriz MID - Influencias Directas entre Actores",
                xaxis_title="Actores (influenciados)",
                yaxis_title="Actores (influyentes)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Primero configura los actores en el panel lateral")

# TAB 5: CALCULOS
with tab5:
    st.header("🔄 Calculos y Analisis")
    
    # Verificar que todas las matrices estan completas
    matrices_completas = all([
        st.session_state.mao is not None and st.session_state.mao.values.sum() > 0,
        st.session_state.mao_1 is not None,
        st.session_state.mao_2 is not None,
        st.session_state.mid is not None and st.session_state.mid.values.sum() > 0
    ])
    
    if matrices_completas:
        if st.button("🚀 Calcular Todo", type="primary", key="calc_all"):
            with st.spinner("Calculando matrices..."):
                # Obtener arrays
                mao_array = st.session_state.mao.values.astype(float)
                mao_1_array = st.session_state.mao_1.values.astype(float)
                mao_2_array = st.session_state.mao_2.values.astype(float)
                mid_array = st.session_state.mid.values.astype(float)
                
                # Calcular MIDI
                midi_array = calcular_midi(mid_array, k=k_potencia)
                st.session_state.midi = pd.DataFrame(
                    midi_array,
                    index=st.session_state.actores,
                    columns=st.session_state.actores
                )
                
                # Calcular balance
                influencias, dependencias, balance = calcular_balance(midi_array)
                st.session_state.influencias = influencias
                st.session_state.dependencias = dependencias
                st.session_state.balance = balance
                
                # Clasificar actores
                st.session_state.clasificacion = clasificar_actores(influencias, dependencias)
                
                # Calcular 3MAO
                mao_3_array = calcular_3mao(mao_array, midi_array)
                st.session_state.mao_3 = pd.DataFrame(
                    mao_3_array,
                    index=st.session_state.actores,
                    columns=st.session_state.objetivos
                )
                
                # Calcular convergencias y divergencias
                conv_array = calcular_convergencias(mao_1_array)
                div_array = calcular_divergencias(mao_1_array)
                st.session_state.convergencias = pd.DataFrame(
                    conv_array,
                    index=st.session_state.actores,
                    columns=st.session_state.actores
                )
                st.session_state.divergencias = pd.DataFrame(
                    div_array,
                    index=st.session_state.actores,
                    columns=st.session_state.actores
                )
                
                st.success("✅ Todos los calculos completados!")
        
        # Mostrar resultados si existen
        if hasattr(st.session_state, 'midi') and st.session_state.midi is not None:
            subtab1, subtab2, subtab3, subtab4 = st.tabs([
                "MIDI", "Plano de Actores", "3MAO", "Convergencias/Divergencias"
            ])
            
            with subtab1:
                st.subheader("Matriz MIDI - Influencias Directas e Indirectas")
                st.dataframe(st.session_state.midi.round(2), use_container_width=True)
                
                # Grafico de barras
                df_balance = pd.DataFrame({
                    'Actor': st.session_state.actores,
                    'Influencia': st.session_state.influencias,
                    'Dependencia': st.session_state.dependencias,
                    'Balance': st.session_state.balance
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Influencia',
                    x=df_balance['Actor'],
                    y=df_balance['Influencia'],
                    marker_color='#2ecc71'
                ))
                fig.add_trace(go.Bar(
                    name='Dependencia',
                    x=df_balance['Actor'],
                    y=df_balance['Dependencia'],
                    marker_color='#e74c3c'
                ))
                fig.update_layout(
                    title="Influencia vs Dependencia por Actor",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab2:
                st.subheader("Plano de Influencias/Dependencias")
                
                df_plano = pd.DataFrame({
                    'Actor': st.session_state.actores,
                    'Influencia': st.session_state.influencias,
                    'Dependencia': st.session_state.dependencias,
                    'Clasificacion': st.session_state.clasificacion
                })
                
                color_map = {
                    'Dominante': '#e74c3c',
                    'Enlace': '#3498db',
                    'Dominado': '#95a5a6',
                    'Autonomo': '#f39c12'
                }
                
                fig = go.Figure()
                
                for clasif in color_map.keys():
                    df_temp = df_plano[df_plano['Clasificacion'] == clasif]
                    if len(df_temp) > 0:
                        fig.add_trace(go.Scatter(
                            x=df_temp['Dependencia'],
                            y=df_temp['Influencia'],
                            mode='markers+text',
                            name=clasif,
                            text=df_temp['Actor'],
                            textposition='top center',
                            marker=dict(size=15, color=color_map[clasif])
                        ))
                
                # Lineas de medianas
                med_inf = np.median(st.session_state.influencias)
                med_dep = np.median(st.session_state.dependencias)
                
                fig.add_hline(y=med_inf, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=med_dep, line_dash="dash", line_color="gray", opacity=0.5)
                
                fig.update_layout(
                    title="Plano de Actores - Clasificacion Estrategica",
                    xaxis_title="Dependencia",
                    yaxis_title="Influencia",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de clasificacion
                st.dataframe(df_plano, use_container_width=True)
            
            with subtab3:
                st.subheader("Matriz 3MAO - Posicion Valorada")
                st.dataframe(st.session_state.mao_3.round(2), use_container_width=True)
                
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.mao_3.values,
                    x=st.session_state.mao_3.columns.tolist(),
                    y=st.session_state.mao_3.index.tolist(),
                    colorscale='RdYlGn',
                    zmid=0,
                    showscale=True
                ))
                fig.update_layout(
                    title="3MAO - Posicion Valorada de Actores sobre Objetivos",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab4:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Matriz de Convergencias")
                    st.dataframe(st.session_state.convergencias, use_container_width=True)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=st.session_state.convergencias.values,
                        x=st.session_state.convergencias.columns.tolist(),
                        y=st.session_state.convergencias.index.tolist(),
                        colorscale='Greens',
                        showscale=True
                    ))
                    fig.update_layout(title="Convergencias entre Actores", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Matriz de Divergencias")
                    st.dataframe(st.session_state.divergencias, use_container_width=True)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=st.session_state.divergencias.values,
                        x=st.session_state.divergencias.columns.tolist(),
                        y=st.session_state.divergencias.index.tolist(),
                        colorscale='Reds',
                        showscale=True
                    ))
                    fig.update_layout(title="Divergencias entre Actores", height=400)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Matrices incompletas</strong>
        <p>Para realizar los calculos, necesitas completar:</p>
        <ul>
            <li>MAO - Matriz de actores x objetivos</li>
            <li>1MAO - Posiciones de actores</li>
            <li>2MAO - Prioridades de actores</li>
            <li>MID - Influencias entre actores</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# TAB 6: EXPORTAR
with tab6:
    st.header("💾 Exportar Resultados")
    
    if hasattr(st.session_state, 'midi') and st.session_state.midi is not None:
        nombre_proyecto = st.text_input("Nombre del proyecto", value="analisis_mactor")
        
        if st.button("📥 Generar Excel Completo", type="primary"):
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Matrices de entrada
                if st.session_state.mao is not None:
                    st.session_state.mao.to_excel(writer, sheet_name='MAO')
                if st.session_state.mao_1 is not None:
                    st.session_state.mao_1.to_excel(writer, sheet_name='1MAO')
                if st.session_state.mao_2 is not None:
                    st.session_state.mao_2.to_excel(writer, sheet_name='2MAO')
                if st.session_state.mid is not None:
                    st.session_state.mid.to_excel(writer, sheet_name='MID')
                
                # Matrices calculadas
                st.session_state.midi.to_excel(writer, sheet_name='MIDI')
                st.session_state.mao_3.to_excel(writer, sheet_name='3MAO')
                st.session_state.convergencias.to_excel(writer, sheet_name='Convergencias')
                st.session_state.divergencias.to_excel(writer, sheet_name='Divergencias')
                
                # Resumen de actores
                df_resumen = pd.DataFrame({
                    'Actor': st.session_state.actores,
                    'Influencia': st.session_state.influencias,
                    'Dependencia': st.session_state.dependencias,
                    'Balance': st.session_state.balance,
                    'Clasificacion': st.session_state.clasificacion
                })
                df_resumen.to_excel(writer, sheet_name='Resumen_Actores', index=False)
            
            buffer.seek(0)
            
            st.download_button(
                label="📥 Descargar Excel",
                data=buffer,
                file_name=f"{nombre_proyecto}_mactor.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success("✅ Excel generado correctamente!")
    else:
        st.warning("⚠️ Primero completa los calculos en la pestana 'Calculos'")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>MACTOR</strong> - Metodo de Analisis de Juego de Actores</p>
    <p>Basado en la metodologia de Michel Godet</p>
    <p>Desarrollado por <strong>JETLEX Strategic Consulting</strong></p>
    <p>Martin Ezequiel CUELLO - 2025</p>
</div>
""", unsafe_allow_html=True)
