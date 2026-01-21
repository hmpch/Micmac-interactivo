"""
MACTOR - Método de Análisis de Actores
Matriz de Alianzas y Conflictos: Tácticas, Objetivos y Recomendaciones

Autor: JETLEX Strategic Consulting / Martín Ezequiel CUELLO
Basado en el método de Michel Godet
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

# Configuración
st.set_page_config(
    page_title="MACTOR - Análisis de Actores",
    page_icon="🎭",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Funciones
def calcular_midi(mid, k=2):
    """Calcula MIDI"""
    n = mid.shape[0]
    midi = mid.copy()
    mid_power = mid.copy()
    
    for i in range(2, k + 1):
        mid_power = np.dot(mid_power, mid)
        midi += mid_power
    
    return midi

def calcular_3mao(mao, midi):
    """Calcula 3MAO"""
    return np.dot(midi, mao)

def calcular_4mao(mao_1, mao_2, midi):
    """Calcula 4MAO"""
    mao_ponderada = mao_1 * mao_2
    return np.dot(midi, mao_ponderada)

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

# Header
st.markdown('<div class="main-header">🎭 MACTOR</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Método de Análisis de Actores</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Configuración")
    
    num_actores = st.number_input("Número de actores", 2, 30, 5, 1)
    
    if st.button("Configurar Actores"):
        st.session_state.actores = [f"Actor_{i+1}" for i in range(num_actores)]
    
    if st.session_state.actores:
        st.write("**Editar nombres:**")
        for i in range(len(st.session_state.actores)):
            st.session_state.actores[i] = st.text_input(
                f"Actor {i+1}",
                value=st.session_state.actores[i],
                key=f"actor_{i}"
            )
    
    st.divider()
    
    num_objetivos = st.number_input("Número de objetivos", 2, 30, 5, 1)
    
    if st.button("Configurar Objetivos"):
        st.session_state.objetivos = [f"Objetivo_{i+1}" for i in range(num_objetivos)]
    
    if st.session_state.objetivos:
        st.write("**Editar nombres:**")
        for i in range(len(st.session_state.objetivos)):
            st.session_state.objetivos[i] = st.text_input(
                f"Objetivo {i+1}",
                value=st.session_state.objetivos[i],
                key=f"obj_{i}"
            )
    
    st.divider()
    k_potencia = st.slider("Potencia K", 2, 5, 2)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Intro",
    "🎯 MAO",
    "📊 1MAO/2MAO",
    "🔗 MID",
    "🔄 Análisis"
])

with tab1:
    st.header("¿Qué es MACTOR?")
    st.markdown("""
    <div class="info-box">
    <h3>Método MACTOR</h3>
    Herramienta de análisis prospectivo que permite:
    - Identificar actores clave
    - Analizar relaciones de fuerza
    - Evaluar convergencias y divergencias
    - Identificar alianzas y conflictos
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Matrices:**
    1. **MAO** - Influencia actores sobre objetivos (0-4)
    2. **1MAO** - Posición de actores (-1, 0, +1)
    3. **2MAO** - Prioridad de objetivos (0-4)
    4. **MID** - Influencias entre actores (0-4)
    5. **MIDI** - Calculada automáticamente
    6. **3MAO** - Calculada automáticamente
    7. **4MAO** - Calculada automáticamente
    """)

with tab2:
    st.header("Matriz MAO")
    st.markdown("""
    <div class="info-box">
    Evalúa influencia de actores sobre objetivos (0-4)
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.actores and st.session_state.objetivos:
        if st.session_state.mao is None:
            st.session_state.mao = pd.DataFrame(
                0,
                index=st.session_state.actores,
                columns=st.session_state.objetivos
            )
        
        mao_editada = st.data_editor(
            st.session_state.mao,
            use_container_width=True,
            key="mao_editor"
        )
        
        if st.button("💾 Guardar MAO"):
            st.session_state.mao = mao_editada
            st.success("✅ Guardado")
            
            fig = go.Figure(data=go.Heatmap(
                z=st.session_state.mao.values,
                x=st.session_state.mao.columns,
                y=st.session_state.mao.index,
                colorscale='Blues'
            ))
            fig.update_layout(title="Matriz MAO", height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Configure actores y objetivos")

with tab3:
    st.header("Matrices 1MAO y 2MAO")
    
    if st.session_state.actores and st.session_state.objetivos:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1MAO - Posición")
            st.caption("(-1, 0, +1)")
            
            if st.session_state.mao_1 is None:
                st.session_state.mao_1 = pd.DataFrame(
                    0,
                    index=st.session_state.actores,
                    columns=st.session_state.objetivos
                )
            
            mao_1_editada = st.data_editor(
                st.session_state.mao_1,
                use_container_width=True,
                key="mao1_editor"
            )
            
            if st.button("💾 Guardar 1MAO"):
                st.session_state.mao_1 = mao_1_editada
                st.success("✅ Guardado")
        
        with col2:
            st.subheader("2MAO - Prioridad")
            st.caption("(0-4)")
            
            if st.session_state.mao_2 is None:
                st.session_state.mao_2 = pd.DataFrame(
                    0,
                    index=st.session_state.actores,
                    columns=st.session_state.objetivos
                )
            
            mao_2_editada = st.data_editor(
                st.session_state.mao_2,
                use_container_width=True,
                key="mao2_editor"
            )
            
            if st.button("💾 Guardar 2MAO"):
                st.session_state.mao_2 = mao_2_editada
                st.success("✅ Guardado")
    else:
        st.warning("Configure actores y objetivos")

with tab4:
    st.header("Matriz MID")
    st.caption("Influencias entre actores (0-4, diagonal = 0)")
    
    if st.session_state.actores:
        if st.session_state.mid is None:
            st.session_state.mid = pd.DataFrame(
                0,
                index=st.session_state.actores,
                columns=st.session_state.actores
            )
        
        mid_editada = st.data_editor(
            st.session_state.mid,
            use_container_width=True,
            key="mid_editor"
        )
        
        if st.button("💾 Guardar MID"):
            for i in range(len(st.session_state.actores)):
                mid_editada.iloc[i, i] = 0
            
            st.session_state.mid = mid_editada
            st.success("✅ Guardado (diagonal = 0)")
            
            fig = go.Figure(data=go.Heatmap(
                z=st.session_state.mid.values,
                x=st.session_state.mid.columns,
                y=st.session_state.mid.index,
                colorscale='Viridis'
            ))
            fig.update_layout(title="Matriz MID", height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Configure actores")

with tab5:
    st.header("Análisis Completo")
    
    todas_completas = all([
        st.session_state.mao is not None,
        st.session_state.mao_1 is not None,
        st.session_state.mao_2 is not None,
        st.session_state.mid is not None
    ])
    
    if todas_completas:
        if st.button("🚀 Calcular"):
            with st.spinner("Calculando..."):
                mao_array = st.session_state.mao.values
                mao_1_array = st.session_state.mao_1.values
                mao_2_array = st.session_state.mao_2.values
                mid_array = st.session_state.mid.values
                
                midi_array = calcular_midi(mid_array, k=k_potencia)
                st.session_state.midi = pd.DataFrame(
                    midi_array,
                    index=st.session_state.actores,
                    columns=st.session_state.actores
                )
                
                mao_3_array = calcular_3mao(mao_array, midi_array)
                st.session_state.mao_3 = pd.DataFrame(
                    mao_3_array,
                    index=st.session_state.actores,
                    columns=st.session_state.objetivos
                )
                
                mao_4_array = calcular_4mao(mao_1_array, mao_2_array, midi_array)
                st.session_state.mao_4 = pd.DataFrame(
                    mao_4_array,
                    index=st.session_state.actores,
                    columns=st.session_state.objetivos
                )
                
                st.success("✅ Completado")
        
        if hasattr(st.session_state, 'midi'):
            subtab1, subtab2, subtab3 = st.tabs(["MIDI", "3MAO", "4MAO"])
            
            with subtab1:
                st.dataframe(st.session_state.midi)
                
                influencias = st.session_state.midi.sum(axis=1)
                dependencias = st.session_state.midi.sum(axis=0)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Influencia',
                    x=st.session_state.actores,
                    y=influencias
                ))
                fig.add_trace(go.Bar(
                    name='Dependencia',
                    x=st.session_state.actores,
                    y=dependencias
                ))
                fig.update_layout(title="MIDI - Influencias", barmode='group', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab2:
                st.dataframe(st.session_state.mao_3)
                
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.mao_3.values,
                    x=st.session_state.mao_3.columns,
                    y=st.session_state.mao_3.index,
                    colorscale='YlOrRd'
                ))
                fig.update_layout(title="3MAO - Implicación", height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with subtab3:
                st.dataframe(st.session_state.mao_4)
                
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.mao_4.values,
                    x=st.session_state.mao_4.columns,
                    y=st.session_state.mao_4.index,
                    colorscale='RdYlGn',
                    zmid=0
                ))
                fig.update_layout(title="4MAO - Movilización", height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Exportar
            st.divider()
            if st.button("📥 Exportar Excel"):
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    st.session_state.mao.to_excel(writer, sheet_name='MAO')
                    st.session_state.mao_1.to_excel(writer, sheet_name='1MAO')
                    st.session_state.mao_2.to_excel(writer, sheet_name='2MAO')
                    st.session_state.mid.to_excel(writer, sheet_name='MID')
                    st.session_state.midi.to_excel(writer, sheet_name='MIDI')
                    st.session_state.mao_3.to_excel(writer, sheet_name='3MAO')
                    st.session_state.mao_4.to_excel(writer, sheet_name='4MAO')
                
                buffer.seek(0)
                st.download_button(
                    "📥 Descargar",
                    buffer,
                    "mactor_resultados.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.warning("Complete todas las matrices")

st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<p><strong>MACTOR</strong> - JETLEX Strategic Consulting</p>
<p>Martín Ezequiel CUELLO © 2025</p>
</div>
""", unsafe_allow_html=True)
```

4. **Commit new file** con mensaje: `Agregar MACTOR completo`

---

## 📄 **ARCHIVO 3: `requirements.txt`**

**¿Qué hace?** Lista todas las librerías que necesitan las apps

**Cómo crearlo:**
1. **Add file** → **Create new file**
2. Nombre: `requirements.txt`
3. Copia esto:
```
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.3
plotly==5.18.0
openpyxl==3.1.2
scipy==1.11.4
networkx==3.2.1
matplotlib==3.8.2
seaborn==0.13.0
