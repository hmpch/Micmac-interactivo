"""
MACTOR PRO - Método de Análisis de Actores
Matriz de Alianzas y Conflictos: Tácticas, Objetivos y Recomendaciones

Autor: JETLEX Strategic Consulting / Martín Pratto Chiarella
Basado en el método de Michel Godet (LIPSOR)
Versión: 4.0 - Entrada dual (manual + archivo) + Gráficos completos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="MACTOR PRO - Análisis de Actores",
    page_icon="🎭",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #8B5CF6;
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
        background-color: #EDE9FE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #8B5CF6;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES DE PROCESAMIENTO DE ARCHIVOS EXCEL
# ============================================================

def procesar_archivo_mactor(uploaded_file):
    """Procesa archivo Excel MACTOR con detección automática"""
    try:
        xl = pd.ExcelFile(uploaded_file)
        hojas = xl.sheet_names
        
        resultado = {
            'hojas': hojas,
            'MAA': None,
            'MAO_2': None,
            'actores': None,
            'objetivos': None,
            'mensajes': []
        }
        
        # Buscar y procesar MAA
        for hoja in hojas:
            if 'MAA' in hoja.upper() and 'JUST' not in hoja.upper():
                df = pd.read_excel(xl, sheet_name=hoja, header=None)
                
                # Detectar inicio de datos
                fila_inicio = 0
                for i in range(min(10, len(df))):
                    val = str(df.iloc[i, 0]).lower() if pd.notna(df.iloc[i, 0]) else ""
                    if 'de' in val and ('sobre' in val or '\\' in val):
                        fila_inicio = i + 1
                        break
                
                if fila_inicio == 0:
                    for i in range(min(10, len(df))):
                        if pd.notna(df.iloc[i, 1]):
                            try:
                                float(df.iloc[i, 1])
                                fila_inicio = i
                                break
                            except:
                                pass
                
                # Extraer actores y matriz
                actores = []
                filas_datos = []
                for i in range(fila_inicio, min(fila_inicio + 25, len(df))):
                    nombre = df.iloc[i, 0]
                    if pd.isna(nombre):
                        continue
                    nombre_str = str(nombre).strip()
                    if nombre_str and not any(x in nombre_str.lower() for x in ['suma', 'total', 'ii', 'di', 'dependencia', 'influencia']):
                        actores.append(nombre_str)
                        filas_datos.append(i)
                
                n = len(actores)
                matriz = np.zeros((n, n))
                for i, fila_idx in enumerate(filas_datos):
                    for j in range(n):
                        val = df.iloc[fila_idx, j + 1]
                        if pd.notna(val):
                            try:
                                matriz[i, j] = float(val)
                            except:
                                matriz[i, j] = 0
                
                np.fill_diagonal(matriz, 0)
                resultado['MAA'] = pd.DataFrame(matriz, index=actores, columns=actores)
                resultado['actores'] = actores
                resultado['mensajes'].append(f"✅ MAA {n}×{n} procesada desde '{hoja}'")
                break
        
        # Buscar y procesar 2MAO
        for hoja in hojas:
            if '2MAO' in hoja.upper() and 'JUST' not in hoja.upper():
                df = pd.read_excel(xl, sheet_name=hoja, header=None)
                
                # Detectar inicio
                fila_headers = 0
                for i in range(min(10, len(df))):
                    val = str(df.iloc[i, 0]).lower() if pd.notna(df.iloc[i, 0]) else ""
                    if 'actor' in val or '\\' in val:
                        fila_headers = i
                        break
                
                fila_inicio = fila_headers + 1
                
                # Extraer objetivos
                objetivos = []
                for j in range(1, min(40, len(df.columns))):
                    val = df.iloc[fila_headers, j]
                    if pd.notna(val):
                        obj_str = str(val).strip()
                        if obj_str.upper().startswith('O'):
                            objetivos.append(obj_str.upper())
                
                # Extraer actores y matriz
                actores_mao = []
                filas_datos = []
                for i in range(fila_inicio, min(fila_inicio + 25, len(df))):
                    nombre = df.iloc[i, 0]
                    if pd.isna(nombre):
                        continue
                    nombre_str = str(nombre).strip()
                    if nombre_str and not any(x in nombre_str.lower() for x in ['suma', 'total', 'moviliz']):
                        actores_mao.append(nombre_str)
                        filas_datos.append(i)
                
                n_actores = len(actores_mao)
                n_objetivos = len(objetivos)
                matriz = np.zeros((n_actores, n_objetivos))
                
                for i, fila_idx in enumerate(filas_datos):
                    for j in range(n_objetivos):
                        val = df.iloc[fila_idx, j + 1]
                        if pd.notna(val):
                            try:
                                matriz[i, j] = float(val)
                            except:
                                matriz[i, j] = 0
                
                resultado['MAO_2'] = pd.DataFrame(matriz, index=actores_mao, columns=objetivos)
                resultado['objetivos'] = objetivos
                if resultado['actores'] is None:
                    resultado['actores'] = actores_mao
                resultado['mensajes'].append(f"✅ 2MAO {n_actores}×{n_objetivos} procesada desde '{hoja}'")
                break
        
        return resultado
    
    except Exception as e:
        return {'error': str(e)}

# ============================================================
# FUNCIONES DE CÁLCULO MACTOR
# ============================================================

def calcular_MIDI(MAA, k=2):
    """Calcula la Matriz de Influencias Directas e Indirectas"""
    M = MAA.values.astype(float)
    n = M.shape[0]
    
    MIDI = M.copy()
    M_power = M.copy()
    
    for i in range(2, k + 1):
        M_power = np.dot(M_power, M)
        MIDI += M_power
    
    np.fill_diagonal(MIDI, 0)
    return pd.DataFrame(MIDI, index=MAA.index, columns=MAA.columns)

def calcular_balance_MIDI(MIDI):
    """Calcula el balance de relaciones de fuerza"""
    M = MIDI.values
    Ii = M.sum(axis=1)
    Di = M.sum(axis=0)
    Ri = Ii / (Di + 0.001)
    Ri_neto = Ii - Di
    
    return pd.DataFrame({
        'Actor': MIDI.index,
        'Ii': np.round(Ii, 1),
        'Di': np.round(Di, 1),
        'Ri': np.round(Ri, 2),
        'Ri_neto': np.round(Ri_neto, 1)
    })

def clasificar_actores(balance_df):
    """Clasifica actores según influencia y dependencia"""
    Ii = balance_df['Ii'].values
    Di = balance_df['Di'].values
    
    med_Ii = np.median(Ii)
    med_Di = np.median(Di)
    
    clasificacion = []
    for i, d in zip(Ii, Di):
        if i >= med_Ii and d < med_Di:
            clasificacion.append("Dominante")
        elif i >= med_Ii and d >= med_Di:
            clasificacion.append("Enlace")
        elif i < med_Ii and d >= med_Di:
            clasificacion.append("Dominado")
        else:
            clasificacion.append("Autónomo")
    
    return clasificacion, med_Ii, med_Di

def calcular_convergencias_divergencias(MAO):
    """Calcula matrices de convergencias y divergencias (simples y ponderadas)"""
    M = MAO.values
    n_actores = M.shape[0]
    actores = MAO.index.tolist()
    
    # Matrices simples (conteo)
    CAA = np.zeros((n_actores, n_actores))
    DAA = np.zeros((n_actores, n_actores))
    
    # Matrices ponderadas (intensidad)
    CAA_pond = np.zeros((n_actores, n_actores))
    DAA_pond = np.zeros((n_actores, n_actores))
    
    for i in range(n_actores):
        for j in range(n_actores):
            if i != j:
                for k in range(M.shape[1]):
                    vi, vj = M[i, k], M[j, k]
                    if vi != 0 and vj != 0:
                        intensidad = min(abs(vi), abs(vj))
                        if (vi > 0 and vj > 0) or (vi < 0 and vj < 0):
                            CAA[i, j] += 1
                            CAA_pond[i, j] += intensidad
                        elif (vi > 0 and vj < 0) or (vi < 0 and vj > 0):
                            DAA[i, j] += 1
                            DAA_pond[i, j] += intensidad
    
    return (pd.DataFrame(CAA, index=actores, columns=actores),
            pd.DataFrame(DAA, index=actores, columns=actores),
            pd.DataFrame(CAA_pond, index=actores, columns=actores),
            pd.DataFrame(DAA_pond, index=actores, columns=actores))

def calcular_3MAO(MAO_2, MIDI):
    """Calcula 3MAO - Posiciones valoradas por coeficiente de poder"""
    actores_mao = MAO_2.index.tolist()
    actores_midi = MIDI.index.tolist()
    
    actores_comunes = [a for a in actores_mao if a in actores_midi]
    
    if len(actores_comunes) == 0:
        return None
    
    MAO = MAO_2.loc[actores_comunes].values
    M = MIDI.loc[actores_comunes, actores_comunes].values
    
    Ii = M.sum(axis=1)
    Di = M.sum(axis=0)
    Ri = Ii - Di
    
    if Ri.max() != Ri.min():
        Ri_norm = 0.5 + (Ri - Ri.min()) / (Ri.max() - Ri.min())
    else:
        Ri_norm = np.ones_like(Ri)
    
    MAO_3 = MAO * Ri_norm.reshape(-1, 1)
    
    return pd.DataFrame(MAO_3, index=actores_comunes, columns=MAO_2.columns)

def calcular_balance_objetivos(MAO, balance_actores=None):
    """Calcula balance neto ponderado por objetivo (viabilidad)"""
    M = MAO.values
    objetivos = MAO.columns.tolist()
    actores = MAO.index.tolist()
    
    # Si hay balance de actores, usar coeficiente de poder
    if balance_actores is not None:
        Ri_neto = balance_actores['Ri_neto'].values
        if Ri_neto.max() != Ri_neto.min():
            coef_poder = 0.5 + (Ri_neto - Ri_neto.min()) / (Ri_neto.max() - Ri_neto.min())
        else:
            coef_poder = np.ones(len(actores))
    else:
        coef_poder = np.ones(len(actores))
    
    resultados = []
    
    for j, obj in enumerate(objetivos):
        posiciones = M[:, j]
        
        # Conteos simples
        n_favor = (posiciones > 0).sum()
        n_contra = (posiciones < 0).sum()
        n_neutro = (posiciones == 0).sum()
        
        # Suma simple
        suma_favor = np.where(posiciones > 0, posiciones, 0).sum()
        suma_contra = np.where(posiciones < 0, posiciones, 0).sum()
        balance_simple = suma_favor + suma_contra
        
        # Balance ponderado por poder
        balance_pond = np.sum(posiciones * coef_poder)
        
        # Intensidad de movilización
        movilizacion = np.abs(posiciones).sum()
        
        # Índice de viabilidad (normalizado)
        if movilizacion > 0:
            viabilidad = balance_pond / movilizacion
        else:
            viabilidad = 0
        
        resultados.append({
            'Objetivo': obj,
            'A_favor': n_favor,
            'Neutros': n_neutro,
            'En_contra': n_contra,
            'Suma_favor': suma_favor,
            'Suma_contra': suma_contra,
            'Balance_simple': balance_simple,
            'Balance_ponderado': round(balance_pond, 2),
            'Movilizacion': movilizacion,
            'Viabilidad': round(viabilidad, 2)
        })
    
    return pd.DataFrame(resultados)

def calcular_implicacion_actores(MAO):
    """Calcula el nivel de implicación de cada actor en el sistema"""
    M = MAO.values
    actores = MAO.index.tolist()
    
    resultados = []
    for i, actor in enumerate(actores):
        posiciones = M[i, :]
        
        n_favor = (posiciones > 0).sum()
        n_contra = (posiciones < 0).sum()
        n_neutro = (posiciones == 0).sum()
        
        intensidad_favor = np.where(posiciones > 0, posiciones, 0).sum()
        intensidad_contra = np.abs(np.where(posiciones < 0, posiciones, 0).sum())
        
        implicacion_total = np.abs(posiciones).sum()
        
        resultados.append({
            'Actor': actor,
            'Obj_favor': n_favor,
            'Obj_neutro': n_neutro,
            'Obj_contra': n_contra,
            'Intensidad_favor': intensidad_favor,
            'Intensidad_contra': intensidad_contra,
            'Implicacion_total': implicacion_total
        })
    
    return pd.DataFrame(resultados)

def calcular_ambivalencia(CAA, DAA):
    """Calcula índice de ambivalencia entre actores"""
    C = CAA.values
    D = DAA.values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        amb = np.minimum(C, D) / (np.maximum(C, D) + 0.001)
        amb = np.nan_to_num(amb, nan=0.0)
    
    return pd.DataFrame(amb, index=CAA.index, columns=CAA.columns)

# ============================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================

def mostrar_grafico_con_descargas(fig, nombre_base, key_suffix=""):
    """Muestra gráfico con opciones de descarga"""
    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': nombre_base,
            'height': 800,
            'width': 1200,
            'scale': 2
        },
        'displaylogo': False,
        'displayModeBar': True
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    col1, col2 = st.columns(2)
    with col1:
        html_buffer = BytesIO()
        html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html_buffer.write(html_content.encode())
        html_buffer.seek(0)
        st.download_button("📥 HTML", html_buffer, f"{nombre_base}.html", "text/html", key=f"html_{key_suffix}")
    
    with col2:
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            st.download_button("📥 PNG", img_bytes, f"{nombre_base}.png", "image/png", key=f"png_{key_suffix}")
        except:
            st.info("📷 Usa el ícono de cámara")

# ============================================================
# GRÁFICOS MACTOR
# ============================================================

def crear_plano_influencias(balance_df, med_Ii, med_Di):
    """Plano de influencias/dependencias de actores"""
    clasificacion, _, _ = clasificar_actores(balance_df)
    balance_df = balance_df.copy()
    balance_df['Clasificación'] = clasificacion
    
    colores = {
        'Dominante': '#DC2626',
        'Enlace': '#7C3AED',
        'Dominado': '#2563EB',
        'Autónomo': '#D97706'
    }
    
    fig = go.Figure()
    
    for clasif in ['Dominante', 'Enlace', 'Dominado', 'Autónomo']:
        df_temp = balance_df[balance_df['Clasificación'] == clasif]
        if len(df_temp) > 0:
            fig.add_trace(go.Scatter(
                x=df_temp['Di'],
                y=df_temp['Ii'],
                mode='markers+text',
                name=clasif,
                text=df_temp['Actor'],
                textposition='top center',
                marker=dict(size=16, color=colores[clasif], line=dict(width=2, color='white')),
                hovertemplate="<b>%{text}</b><br>Influencia (Ii): %{y:.1f}<br>Dependencia (Di): %{x:.1f}<extra></extra>"
            ))
    
    fig.add_hline(y=med_Ii, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Mediana Ii={med_Ii:.1f}")
    fig.add_vline(x=med_Di, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Mediana Di={med_Di:.1f}")
    
    max_Ii = balance_df['Ii'].max() * 1.1
    max_Di = balance_df['Di'].max() * 1.1
    
    fig.add_annotation(x=med_Di*0.3, y=max_Ii*0.95, text="🔴 DOMINANTES", showarrow=False, font=dict(size=14, color='#DC2626'))
    fig.add_annotation(x=max_Di*0.85, y=max_Ii*0.95, text="🟣 ENLACE", showarrow=False, font=dict(size=14, color='#7C3AED'))
    fig.add_annotation(x=max_Di*0.85, y=med_Ii*0.3, text="🔵 DOMINADOS", showarrow=False, font=dict(size=14, color='#2563EB'))
    fig.add_annotation(x=med_Di*0.3, y=med_Ii*0.3, text="🟠 AUTÓNOMOS", showarrow=False, font=dict(size=14, color='#D97706'))
    
    fig.update_layout(
        title="Plano de Influencias/Dependencias entre Actores (MIDI)",
        xaxis_title="Dependencia (Di) →",
        yaxis_title="← Influencia (Ii)",
        height=650,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig

def crear_histograma_balance(balance_df):
    """Histograma de balance de fuerzas"""
    df = balance_df.sort_values('Ri_neto', ascending=True).copy()
    colores = ['#DC2626' if x > 0 else '#2563EB' for x in df['Ri_neto']]
    
    fig = go.Figure(go.Bar(
        x=df['Ri_neto'],
        y=df['Actor'],
        orientation='h',
        marker_color=colores,
        text=df['Ri_neto'].apply(lambda x: f"{x:+.0f}"),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Balance: %{x:+.1f}<extra></extra>"
    ))
    
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
    
    fig.update_layout(
        title="Balance de Relaciones de Fuerza (Ri = Ii - Di)",
        xaxis_title="Balance Neto (+ = Dominante, - = Dominado)",
        yaxis_title="",
        height=500
    )
    
    return fig

def crear_heatmap_matriz(matriz, titulo, colorscale='Blues', show_text=True, zmid=None):
    """Heatmap genérico para matrices"""
    fig = go.Figure(data=go.Heatmap(
        z=matriz.values,
        x=matriz.columns.tolist(),
        y=matriz.index.tolist(),
        colorscale=colorscale,
        zmid=zmid,
        text=matriz.values.round(1) if show_text else None,
        texttemplate="%{text}" if show_text else None,
        hovertemplate='%{y} → %{x}<br>Valor: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(title=titulo, height=600)
    return fig

def crear_grafico_balance_objetivos(balance_obj_df):
    """Gráfico de balance neto ponderado por objetivo"""
    df = balance_obj_df.sort_values('Balance_ponderado', ascending=True).copy()
    
    colores = ['#10B981' if x > 0 else '#EF4444' if x < 0 else '#6B7280' for x in df['Balance_ponderado']]
    
    fig = go.Figure(go.Bar(
        x=df['Balance_ponderado'],
        y=df['Objetivo'],
        orientation='h',
        marker_color=colores,
        text=df['Balance_ponderado'].apply(lambda x: f"{x:+.1f}"),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Balance ponderado: %{x:+.1f}<extra></extra>"
    ))
    
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
    
    fig.update_layout(
        title="Balance Neto Ponderado por Objetivo (Viabilidad)",
        xaxis_title="Balance (+ = Viable, - = Bloqueado)",
        yaxis_title="",
        height=max(400, len(df) * 25)
    )
    
    return fig

def crear_grafico_viabilidad_objetivos(balance_obj_df):
    """Gráfico de índice de viabilidad por objetivo"""
    df = balance_obj_df.sort_values('Viabilidad', ascending=True).copy()
    
    colores = ['#10B981' if x > 0.3 else '#EF4444' if x < -0.3 else '#F59E0B' for x in df['Viabilidad']]
    
    fig = go.Figure(go.Bar(
        x=df['Viabilidad'],
        y=df['Objetivo'],
        orientation='h',
        marker_color=colores,
        text=df['Viabilidad'].apply(lambda x: f"{x:+.2f}"),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Viabilidad: %{x:+.2f}<extra></extra>"
    ))
    
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
    fig.add_vline(x=0.3, line_dash="dash", line_color="green", line_width=1, annotation_text="Alta viabilidad")
    fig.add_vline(x=-0.3, line_dash="dash", line_color="red", line_width=1, annotation_text="Bloqueado")
    
    fig.update_layout(
        title="Índice de Viabilidad por Objetivo (-1 a +1)",
        xaxis_title="Viabilidad (normalizada por movilización)",
        yaxis_title="",
        height=max(400, len(df) * 25)
    )
    
    return fig

def crear_grafico_movilizacion_objetivos(balance_obj_df):
    """Gráfico de movilización de actores sobre objetivos"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='A favor (+)',
        x=balance_obj_df['Objetivo'],
        y=balance_obj_df['A_favor'],
        marker_color='#10B981',
        text=balance_obj_df['A_favor'],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='En contra (-)',
        x=balance_obj_df['Objetivo'],
        y=-balance_obj_df['En_contra'],
        marker_color='#EF4444',
        text=balance_obj_df['En_contra'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Movilización de Actores sobre Objetivos",
        barmode='relative',
        height=500,
        xaxis_title="Objetivo",
        yaxis_title="Número de actores"
    )
    
    return fig

def crear_grafico_implicacion_actores(impl_df):
    """Gráfico de implicación de actores en el sistema"""
    df = impl_df.sort_values('Implicacion_total', ascending=True).copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Intensidad a favor',
        y=df['Actor'],
        x=df['Intensidad_favor'],
        orientation='h',
        marker_color='#10B981'
    ))
    
    fig.add_trace(go.Bar(
        name='Intensidad en contra',
        y=df['Actor'],
        x=df['Intensidad_contra'],
        orientation='h',
        marker_color='#EF4444'
    ))
    
    fig.update_layout(
        title="Implicación de Actores en el Sistema (Intensidad acumulada)",
        barmode='stack',
        height=500,
        xaxis_title="Intensidad total",
        yaxis_title=""
    )
    
    return fig

def crear_grafico_convergencias_actor(CAA, DAA, actores):
    """Gráfico de convergencias vs divergencias por actor"""
    conv_total = CAA.values.sum(axis=1)
    div_total = DAA.values.sum(axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Convergencias',
        x=actores,
        y=conv_total,
        marker_color='#10B981'
    ))
    
    fig.add_trace(go.Bar(
        name='Divergencias',
        x=actores,
        y=div_total,
        marker_color='#EF4444'
    ))
    
    fig.update_layout(
        title='Convergencias vs Divergencias por Actor (Simple)',
        barmode='group',
        height=500,
        xaxis_title='Actor',
        yaxis_title='Número de coincidencias/conflictos'
    )
    
    return fig

def crear_grafico_convergencias_ponderadas(CAA_pond, DAA_pond, actores):
    """Gráfico de convergencias vs divergencias PONDERADAS por actor"""
    conv_total = CAA_pond.values.sum(axis=1)
    div_total = DAA_pond.values.sum(axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Convergencias (intensidad)',
        x=actores,
        y=conv_total,
        marker_color='#059669'
    ))
    
    fig.add_trace(go.Bar(
        name='Divergencias (intensidad)',
        x=actores,
        y=div_total,
        marker_color='#DC2626'
    ))
    
    fig.update_layout(
        title='Convergencias vs Divergencias PONDERADAS por Actor',
        barmode='group',
        height=500,
        xaxis_title='Actor',
        yaxis_title='Intensidad acumulada'
    )
    
    return fig

def crear_matriz_alianzas_conflictos(CAA, DAA, ponderada=False):
    """Matriz visual de alianzas y conflictos"""
    balance = CAA - DAA
    titulo = "Matriz de Alianzas (+) y Conflictos (-) " + ("PONDERADA" if ponderada else "Simple")
    
    fig = go.Figure(data=go.Heatmap(
        z=balance.values,
        x=balance.columns.tolist(),
        y=balance.index.tolist(),
        colorscale='RdYlGn',
        zmid=0,
        text=balance.values.round(1 if ponderada else 0),
        texttemplate="%{text}",
        hovertemplate='%{y} ↔ %{x}<br>Balance: %{z:+.1f}<extra></extra>'
    ))
    
    fig.update_layout(title=titulo, height=600)
    return fig

def crear_red_actores(balance_df, CAA, DAA):
    """Red de actores"""
    actores = balance_df['Actor'].tolist()
    n = len(actores)
    
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos_x = np.cos(angles)
    pos_y = np.sin(angles)
    
    clasificacion, _, _ = clasificar_actores(balance_df)
    colores_clasif = {
        'Dominante': '#DC2626',
        'Enlace': '#7C3AED',
        'Dominado': '#2563EB',
        'Autónomo': '#D97706'
    }
    node_colors = [colores_clasif[c] for c in clasificacion]
    
    Ii = balance_df['Ii'].values
    node_sizes = 20 + (Ii / max(Ii.max(), 1)) * 30
    
    edge_x, edge_y = [], []
    balance_neto = (CAA - DAA).values
    
    for i in range(n):
        for j in range(i+1, n):
            if balance_neto[i, j] > 2:
                edge_x.extend([pos_x[i], pos_x[j], None])
                edge_y.extend([pos_y[i], pos_y[j], None])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='rgba(16, 185, 129, 0.4)'),
        hoverinfo='none',
        name='Alianzas'
    ))
    
    fig.add_trace(go.Scatter(
        x=pos_x, y=pos_y,
        mode='markers+text',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
        text=actores,
        textposition='top center',
        name='Actores'
    ))
    
    fig.update_layout(
        title="Red de Actores (nodos por influencia, líneas = alianzas)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def crear_grafico_radar_actores(balance_df, top_n=6):
    """Radar comparando actores principales"""
    df = balance_df.nlargest(top_n, 'Ii').copy()
    
    df['Ii_norm'] = df['Ii'] / df['Ii'].max()
    df['Di_norm'] = df['Di'] / (df['Di'].max() + 0.001)
    df['Ri_norm'] = (df['Ri'] - df['Ri'].min()) / (df['Ri'].max() - df['Ri'].min() + 0.001)
    
    fig = go.Figure()
    
    for _, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Ii_norm'], row['Di_norm'], row['Ri_norm'], row['Ii_norm']],
            theta=['Influencia', 'Dependencia', 'Ratio Poder', 'Influencia'],
            name=row['Actor'],
            fill='toself',
            opacity=0.6
        ))
    
    fig.update_layout(
        title=f"Perfil Comparativo de Actores Principales (Top {top_n})",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500
    )
    
    return fig

# ============================================================
# GENERACIÓN DE INFORME
# ============================================================

def generar_informe_excel(datos, nombre_proyecto):
    """Genera informe Excel completo"""
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Resumen
        resumen_data = {
            'Parámetro': ['Fecha', 'Proyecto', 'Actores', 'Objetivos', 'K (MIDI)'],
            'Valor': [
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                nombre_proyecto,
                len(datos.get('actores', [])),
                len(datos.get('objetivos', [])),
                datos.get('k', 2)
            ]
        }
        pd.DataFrame(resumen_data).to_excel(writer, sheet_name='Resumen', index=False)
        
        if 'balance' in datos:
            datos['balance'].to_excel(writer, sheet_name='Balance_Actores', index=False)
        
        if 'MAA' in datos and datos['MAA'] is not None:
            datos['MAA'].to_excel(writer, sheet_name='MAA')
        
        if 'MIDI' in datos and datos['MIDI'] is not None:
            datos['MIDI'].to_excel(writer, sheet_name='MIDI')
        
        if 'MAO_2' in datos and datos['MAO_2'] is not None:
            datos['MAO_2'].to_excel(writer, sheet_name='2MAO')
        
        if 'MAO_3' in datos and datos['MAO_3'] is not None:
            datos['MAO_3'].to_excel(writer, sheet_name='3MAO')
        
        if 'CAA' in datos:
            datos['CAA'].to_excel(writer, sheet_name='Convergencias_Simple')
        
        if 'DAA' in datos:
            datos['DAA'].to_excel(writer, sheet_name='Divergencias_Simple')
        
        if 'CAA_pond' in datos:
            datos['CAA_pond'].to_excel(writer, sheet_name='Convergencias_Ponderada')
        
        if 'DAA_pond' in datos:
            datos['DAA_pond'].to_excel(writer, sheet_name='Divergencias_Ponderada')
        
        if 'CAA' in datos and 'DAA' in datos:
            (datos['CAA'] - datos['DAA']).to_excel(writer, sheet_name='Balance_Alianzas_Simple')
        
        if 'CAA_pond' in datos and 'DAA_pond' in datos:
            (datos['CAA_pond'] - datos['DAA_pond']).to_excel(writer, sheet_name='Balance_Alianzas_Pond')
        
        if 'balance_objetivos' in datos:
            datos['balance_objetivos'].to_excel(writer, sheet_name='Balance_Objetivos', index=False)
        
        if 'implicacion_actores' in datos:
            datos['implicacion_actores'].to_excel(writer, sheet_name='Implicacion_Actores', index=False)
    
    buffer.seek(0)
    return buffer

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

st.markdown('<div class="main-header">🎭 MACTOR PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Método de Análisis de Actores - Michel Godet (LIPSOR)</div>', unsafe_allow_html=True)

# Inicializar session state
for key in ['mactor_data', 'mactor_resultados', 'modo_entrada', 'actores_manual', 'objetivos_manual', 'MAA_manual', 'MAO_manual']:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selector de modo
    st.subheader("1. Modo de Entrada")
    modo = st.radio(
        "Seleccionar modo:",
        ["📁 Cargar archivo Excel", "✏️ Entrada manual"],
        index=0,
        help="Elige cómo ingresar los datos"
    )
    st.session_state.modo_entrada = modo
    
    st.divider()
    
    if modo == "📁 Cargar archivo Excel":
        st.subheader("2. Cargar Archivo")
        uploaded_file = st.file_uploader(
            "Subir archivo MACTOR",
            type=['xlsx', 'xls'],
            help="Debe contener hojas MAA y 2MAO"
        )
    else:
        st.subheader("2. Configurar Dimensiones")
        n_actores = st.number_input("Número de actores", min_value=2, max_value=20, value=5)
        n_objetivos = st.number_input("Número de objetivos", min_value=2, max_value=40, value=10)
        
        if st.button("📋 Crear plantillas"):
            st.session_state.actores_manual = [f"Actor_{i+1}" for i in range(n_actores)]
            st.session_state.objetivos_manual = [f"O{i+1}" for i in range(n_objetivos)]
            st.session_state.MAA_manual = pd.DataFrame(
                np.zeros((n_actores, n_actores)),
                index=st.session_state.actores_manual,
                columns=st.session_state.actores_manual
            )
            st.session_state.MAO_manual = pd.DataFrame(
                np.zeros((n_actores, n_objetivos)),
                index=st.session_state.actores_manual,
                columns=st.session_state.objetivos_manual
            )
            st.success("✅ Plantillas creadas")
        
        uploaded_file = None
    
    st.divider()
    
    st.subheader("3. Parámetros MIDI")
    k_midi = st.slider("K (profundidad)", min_value=2, max_value=5, value=2)
    
    st.divider()
    
    st.subheader("4. Visualización")
    mostrar_valores = st.checkbox("Mostrar valores", value=True)
    top_n_radar = st.slider("Actores en radar", min_value=4, max_value=10, value=6)

# Pestañas principales
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Datos",
    "👥 Actores (MIDI)",
    "🎯 Objetivos (MAO)",
    "🤝 Convergencias",
    "🔬 Análisis Avanzado",
    "📊 Síntesis",
    "📥 Exportar"
])

# ============================================================
# TAB 1: DATOS
# ============================================================
with tab1:
    st.header("📋 Carga y Configuración de Datos")
    
    if st.session_state.modo_entrada == "📁 Cargar archivo Excel":
        # MODO ARCHIVO
        if uploaded_file is not None:
            resultado = procesar_archivo_mactor(uploaded_file)
            
            if 'error' in resultado:
                st.error(f"❌ Error: {resultado['error']}")
            else:
                st.session_state.mactor_data = resultado
                
                for msg in resultado.get('mensajes', []):
                    st.success(msg)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("👥 Actores", len(resultado.get('actores', [])))
                col2.metric("🎯 Objetivos", len(resultado.get('objetivos', [])))
                col3.metric("📑 Hojas", len(resultado.get('hojas', [])))
                
                if resultado.get('actores'):
                    st.subheader("👥 Actores del sistema")
                    cols = st.columns(5)
                    for i, actor in enumerate(resultado['actores']):
                        cols[i % 5].markdown(f"**A{i+1}.** {actor}")
                
                if resultado.get('objetivos'):
                    st.subheader("🎯 Objetivos estratégicos")
                    cols = st.columns(10)
                    for i, obj in enumerate(resultado['objetivos']):
                        cols[i % 10].markdown(f"**{obj}**")
                
                st.subheader("📊 Vista previa de matrices")
                col1, col2 = st.columns(2)
                
                with col1:
                    if resultado.get('MAA') is not None:
                        st.markdown("**Matriz MAA (Influencias)**")
                        st.dataframe(resultado['MAA'], use_container_width=True, height=300)
                
                with col2:
                    if resultado.get('MAO_2') is not None:
                        st.markdown("**Matriz 2MAO (Posiciones)**")
                        st.dataframe(resultado['MAO_2'], use_container_width=True, height=300)
        else:
            st.info("👆 Sube un archivo Excel con las matrices MACTOR")
            st.markdown("""
            **Formato esperado:**
            - Hoja con **MAA**: Matriz de Actores × Actores (influencias 0-4)
            - Hoja con **2MAO**: Matriz Actores × Objetivos (posiciones -3 a +3)
            """)
    
    else:
        # MODO MANUAL
        st.markdown("""
        <div class="info-box">
        <b>Modo de entrada manual:</b> Configura los actores y objetivos, luego completa las matrices.
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.actores_manual is not None:
            # Editar nombres de actores
            st.subheader("👥 Configurar Actores")
            actores_editados = []
            cols = st.columns(5)
            for i, actor in enumerate(st.session_state.actores_manual):
                with cols[i % 5]:
                    nuevo_nombre = st.text_input(f"A{i+1}", value=actor, key=f"actor_{i}")
                    actores_editados.append(nuevo_nombre)
            
            st.session_state.actores_manual = actores_editados
            
            # Editar nombres de objetivos
            st.subheader("🎯 Configurar Objetivos")
            objetivos_editados = []
            cols = st.columns(10)
            for i, obj in enumerate(st.session_state.objetivos_manual):
                with cols[i % 10]:
                    nuevo_nombre = st.text_input(f"Obj{i+1}", value=obj, key=f"obj_{i}", label_visibility="collapsed")
                    objetivos_editados.append(nuevo_nombre)
            
            st.session_state.objetivos_manual = objetivos_editados
            
            # Editar MAA
            st.subheader("📊 Matriz MAA (Influencias entre Actores)")
            st.markdown("**Escala:** 0=Sin influencia, 1=Procesos, 2=Proyectos, 3=Misión, 4=Existencia")
            
            MAA_edit = st.session_state.MAA_manual.copy()
            MAA_edit.index = actores_editados
            MAA_edit.columns = actores_editados
            
            edited_MAA = st.data_editor(
                MAA_edit,
                use_container_width=True,
                height=400,
                key="maa_editor"
            )
            st.session_state.MAA_manual = edited_MAA
            
            # Editar 2MAO
            st.subheader("📊 Matriz 2MAO (Posiciones sobre Objetivos)")
            st.markdown("**Escala:** -3=Vital en contra, -2, -1, 0=Neutro, +1, +2, +3=Vital a favor")
            
            MAO_edit = st.session_state.MAO_manual.copy()
            MAO_edit.index = actores_editados
            MAO_edit.columns = objetivos_editados
            
            edited_MAO = st.data_editor(
                MAO_edit,
                use_container_width=True,
                height=400,
                key="mao_editor"
            )
            st.session_state.MAO_manual = edited_MAO
            
            # Guardar datos procesados
            if st.button("✅ Confirmar datos y procesar", type="primary"):
                np.fill_diagonal(edited_MAA.values, 0)
                
                st.session_state.mactor_data = {
                    'MAA': edited_MAA,
                    'MAO_2': edited_MAO,
                    'actores': actores_editados,
                    'objetivos': objetivos_editados,
                    'mensajes': ['✅ Datos manuales procesados correctamente'],
                    'hojas': ['Manual']
                }
                st.success("✅ Datos guardados correctamente. Ve a las siguientes pestañas para ver el análisis.")
        else:
            st.warning("👈 Primero configura las dimensiones y crea las plantillas en el sidebar")

# ============================================================
# TAB 2: ACTORES (MIDI)
# ============================================================
with tab2:
    st.header("👥 Análisis de Relaciones entre Actores")
    
    if st.session_state.mactor_data is not None and st.session_state.mactor_data.get('MAA') is not None:
        data = st.session_state.mactor_data
        MAA = data['MAA']
        actores = data['actores']
        
        # Calcular MIDI
        MIDI = calcular_MIDI(MAA, k=k_midi)
        balance = calcular_balance_MIDI(MIDI)
        clasificacion, med_Ii, med_Di = clasificar_actores(balance)
        balance['Clasificación'] = clasificacion
        
        # Guardar resultados
        if st.session_state.mactor_resultados is None:
            st.session_state.mactor_resultados = {}
        st.session_state.mactor_resultados.update({
            'MIDI': MIDI,
            'balance': balance,
            'med_Ii': med_Ii,
            'med_Di': med_Di,
            'k': k_midi
        })
        
        # Métricas
        st.subheader("📈 Distribución de Actores")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 Dominantes", sum(c == 'Dominante' for c in clasificacion))
        col2.metric("🟣 Enlace", sum(c == 'Enlace' for c in clasificacion))
        col3.metric("🔵 Dominados", sum(c == 'Dominado' for c in clasificacion))
        col4.metric("🟠 Autónomos", sum(c == 'Autónomo' for c in clasificacion))
        
        # Tabla de balance
        st.subheader("📊 Balance de Relaciones de Fuerza")
        
        def color_clasif(val):
            colores = {
                'Dominante': 'background-color: #FEE2E2; color: #DC2626',
                'Enlace': 'background-color: #EDE9FE; color: #7C3AED',
                'Dominado': 'background-color: #DBEAFE; color: #2563EB',
                'Autónomo': 'background-color: #FEF3C7; color: #D97706'
            }
            return colores.get(val, '')
        
        st.dataframe(
            balance.style.applymap(color_clasif, subset=['Clasificación']),
            use_container_width=True, height=400
        )
        
        # Plano de influencias
        st.subheader("🗺️ Plano de Influencias/Dependencias")
        fig_plano = crear_plano_influencias(balance, med_Ii, med_Di)
        mostrar_grafico_con_descargas(fig_plano, "plano_actores", "tab2_plano")
        
        # Balance neto
        st.subheader("📊 Balance Neto (Ii - Di)")
        fig_hist = crear_histograma_balance(balance)
        mostrar_grafico_con_descargas(fig_hist, "balance_neto", "tab2_hist")
        
        # Heatmaps
        st.subheader("🔥 Matrices de Influencias")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_maa = crear_heatmap_matriz(MAA, "Matriz MAA (Influencias Directas)", 'Blues', mostrar_valores)
            st.plotly_chart(fig_maa, use_container_width=True)
        
        with col2:
            fig_midi = crear_heatmap_matriz(MIDI, f"Matriz MIDI (k={k_midi})", 'Purples', mostrar_valores)
            st.plotly_chart(fig_midi, use_container_width=True)
    
    else:
        st.warning("⚠️ Primero carga o ingresa datos en la pestaña 'Datos'")

# ============================================================
# TAB 3: OBJETIVOS (MAO)
# ============================================================
with tab3:
    st.header("🎯 Análisis de Posiciones sobre Objetivos")
    
    data = st.session_state.mactor_data
    resultados = st.session_state.mactor_resultados
    
    if data is not None and data.get('MAO_2') is not None:
        MAO_2 = data['MAO_2']
        
        # Calcular balance de objetivos
        balance_actores = resultados.get('balance') if resultados else None
        balance_obj = calcular_balance_objetivos(MAO_2, balance_actores)
        impl_actores = calcular_implicacion_actores(MAO_2)
        
        # Guardar
        if st.session_state.mactor_resultados is None:
            st.session_state.mactor_resultados = {}
        st.session_state.mactor_resultados['balance_objetivos'] = balance_obj
        st.session_state.mactor_resultados['implicacion_actores'] = impl_actores
        
        # Calcular 3MAO
        if resultados and 'MIDI' in resultados:
            MAO_3 = calcular_3MAO(MAO_2, resultados['MIDI'])
            if MAO_3 is not None:
                st.session_state.mactor_resultados['MAO_3'] = MAO_3
        
        # Métricas
        st.subheader("📈 Resumen de Objetivos")
        col1, col2, col3 = st.columns(3)
        
        idx_max = balance_obj['Balance_ponderado'].idxmax()
        idx_min = balance_obj['Balance_ponderado'].idxmin()
        idx_movil = balance_obj['Movilizacion'].idxmax()
        
        col1.metric("✅ Más viable", balance_obj.loc[idx_max, 'Objetivo'])
        col2.metric("❌ Más bloqueado", balance_obj.loc[idx_min, 'Objetivo'])
        col3.metric("⚡ Más movilizador", balance_obj.loc[idx_movil, 'Objetivo'])
        
        # Balance ponderado por objetivo
        st.subheader("📊 Balance Neto Ponderado por Objetivo")
        st.markdown("*Ponderado por el coeficiente de poder de cada actor (Ri)*")
        fig_balance = crear_grafico_balance_objetivos(balance_obj)
        mostrar_grafico_con_descargas(fig_balance, "balance_objetivos", "tab3_balance")
        
        # Viabilidad
        st.subheader("📊 Índice de Viabilidad por Objetivo")
        st.markdown("*Viabilidad = Balance ponderado / Movilización total (-1 a +1)*")
        fig_viab = crear_grafico_viabilidad_objetivos(balance_obj)
        mostrar_grafico_con_descargas(fig_viab, "viabilidad_objetivos", "tab3_viab")
        
        # Movilización
        st.subheader("📊 Movilización de Actores")
        fig_mob = crear_grafico_movilizacion_objetivos(balance_obj)
        mostrar_grafico_con_descargas(fig_mob, "movilizacion", "tab3_mob")
        
        # Implicación de actores
        st.subheader("📊 Implicación de Actores en el Sistema")
        fig_impl = crear_grafico_implicacion_actores(impl_actores)
        mostrar_grafico_con_descargas(fig_impl, "implicacion_actores", "tab3_impl")
        
        # Heatmap 2MAO
        st.subheader("🔥 Matriz 2MAO Completa")
        fig_2mao = crear_heatmap_matriz(MAO_2, "Posiciones Actores × Objetivos (2MAO)", 'RdYlGn', mostrar_valores, zmid=0)
        mostrar_grafico_con_descargas(fig_2mao, "matriz_2mao", "tab3_2mao")
        
        # Tabla resumen
        st.subheader("📋 Tabla de Balance por Objetivo")
        st.dataframe(balance_obj, use_container_width=True)
    
    else:
        st.warning("⚠️ Primero carga o ingresa datos con matriz 2MAO")

# ============================================================
# TAB 4: CONVERGENCIAS
# ============================================================
with tab4:
    st.header("🤝 Análisis de Convergencias y Divergencias")
    
    data = st.session_state.mactor_data
    
    if data is not None and data.get('MAO_2') is not None:
        MAO_2 = data['MAO_2']
        actores = data['actores']
        
        # Calcular convergencias y divergencias
        CAA, DAA, CAA_pond, DAA_pond = calcular_convergencias_divergencias(MAO_2)
        
        # Guardar
        st.session_state.mactor_resultados['CAA'] = CAA
        st.session_state.mactor_resultados['DAA'] = DAA
        st.session_state.mactor_resultados['CAA_pond'] = CAA_pond
        st.session_state.mactor_resultados['DAA_pond'] = DAA_pond
        
        # Métricas
        st.subheader("📈 Resumen de Relaciones")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Convergencias (simple)", int(CAA.values.sum() / 2))
        col2.metric("Divergencias (simple)", int(DAA.values.sum() / 2))
        col3.metric("Conv. ponderadas", f"{CAA_pond.values.sum() / 2:.0f}")
        col4.metric("Div. ponderadas", f"{DAA_pond.values.sum() / 2:.0f}")
        
        # Gráfico simple
        st.subheader("📊 Convergencias vs Divergencias (Simple)")
        fig_simple = crear_grafico_convergencias_actor(CAA, DAA, actores)
        mostrar_grafico_con_descargas(fig_simple, "conv_div_simple", "tab4_simple")
        
        # Gráfico ponderado
        st.subheader("📊 Convergencias vs Divergencias (PONDERADAS)")
        st.markdown("*Las matrices ponderadas consideran la intensidad de las posiciones, no solo el conteo*")
        fig_pond = crear_grafico_convergencias_ponderadas(CAA_pond, DAA_pond, actores)
        mostrar_grafico_con_descargas(fig_pond, "conv_div_ponderada", "tab4_pond")
        
        # Matriz de alianzas simple
        st.subheader("🔥 Matriz de Alianzas y Conflictos (Simple)")
        fig_alianzas = crear_matriz_alianzas_conflictos(CAA, DAA, ponderada=False)
        mostrar_grafico_con_descargas(fig_alianzas, "alianzas_simple", "tab4_alianzas")
        
        # Matriz de alianzas ponderada
        st.subheader("🔥 Matriz de Alianzas y Conflictos (PONDERADA)")
        st.markdown("*Muestra la intensidad de las alianzas/conflictos, no solo su existencia*")
        fig_alianzas_pond = crear_matriz_alianzas_conflictos(CAA_pond, DAA_pond, ponderada=True)
        mostrar_grafico_con_descargas(fig_alianzas_pond, "alianzas_ponderada", "tab4_alianzas_pond")
        
        # Heatmaps detallados
        st.subheader("🔥 Matrices de Convergencias/Divergencias Ponderadas")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_caa = crear_heatmap_matriz(CAA_pond, "Convergencias Ponderadas", 'Greens', mostrar_valores)
            st.plotly_chart(fig_caa, use_container_width=True)
        
        with col2:
            fig_daa = crear_heatmap_matriz(DAA_pond, "Divergencias Ponderadas", 'Reds', mostrar_valores)
            st.plotly_chart(fig_daa, use_container_width=True)
        
        # Ambivalencia
        st.subheader("🔄 Índice de Ambivalencia")
        amb = calcular_ambivalencia(CAA, DAA)
        fig_amb = crear_heatmap_matriz(amb, "Índice de Ambivalencia", 'YlOrRd', mostrar_valores)
        mostrar_grafico_con_descargas(fig_amb, "ambivalencia", "tab4_amb")
    
    else:
        st.warning("⚠️ Primero carga o ingresa datos con matriz 2MAO")

# ============================================================
# TAB 5: ANÁLISIS AVANZADO
# ============================================================
with tab5:
    st.header("🔬 Análisis Avanzado")
    
    data = st.session_state.mactor_data
    resultados = st.session_state.mactor_resultados
    
    if data is not None and resultados is not None:
        
        # Red de actores
        if 'balance' in resultados and 'CAA' in resultados:
            st.subheader("🕸️ Red de Actores")
            fig_red = crear_red_actores(resultados['balance'], resultados['CAA'], resultados['DAA'])
            mostrar_grafico_con_descargas(fig_red, "red_actores", "tab5_red")
        
        # Radar
        if 'balance' in resultados:
            st.subheader("📡 Perfil Comparativo de Actores")
            fig_radar = crear_grafico_radar_actores(resultados['balance'], top_n=top_n_radar)
            mostrar_grafico_con_descargas(fig_radar, "radar_actores", "tab5_radar")
        
        # 3MAO
        if 'MAO_3' in resultados and resultados['MAO_3'] is not None:
            st.subheader("🔥 Matriz 3MAO (Posiciones Valoradas por Poder)")
            fig_3mao = crear_heatmap_matriz(resultados['MAO_3'], "Matriz 3MAO", 'RdYlGn', mostrar_valores, zmid=0)
            mostrar_grafico_con_descargas(fig_3mao, "matriz_3mao", "tab5_3mao")
        
        # Clusters
        st.subheader("🎯 Identificación de Alianzas y Conflictos Clave")
        
        if 'CAA_pond' in resultados and 'DAA_pond' in resultados:
            balance_matriz = resultados['CAA_pond'] - resultados['DAA_pond']
            actores = data['actores']
            
            pares_alianza = []
            pares_conflicto = []
            
            for i in range(len(actores)):
                for j in range(i+1, len(actores)):
                    val = balance_matriz.iloc[i, j]
                    if val > 2:
                        pares_alianza.append((actores[i], actores[j], val))
                    elif val < -1:
                        pares_conflicto.append((actores[i], actores[j], val))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🟢 Alianzas más fuertes (ponderadas):**")
                pares_alianza.sort(key=lambda x: x[2], reverse=True)
                for a1, a2, v in pares_alianza[:7]:
                    st.markdown(f"- {a1} ↔ {a2}: **+{v:.1f}**")
            
            with col2:
                st.markdown("**🔴 Conflictos más fuertes (ponderados):**")
                pares_conflicto.sort(key=lambda x: x[2])
                for a1, a2, v in pares_conflicto[:7]:
                    st.markdown(f"- {a1} ↔ {a2}: **{v:.1f}**")
    
    else:
        st.warning("⚠️ Ejecuta primero el análisis en las pestañas anteriores")

# ============================================================
# TAB 6: SÍNTESIS
# ============================================================
with tab6:
    st.header("📊 Síntesis Estratégica")
    
    data = st.session_state.mactor_data
    resultados = st.session_state.mactor_resultados
    
    if data is not None and resultados is not None:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Actores Clave")
            
            if 'balance' in resultados:
                balance = resultados['balance']
                
                st.markdown("**🔴 Top 3 Dominantes:**")
                for _, row in balance.nlargest(3, 'Ri_neto').iterrows():
                    st.markdown(f"- **{row['Actor']}**: Balance=**+{row['Ri_neto']:.0f}**")
                
                st.markdown("**🔵 Top 3 Dominados:**")
                for _, row in balance.nsmallest(3, 'Ri_neto').iterrows():
                    st.markdown(f"- **{row['Actor']}**: Balance=**{row['Ri_neto']:.0f}**")
                
                st.markdown("**🟣 Actores Enlace:**")
                enlace = balance[balance['Clasificación'] == 'Enlace']
                for _, row in enlace.iterrows():
                    st.markdown(f"- **{row['Actor']}**")
        
        with col2:
            st.subheader("🎯 Objetivos Clave")
            
            if 'balance_objetivos' in resultados:
                bal_obj = resultados['balance_objetivos']
                
                st.markdown("**✅ Objetivos más viables:**")
                for _, row in bal_obj.nlargest(3, 'Viabilidad').iterrows():
                    st.markdown(f"- **{row['Objetivo']}**: Viabilidad=**{row['Viabilidad']:+.2f}**")
                
                st.markdown("**❌ Objetivos bloqueados:**")
                for _, row in bal_obj.nsmallest(3, 'Viabilidad').iterrows():
                    st.markdown(f"- **{row['Objetivo']}**: Viabilidad=**{row['Viabilidad']:+.2f}**")
                
                st.markdown("**⚡ Mayor movilización:**")
                for _, row in bal_obj.nlargest(3, 'Movilizacion').iterrows():
                    st.markdown(f"- **{row['Objetivo']}**: Movilización=**{row['Movilizacion']:.0f}**")
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones Estratégicas")
        
        st.markdown("""
        <div class="info-box">
        <h4>Basado en el análisis MACTOR:</h4>
        
        <b>1. Actores Dominantes:</b> Negociar primero con ellos - determinan el éxito.<br>
        <b>2. Actores Enlace:</b> Útiles como mediadores entre grupos.<br>
        <b>3. Objetivos Viables:</b> Priorizar para construir coaliciones.<br>
        <b>4. Objetivos Bloqueados:</b> Requieren reformulación o negociación.<br>
        <b>5. Alianzas Ponderadas:</b> Los pares con alta convergencia ponderada son aliados naturales con compromisos fuertes.
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Ejecuta primero el análisis completo")

# ============================================================
# TAB 7: EXPORTAR
# ============================================================
with tab7:
    st.header("📥 Exportar Resultados")
    
    data = st.session_state.mactor_data
    resultados = st.session_state.mactor_resultados
    
    if data is not None and resultados is not None:
        nombre_proyecto = st.text_input("Nombre del proyecto", value="Analisis_MACTOR")
        
        datos_export = {
            'actores': data.get('actores'),
            'objetivos': data.get('objetivos'),
            'MAA': data.get('MAA'),
            'MAO_2': data.get('MAO_2'),
            'k': k_midi
        }
        datos_export.update(resultados)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Informe Excel Completo")
            st.markdown("""
            **Incluye:**
            - Balance de actores
            - Matrices MAA, MIDI, 2MAO, 3MAO
            - Convergencias/Divergencias (simple y ponderada)
            - Balance de alianzas
            - Balance de objetivos
            - Implicación de actores
            """)
            
            buffer = generar_informe_excel(datos_export, nombre_proyecto)
            st.download_button(
                label="📥 Descargar Informe Excel",
                data=buffer,
                file_name=f"{nombre_proyecto}_MACTOR.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        
        with col2:
            st.subheader("📋 Exportaciones CSV")
            
            if 'balance' in resultados:
                st.download_button("📥 Balance Actores", resultados['balance'].to_csv(index=False), "balance_actores.csv", "text/csv")
            
            if 'CAA_pond' in resultados:
                st.download_button("📥 Convergencias Pond.", resultados['CAA_pond'].to_csv(), "convergencias_pond.csv", "text/csv")
            
            if 'DAA_pond' in resultados:
                st.download_button("📥 Divergencias Pond.", resultados['DAA_pond'].to_csv(), "divergencias_pond.csv", "text/csv")
            
            if 'balance_objetivos' in resultados:
                st.download_button("📥 Balance Objetivos", resultados['balance_objetivos'].to_csv(index=False), "balance_objetivos.csv", "text/csv")
    
    else:
        st.warning("⚠️ Primero carga y procesa los datos")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<b>MACTOR PRO v4.0</b> | Metodología Michel Godet (LIPSOR) | JETLEX Strategic Consulting | M. Pratto Chiarella 2025
</div>
""", unsafe_allow_html=True)
