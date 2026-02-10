"""
MACTOR PRO v4.1 - Método de Análisis de Actores
Matriz de Alianzas y Conflictos: Tácticas, Objetivos y Recomendaciones

Autor: JETLEX Strategic Consulting / Martín Pratto Chiarella
Basado en el método de Michel Godet (LIPSOR)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="MACTOR PRO v4.1",
    page_icon="🎭",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #8B5CF6; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem;}
    .info-box {background-color: #EDE9FE; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #8B5CF6; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================

def procesar_archivo_mactor(uploaded_file):
    """Procesa archivo Excel MACTOR"""
    try:
        xl = pd.ExcelFile(uploaded_file)
        hojas = xl.sheet_names
        
        resultado = {
            'hojas': hojas, 'MAA': None, 'MAO_2': None,
            'actores': None, 'objetivos': None, 'mensajes': []
        }
        
        # Buscar MAA
        for hoja in hojas:
            if 'MAA' in hoja.upper() and 'JUST' not in hoja.upper():
                df = pd.read_excel(xl, sheet_name=hoja, header=None)
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
                
                actores, filas_datos = [], []
                for i in range(fila_inicio, min(fila_inicio + 25, len(df))):
                    nombre = df.iloc[i, 0]
                    if pd.isna(nombre): continue
                    nombre_str = str(nombre).strip()
                    if nombre_str and not any(x in nombre_str.lower() for x in ['suma', 'total', 'ii', 'di']):
                        actores.append(nombre_str)
                        filas_datos.append(i)
                
                n = len(actores)
                matriz = np.zeros((n, n))
                for i, fila_idx in enumerate(filas_datos):
                    for j in range(n):
                        val = df.iloc[fila_idx, j + 1]
                        if pd.notna(val):
                            try: matriz[i, j] = float(val)
                            except: pass
                
                np.fill_diagonal(matriz, 0)
                resultado['MAA'] = pd.DataFrame(matriz, index=actores, columns=actores)
                resultado['actores'] = actores
                resultado['mensajes'].append(f"✅ MAA {n}×{n} procesada")
                break
        
        # Buscar 2MAO
        for hoja in hojas:
            if '2MAO' in hoja.upper() and 'JUST' not in hoja.upper():
                df = pd.read_excel(xl, sheet_name=hoja, header=None)
                fila_headers = 0
                for i in range(min(10, len(df))):
                    val = str(df.iloc[i, 0]).lower() if pd.notna(df.iloc[i, 0]) else ""
                    if 'actor' in val or '\\' in val:
                        fila_headers = i
                        break
                
                objetivos = []
                for j in range(1, min(40, len(df.columns))):
                    val = df.iloc[fila_headers, j]
                    if pd.notna(val):
                        obj_str = str(val).strip()
                        if obj_str.upper().startswith('O'):
                            objetivos.append(obj_str.upper())
                
                actores_mao, filas_datos = [], []
                for i in range(fila_headers + 1, min(fila_headers + 26, len(df))):
                    nombre = df.iloc[i, 0]
                    if pd.isna(nombre): continue
                    nombre_str = str(nombre).strip()
                    if nombre_str and not any(x in nombre_str.lower() for x in ['suma', 'total', 'moviliz']):
                        actores_mao.append(nombre_str)
                        filas_datos.append(i)
                
                n_actores, n_objetivos = len(actores_mao), len(objetivos)
                matriz = np.zeros((n_actores, n_objetivos))
                for i, fila_idx in enumerate(filas_datos):
                    for j in range(n_objetivos):
                        val = df.iloc[fila_idx, j + 1]
                        if pd.notna(val):
                            try: matriz[i, j] = float(val)
                            except: pass
                
                resultado['MAO_2'] = pd.DataFrame(matriz, index=actores_mao, columns=objetivos)
                resultado['objetivos'] = objetivos
                if resultado['actores'] is None:
                    resultado['actores'] = actores_mao
                resultado['mensajes'].append(f"✅ 2MAO {n_actores}×{n_objetivos} procesada")
                break
        
        return resultado
    except Exception as e:
        return {'error': str(e)}

# ============================================================
# FUNCIONES DE CÁLCULO MACTOR
# ============================================================

def calcular_MIDI(MAA, k=2):
    """Calcula MIDI"""
    M = MAA.values.astype(float).copy()
    MIDI = M.copy()
    M_power = M.copy()
    for i in range(2, k + 1):
        M_power = np.dot(M_power, M)
        MIDI += M_power
    np.fill_diagonal(MIDI, 0)
    return pd.DataFrame(MIDI, index=MAA.index, columns=MAA.columns)

def calcular_balance_MIDI(MIDI):
    """Calcula balance de fuerza"""
    M = MIDI.values
    Ii = M.sum(axis=1)
    Di = M.sum(axis=0)
    return pd.DataFrame({
        'Actor': MIDI.index.tolist(),
        'Ii': np.round(Ii, 1),
        'Di': np.round(Di, 1),
        'Ri': np.round(Ii / (Di + 0.001), 2),
        'Ri_neto': np.round(Ii - Di, 1)
    })

def clasificar_actores(balance_df):
    """Clasifica actores"""
    Ii, Di = balance_df['Ii'].values, balance_df['Di'].values
    med_Ii, med_Di = np.median(Ii), np.median(Di)
    clasificacion = []
    for i, d in zip(Ii, Di):
        if i >= med_Ii and d < med_Di: clasificacion.append("Dominante")
        elif i >= med_Ii and d >= med_Di: clasificacion.append("Enlace")
        elif i < med_Ii and d >= med_Di: clasificacion.append("Dominado")
        else: clasificacion.append("Autónomo")
    return clasificacion, med_Ii, med_Di

def calcular_convergencias_divergencias(MAO):
    """Calcula CAA y DAA"""
    M = MAO.values.copy()
    n = M.shape[0]
    actores = MAO.index.tolist()
    CAA, DAA = np.zeros((n, n)), np.zeros((n, n))
    CAA_pond, DAA_pond = np.zeros((n, n)), np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
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
    """Calcula 3MAO"""
    actores_comunes = [a for a in MAO_2.index if a in MIDI.index]
    if len(actores_comunes) == 0: return None
    
    MAO = MAO_2.loc[actores_comunes].values.copy()
    M = MIDI.loc[actores_comunes, actores_comunes].values.copy()
    Ri = M.sum(axis=1) - M.sum(axis=0)
    
    if Ri.max() != Ri.min():
        Ri_norm = 0.5 + (Ri - Ri.min()) / (Ri.max() - Ri.min())
    else:
        Ri_norm = np.ones_like(Ri)
    
    return pd.DataFrame(MAO * Ri_norm.reshape(-1, 1), index=actores_comunes, columns=MAO_2.columns)

def calcular_balance_objetivos(MAO, balance_actores=None):
    """Calcula balance por objetivo - FUNCIÓN CRÍTICA"""
    try:
        # Extraer datos
        if hasattr(MAO, 'values'):
            M = MAO.values.copy()
            objetivos = [str(c) for c in MAO.columns.tolist()]
        else:
            M = np.array(MAO).copy()
            objetivos = [f"O{i+1}" for i in range(M.shape[1])]
        
        # Coeficiente de poder
        if (balance_actores is not None and 
            isinstance(balance_actores, pd.DataFrame) and 
            'Ri_neto' in balance_actores.columns):
            Ri_neto = balance_actores['Ri_neto'].values
            if Ri_neto.max() != Ri_neto.min():
                coef_poder = 0.5 + (Ri_neto - Ri_neto.min()) / (Ri_neto.max() - Ri_neto.min())
            else:
                coef_poder = np.ones(M.shape[0])
        else:
            coef_poder = np.ones(M.shape[0])
        
        # Calcular para cada objetivo
        resultados = []
        for j, obj in enumerate(objetivos):
            pos = M[:, j]
            n_favor = int((pos > 0).sum())
            n_contra = int((pos < 0).sum())
            n_neutro = int((pos == 0).sum())
            suma_favor = float(np.where(pos > 0, pos, 0).sum())
            suma_contra = float(np.where(pos < 0, pos, 0).sum())
            balance_pond = float(np.sum(pos * coef_poder[:len(pos)]))
            movilizacion = float(np.abs(pos).sum())
            viabilidad = balance_pond / movilizacion if movilizacion > 0 else 0.0
            
            resultados.append({
                'Objetivo': obj,
                'A_favor': n_favor,
                'Neutros': n_neutro,
                'En_contra': n_contra,
                'Suma_favor': suma_favor,
                'Suma_contra': suma_contra,
                'Balance_simple': suma_favor + suma_contra,
                'Balance_ponderado': round(balance_pond, 2),
                'Movilizacion': movilizacion,
                'Viabilidad': round(viabilidad, 2)
            })
        
        return pd.DataFrame(resultados)
    except Exception as e:
        st.error(f"Error en calcular_balance_objetivos: {e}")
        return pd.DataFrame()

def calcular_implicacion_actores(MAO):
    """Calcula implicación de actores"""
    try:
        M = MAO.values.copy()
        actores = [str(a) for a in MAO.index.tolist()]
        
        resultados = []
        for i, actor in enumerate(actores):
            pos = M[i, :]
            resultados.append({
                'Actor': actor,
                'Obj_favor': int((pos > 0).sum()),
                'Obj_neutro': int((pos == 0).sum()),
                'Obj_contra': int((pos < 0).sum()),
                'Intensidad_favor': float(np.where(pos > 0, pos, 0).sum()),
                'Intensidad_contra': float(np.abs(np.where(pos < 0, pos, 0).sum())),
                'Implicacion_total': float(np.abs(pos).sum())
            })
        
        return pd.DataFrame(resultados)
    except Exception as e:
        st.error(f"Error en calcular_implicacion_actores: {e}")
        return pd.DataFrame()

def calcular_ambivalencia(CAA, DAA):
    """Calcula ambivalencia"""
    C, D = CAA.values.copy(), DAA.values.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        amb = np.minimum(C, D) / (np.maximum(C, D) + 0.001)
        amb = np.nan_to_num(amb, nan=0.0)
    return pd.DataFrame(amb, index=CAA.index, columns=CAA.columns)

# ============================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================

def mostrar_grafico_con_descargas(fig, nombre, key):
    """Muestra gráfico con opciones de descarga"""
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key}")
    col1, col2 = st.columns(2)
    with col1:
        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        st.download_button("📥 HTML", html.encode(), f"{nombre}.html", "text/html", key=f"h_{key}")
    with col2:
        try:
            img = fig.to_image(format="png", width=1200, height=800, scale=2)
            st.download_button("📥 PNG", img, f"{nombre}.png", "image/png", key=f"p_{key}")
        except:
            st.info("📷 Usa el ícono de cámara")

def crear_plano_influencias(balance_df, med_Ii, med_Di):
    """Plano de influencias/dependencias"""
    clasificacion, _, _ = clasificar_actores(balance_df)
    df = balance_df.copy()
    df['Clasificación'] = clasificacion
    
    colores = {'Dominante': '#DC2626', 'Enlace': '#7C3AED', 'Dominado': '#2563EB', 'Autónomo': '#D97706'}
    fig = go.Figure()
    
    for clasif in colores.keys():
        df_temp = df[df['Clasificación'] == clasif]
        if len(df_temp) > 0:
            fig.add_trace(go.Scatter(
                x=df_temp['Di'], y=df_temp['Ii'],
                mode='markers+text', name=clasif,
                text=df_temp['Actor'], textposition='top center',
                marker=dict(size=16, color=colores[clasif], line=dict(width=2, color='white'))
            ))
    
    fig.add_hline(y=med_Ii, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=med_Di, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title="Plano de Influencias/Dependencias", xaxis_title="Dependencia (Di)", yaxis_title="Influencia (Ii)", height=600)
    return fig

def crear_histograma_balance(balance_df):
    """Histograma de balance"""
    df = balance_df.sort_values('Ri_neto', ascending=True).copy()
    colores = ['#DC2626' if x > 0 else '#2563EB' for x in df['Ri_neto']]
    
    fig = go.Figure(go.Bar(
        x=df['Ri_neto'], y=df['Actor'], orientation='h',
        marker_color=colores, text=df['Ri_neto'].apply(lambda x: f"{x:+.0f}"), textposition='outside'
    ))
    fig.add_vline(x=0, line_color="black", line_width=2)
    fig.update_layout(title="Balance de Relaciones de Fuerza (Ii - Di)", height=500)
    return fig

def crear_heatmap_matriz(matriz, titulo, colorscale='Blues', show_text=True, zmid=None):
    """Heatmap genérico"""
    fig = go.Figure(data=go.Heatmap(
        z=matriz.values, x=matriz.columns.tolist(), y=matriz.index.tolist(),
        colorscale=colorscale, zmid=zmid,
        text=matriz.values.round(1) if show_text else None,
        texttemplate="%{text}" if show_text else None
    ))
    fig.update_layout(title=titulo, height=600)
    return fig

def crear_grafico_balance_objetivos(balance_obj_df):
    """Gráfico de balance por objetivo - CON VERIFICACIONES"""
    # VERIFICACIONES CRÍTICAS
    if balance_obj_df is None:
        fig = go.Figure()
        fig.add_annotation(text="Error: datos nulos", showarrow=False, font=dict(size=20))
        return fig
    
    if not isinstance(balance_obj_df, pd.DataFrame):
        fig = go.Figure()
        fig.add_annotation(text=f"Error: tipo inválido ({type(balance_obj_df)})", showarrow=False, font=dict(size=20))
        return fig
    
    if len(balance_obj_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos para mostrar", showarrow=False, font=dict(size=20))
        return fig
    
    if 'Balance_ponderado' not in balance_obj_df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Columnas: {list(balance_obj_df.columns)}", showarrow=False, font=dict(size=14))
        return fig
    
    if 'Objetivo' not in balance_obj_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Falta columna 'Objetivo'", showarrow=False, font=dict(size=20))
        return fig
    
    # CREAR GRÁFICO
    df = balance_obj_df.sort_values('Balance_ponderado', ascending=True).copy()
    colores = ['#10B981' if x > 0 else '#EF4444' if x < 0 else '#6B7280' for x in df['Balance_ponderado']]
    
    fig = go.Figure(go.Bar(
        x=df['Balance_ponderado'], y=df['Objetivo'], orientation='h',
        marker_color=colores, text=df['Balance_ponderado'].apply(lambda x: f"{x:+.1f}"), textposition='outside'
    ))
    fig.add_vline(x=0, line_color="black", line_width=2)
    fig.update_layout(title="Balance Neto Ponderado por Objetivo", xaxis_title="Balance", height=max(400, len(df) * 25))
    return fig

def crear_grafico_viabilidad_objetivos(balance_obj_df):
    """Gráfico de viabilidad - CON VERIFICACIONES"""
    if balance_obj_df is None or not isinstance(balance_obj_df, pd.DataFrame) or len(balance_obj_df) == 0:
        return go.Figure().add_annotation(text="Sin datos", showarrow=False)
    
    if 'Viabilidad' not in balance_obj_df.columns or 'Objetivo' not in balance_obj_df.columns:
        return go.Figure().add_annotation(text="Columnas faltantes", showarrow=False)
    
    df = balance_obj_df.sort_values('Viabilidad', ascending=True).copy()
    colores = ['#10B981' if x > 0.3 else '#EF4444' if x < -0.3 else '#F59E0B' for x in df['Viabilidad']]
    
    fig = go.Figure(go.Bar(
        x=df['Viabilidad'], y=df['Objetivo'], orientation='h',
        marker_color=colores, text=df['Viabilidad'].apply(lambda x: f"{x:+.2f}"), textposition='outside'
    ))
    fig.add_vline(x=0, line_color="black", line_width=2)
    fig.update_layout(title="Índice de Viabilidad por Objetivo", height=max(400, len(df) * 25))
    return fig

def crear_grafico_movilizacion_objetivos(balance_obj_df):
    """Gráfico de movilización - CON VERIFICACIONES"""
    if balance_obj_df is None or not isinstance(balance_obj_df, pd.DataFrame) or len(balance_obj_df) == 0:
        return go.Figure().add_annotation(text="Sin datos", showarrow=False)
    
    required = ['Objetivo', 'A_favor', 'En_contra']
    if not all(c in balance_obj_df.columns for c in required):
        return go.Figure().add_annotation(text="Columnas faltantes", showarrow=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='A favor', x=balance_obj_df['Objetivo'], y=balance_obj_df['A_favor'], marker_color='#10B981'))
    fig.add_trace(go.Bar(name='En contra', x=balance_obj_df['Objetivo'], y=-balance_obj_df['En_contra'], marker_color='#EF4444'))
    fig.update_layout(title="Movilización de Actores", barmode='relative', height=500)
    return fig

def crear_grafico_implicacion_actores(impl_df):
    """Gráfico de implicación - CON VERIFICACIONES"""
    if impl_df is None or not isinstance(impl_df, pd.DataFrame) or len(impl_df) == 0:
        return go.Figure().add_annotation(text="Sin datos", showarrow=False)
    
    required = ['Actor', 'Intensidad_favor', 'Intensidad_contra', 'Implicacion_total']
    if not all(c in impl_df.columns for c in required):
        return go.Figure().add_annotation(text="Columnas faltantes", showarrow=False)
    
    df = impl_df.sort_values('Implicacion_total', ascending=True).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(name='A favor', y=df['Actor'], x=df['Intensidad_favor'], orientation='h', marker_color='#10B981'))
    fig.add_trace(go.Bar(name='En contra', y=df['Actor'], x=df['Intensidad_contra'], orientation='h', marker_color='#EF4444'))
    fig.update_layout(title="Implicación de Actores", barmode='stack', height=500)
    return fig

def crear_grafico_convergencias_actor(CAA, DAA, actores):
    """Convergencias vs divergencias por actor"""
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Convergencias', x=actores, y=CAA.values.sum(axis=1), marker_color='#10B981'))
    fig.add_trace(go.Bar(name='Divergencias', x=actores, y=DAA.values.sum(axis=1), marker_color='#EF4444'))
    fig.update_layout(title='Convergencias vs Divergencias (Simple)', barmode='group', height=500)
    return fig

def crear_grafico_convergencias_ponderadas(CAA_pond, DAA_pond, actores):
    """Convergencias vs divergencias ponderadas"""
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Conv. pond.', x=actores, y=CAA_pond.values.sum(axis=1), marker_color='#059669'))
    fig.add_trace(go.Bar(name='Div. pond.', x=actores, y=DAA_pond.values.sum(axis=1), marker_color='#DC2626'))
    fig.update_layout(title='Convergencias vs Divergencias PONDERADAS', barmode='group', height=500)
    return fig

def crear_matriz_alianzas_conflictos(CAA, DAA, ponderada=False):
    """Matriz de alianzas/conflictos"""
    balance = CAA - DAA
    titulo = "Matriz de Alianzas/Conflictos " + ("PONDERADA" if ponderada else "Simple")
    fig = go.Figure(data=go.Heatmap(
        z=balance.values, x=balance.columns.tolist(), y=balance.index.tolist(),
        colorscale='RdYlGn', zmid=0, text=balance.values.round(1), texttemplate="%{text}"
    ))
    fig.update_layout(title=titulo, height=600)
    return fig

def crear_red_actores(balance_df, CAA, DAA):
    """Red de actores"""
    actores = balance_df['Actor'].tolist()
    n = len(actores)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos_x, pos_y = np.cos(angles), np.sin(angles)
    
    clasificacion, _, _ = clasificar_actores(balance_df)
    colores = {'Dominante': '#DC2626', 'Enlace': '#7C3AED', 'Dominado': '#2563EB', 'Autónomo': '#D97706'}
    node_colors = [colores[c] for c in clasificacion]
    
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
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='rgba(16,185,129,0.4)'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=pos_x, y=pos_y, mode='markers+text', marker=dict(size=node_sizes, color=node_colors), text=actores, textposition='top center'))
    fig.update_layout(title="Red de Actores", showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=600)
    return fig

def crear_grafico_radar_actores(balance_df, top_n=6):
    """Radar de actores"""
    if 'Ii' not in balance_df.columns or len(balance_df) == 0:
        return go.Figure().add_annotation(text="Sin datos", showarrow=False)
    
    df = balance_df.nlargest(top_n, 'Ii').copy()
    df['Ii_norm'] = df['Ii'] / (df['Ii'].max() + 0.001)
    df['Di_norm'] = df['Di'] / (df['Di'].max() + 0.001)
    df['Ri_norm'] = (df['Ri'] - df['Ri'].min()) / (df['Ri'].max() - df['Ri'].min() + 0.001)
    
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Ii_norm'], row['Di_norm'], row['Ri_norm'], row['Ii_norm']],
            theta=['Influencia', 'Dependencia', 'Ratio', 'Influencia'],
            name=row['Actor'], fill='toself', opacity=0.6
        ))
    fig.update_layout(title=f"Radar Top {top_n} Actores", polar=dict(radialaxis=dict(range=[0, 1])), height=500)
    return fig

# ============================================================
# GENERACIÓN DE INFORME
# ============================================================

def generar_informe_excel(datos, nombre):
    """Genera Excel completo"""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Resumen
        pd.DataFrame({'Parámetro': ['Fecha', 'Proyecto'], 'Valor': [datetime.now().strftime('%Y-%m-%d'), nombre]}).to_excel(writer, sheet_name='Resumen', index=False)
        
        for key, sheet in [('balance', 'Balance_Actores'), ('MAA', 'MAA'), ('MIDI', 'MIDI'), ('MAO_2', '2MAO'), ('MAO_3', '3MAO'),
                          ('CAA', 'Convergencias'), ('DAA', 'Divergencias'), ('CAA_pond', 'Conv_Pond'), ('DAA_pond', 'Div_Pond'),
                          ('balance_objetivos', 'Balance_Obj'), ('implicacion_actores', 'Implicacion')]:
            if key in datos and datos[key] is not None:
                if isinstance(datos[key], pd.DataFrame) and len(datos[key]) > 0:
                    datos[key].to_excel(writer, sheet_name=sheet, index=(key not in ['balance', 'balance_objetivos', 'implicacion_actores']))
    
    buffer.seek(0)
    return buffer

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.markdown('<div class="main-header">🎭 MACTOR PRO v4.1</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Método de Análisis de Actores - Michel Godet (LIPSOR)</div>', unsafe_allow_html=True)

# Session state
if 'mactor_data' not in st.session_state: st.session_state.mactor_data = None
if 'mactor_resultados' not in st.session_state: st.session_state.mactor_resultados = {}
if 'actores_manual' not in st.session_state: st.session_state.actores_manual = None
if 'objetivos_manual' not in st.session_state: st.session_state.objetivos_manual = None
if 'MAA_manual' not in st.session_state: st.session_state.MAA_manual = None
if 'MAO_manual' not in st.session_state: st.session_state.MAO_manual = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    modo = st.radio("Modo de entrada:", ["📁 Cargar archivo Excel", "✏️ Entrada manual"])
    
    st.divider()
    
    if modo == "📁 Cargar archivo Excel":
        uploaded_file = st.file_uploader("Subir archivo MACTOR", type=['xlsx', 'xls'])
    else:
        n_actores = st.number_input("Nº actores", 2, 20, 5)
        n_objetivos = st.number_input("Nº objetivos", 2, 40, 10)
        if st.button("📋 Crear plantillas"):
            st.session_state.actores_manual = [f"Actor_{i+1}" for i in range(n_actores)]
            st.session_state.objetivos_manual = [f"O{i+1}" for i in range(n_objetivos)]
            st.session_state.MAA_manual = pd.DataFrame(np.zeros((n_actores, n_actores)), index=st.session_state.actores_manual, columns=st.session_state.actores_manual)
            st.session_state.MAO_manual = pd.DataFrame(np.zeros((n_actores, n_objetivos)), index=st.session_state.actores_manual, columns=st.session_state.objetivos_manual)
            st.success("✅ Plantillas creadas")
        uploaded_file = None
    
    st.divider()
    k_midi = st.slider("K (MIDI)", 2, 5, 2)
    mostrar_valores = st.checkbox("Mostrar valores", True)
    top_n_radar = st.slider("Actores en radar", 4, 10, 6)

# Pestañas
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📋 Datos", "👥 Actores", "🎯 Objetivos", "🤝 Convergencias", "🔬 Avanzado", "📊 Síntesis", "📥 Exportar"])

# TAB 1: DATOS
with tab1:
    st.header("📋 Carga de Datos")
    
    if modo == "📁 Cargar archivo Excel":
        if uploaded_file:
            resultado = procesar_archivo_mactor(uploaded_file)
            if 'error' in resultado:
                st.error(f"❌ {resultado['error']}")
            else:
                st.session_state.mactor_data = resultado
                for msg in resultado.get('mensajes', []): st.success(msg)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("👥 Actores", len(resultado.get('actores', [])))
                col2.metric("🎯 Objetivos", len(resultado.get('objetivos', [])))
                col3.metric("📑 Hojas", len(resultado.get('hojas', [])))
                
                if resultado.get('MAA') is not None:
                    st.subheader("Matriz MAA")
                    st.dataframe(resultado['MAA'], height=300)
                if resultado.get('MAO_2') is not None:
                    st.subheader("Matriz 2MAO")
                    st.dataframe(resultado['MAO_2'], height=300)
        else:
            st.info("👆 Sube un archivo Excel con matrices MAA y 2MAO")
    else:
        if st.session_state.actores_manual:
            st.subheader("Editar Actores")
            actores_editados = [st.text_input(f"A{i+1}", v, key=f"act_{i}") for i, v in enumerate(st.session_state.actores_manual)]
            st.session_state.actores_manual = actores_editados
            
            st.subheader("Matriz MAA")
            MAA_edit = st.session_state.MAA_manual.copy()
            MAA_edit.index, MAA_edit.columns = actores_editados, actores_editados
            st.session_state.MAA_manual = st.data_editor(MAA_edit, key="maa_ed")
            
            st.subheader("Matriz 2MAO")
            MAO_edit = st.session_state.MAO_manual.copy()
            MAO_edit.index = actores_editados
            st.session_state.MAO_manual = st.data_editor(MAO_edit, key="mao_ed")
            
            if st.button("✅ Confirmar y procesar", type="primary"):
                maa_vals = st.session_state.MAA_manual.values.copy()
                np.fill_diagonal(maa_vals, 0)
                st.session_state.mactor_data = {
                    'MAA': pd.DataFrame(maa_vals, index=actores_editados, columns=actores_editados),
                    'MAO_2': st.session_state.MAO_manual.copy(),
                    'actores': actores_editados,
                    'objetivos': st.session_state.objetivos_manual,
                    'mensajes': ['✅ Datos manuales procesados']
                }
                st.success("✅ Datos guardados")
        else:
            st.warning("👈 Crea plantillas en el sidebar")

# TAB 2: ACTORES
with tab2:
    st.header("👥 Análisis de Actores")
    
    if st.session_state.mactor_data and st.session_state.mactor_data.get('MAA') is not None:
        data = st.session_state.mactor_data
        MAA = data['MAA']
        
        MIDI = calcular_MIDI(MAA, k_midi)
        balance = calcular_balance_MIDI(MIDI)
        clasificacion, med_Ii, med_Di = clasificar_actores(balance)
        balance['Clasificación'] = clasificacion
        
        st.session_state.mactor_resultados.update({'MIDI': MIDI, 'balance': balance, 'med_Ii': med_Ii, 'med_Di': med_Di})
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 Dominantes", sum(c == 'Dominante' for c in clasificacion))
        col2.metric("🟣 Enlace", sum(c == 'Enlace' for c in clasificacion))
        col3.metric("🔵 Dominados", sum(c == 'Dominado' for c in clasificacion))
        col4.metric("🟠 Autónomos", sum(c == 'Autónomo' for c in clasificacion))
        
        st.dataframe(balance, use_container_width=True)
        
        st.subheader("Plano Influencias/Dependencias")
        mostrar_grafico_con_descargas(crear_plano_influencias(balance, med_Ii, med_Di), "plano", "t2_1")
        
        st.subheader("Balance Neto")
        mostrar_grafico_con_descargas(crear_histograma_balance(balance), "balance", "t2_2")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(crear_heatmap_matriz(MAA, "MAA", 'Blues', mostrar_valores), use_container_width=True, key="hm_maa")
        with col2:
            st.plotly_chart(crear_heatmap_matriz(MIDI, f"MIDI (k={k_midi})", 'Purples', mostrar_valores), use_container_width=True, key="hm_midi")
    else:
        st.warning("⚠️ Carga datos en la pestaña 'Datos'")

# TAB 3: OBJETIVOS
with tab3:
    st.header("🎯 Análisis de Objetivos")
    
    data = st.session_state.mactor_data
    resultados = st.session_state.mactor_resultados
    
    if data and data.get('MAO_2') is not None:
        MAO_2 = data['MAO_2']
        
        # Calcular con manejo de errores
        balance_actores = resultados.get('balance')
        balance_obj = calcular_balance_objetivos(MAO_2, balance_actores)
        impl_actores = calcular_implicacion_actores(MAO_2)
        
        # DEBUG: mostrar info
        st.caption(f"Debug: balance_obj type={type(balance_obj)}, len={len(balance_obj) if hasattr(balance_obj, '__len__') else 'N/A'}")
        if isinstance(balance_obj, pd.DataFrame):
            st.caption(f"Debug: columns={list(balance_obj.columns)}")
        
        # Guardar si es válido
        if isinstance(balance_obj, pd.DataFrame) and len(balance_obj) > 0:
            st.session_state.mactor_resultados['balance_objetivos'] = balance_obj
        if isinstance(impl_actores, pd.DataFrame) and len(impl_actores) > 0:
            st.session_state.mactor_resultados['implicacion_actores'] = impl_actores
        
        # 3MAO
        if 'MIDI' in resultados:
            MAO_3 = calcular_3MAO(MAO_2, resultados['MIDI'])
            if MAO_3 is not None:
                st.session_state.mactor_resultados['MAO_3'] = MAO_3
        
        # Métricas
        st.subheader("📈 Resumen")
        col1, col2, col3 = st.columns(3)
        
        if isinstance(balance_obj, pd.DataFrame) and len(balance_obj) > 0 and 'Balance_ponderado' in balance_obj.columns:
            try:
                col1.metric("✅ Más viable", balance_obj.loc[balance_obj['Balance_ponderado'].idxmax(), 'Objetivo'])
                col2.metric("❌ Más bloqueado", balance_obj.loc[balance_obj['Balance_ponderado'].idxmin(), 'Objetivo'])
                col3.metric("⚡ Más movilizador", balance_obj.loc[balance_obj['Movilizacion'].idxmax(), 'Objetivo'])
            except:
                col1.metric("✅ Más viable", "Error")
                col2.metric("❌ Más bloqueado", "Error")
                col3.metric("⚡ Más movilizador", "Error")
        else:
            col1.metric("✅ Más viable", "N/A")
            col2.metric("❌ Más bloqueado", "N/A")
            col3.metric("⚡ Más movilizador", "N/A")
        
        # Gráficos
        st.subheader("📊 Balance Ponderado")
        fig_bal = crear_grafico_balance_objetivos(balance_obj)
        mostrar_grafico_con_descargas(fig_bal, "balance_obj", "t3_1")
        
        st.subheader("📊 Viabilidad")
        mostrar_grafico_con_descargas(crear_grafico_viabilidad_objetivos(balance_obj), "viabilidad", "t3_2")
        
        st.subheader("📊 Movilización")
        mostrar_grafico_con_descargas(crear_grafico_movilizacion_objetivos(balance_obj), "movilizacion", "t3_3")
        
        st.subheader("📊 Implicación de Actores")
        mostrar_grafico_con_descargas(crear_grafico_implicacion_actores(impl_actores), "implicacion", "t3_4")
        
        st.subheader("🔥 Matriz 2MAO")
        mostrar_grafico_con_descargas(crear_heatmap_matriz(MAO_2, "2MAO", 'RdYlGn', mostrar_valores, 0), "2mao", "t3_5")
        
        if isinstance(balance_obj, pd.DataFrame) and len(balance_obj) > 0:
            st.subheader("📋 Tabla")
            st.dataframe(balance_obj)
    else:
        st.warning("⚠️ Carga datos con matriz 2MAO")

# TAB 4: CONVERGENCIAS
with tab4:
    st.header("🤝 Convergencias y Divergencias")
    
    data = st.session_state.mactor_data
    
    if data and data.get('MAO_2') is not None:
        MAO_2 = data['MAO_2']
        actores = data['actores']
        
        CAA, DAA, CAA_pond, DAA_pond = calcular_convergencias_divergencias(MAO_2)
        st.session_state.mactor_resultados.update({'CAA': CAA, 'DAA': DAA, 'CAA_pond': CAA_pond, 'DAA_pond': DAA_pond})
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Conv. simple", int(CAA.values.sum()/2))
        col2.metric("Div. simple", int(DAA.values.sum()/2))
        col3.metric("Conv. pond.", f"{CAA_pond.values.sum()/2:.0f}")
        col4.metric("Div. pond.", f"{DAA_pond.values.sum()/2:.0f}")
        
        st.subheader("Simple")
        mostrar_grafico_con_descargas(crear_grafico_convergencias_actor(CAA, DAA, actores), "conv_simple", "t4_1")
        
        st.subheader("Ponderadas")
        mostrar_grafico_con_descargas(crear_grafico_convergencias_ponderadas(CAA_pond, DAA_pond, actores), "conv_pond", "t4_2")
        
        st.subheader("Matriz Alianzas Simple")
        mostrar_grafico_con_descargas(crear_matriz_alianzas_conflictos(CAA, DAA, False), "alianzas_s", "t4_3")
        
        st.subheader("Matriz Alianzas Ponderada")
        mostrar_grafico_con_descargas(crear_matriz_alianzas_conflictos(CAA_pond, DAA_pond, True), "alianzas_p", "t4_4")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(crear_heatmap_matriz(CAA_pond, "Convergencias Pond.", 'Greens', mostrar_valores), use_container_width=True, key="hm_caa")
        with col2:
            st.plotly_chart(crear_heatmap_matriz(DAA_pond, "Divergencias Pond.", 'Reds', mostrar_valores), use_container_width=True, key="hm_daa")
        
        st.subheader("Ambivalencia")
        mostrar_grafico_con_descargas(crear_heatmap_matriz(calcular_ambivalencia(CAA, DAA), "Ambivalencia", 'YlOrRd', mostrar_valores), "ambiv", "t4_5")
    else:
        st.warning("⚠️ Carga datos")

# TAB 5: AVANZADO
with tab5:
    st.header("🔬 Análisis Avanzado")
    
    data = st.session_state.mactor_data
    resultados = st.session_state.mactor_resultados
    
    if data and resultados:
        if 'balance' in resultados and 'CAA' in resultados:
            st.subheader("Red de Actores")
            mostrar_grafico_con_descargas(crear_red_actores(resultados['balance'], resultados['CAA'], resultados['DAA']), "red", "t5_1")
        
        if 'balance' in resultados:
            st.subheader("Radar")
            mostrar_grafico_con_descargas(crear_grafico_radar_actores(resultados['balance'], top_n_radar), "radar", "t5_2")
        
        if 'MAO_3' in resultados and resultados['MAO_3'] is not None:
            st.subheader("3MAO")
            mostrar_grafico_con_descargas(crear_heatmap_matriz(resultados['MAO_3'], "3MAO", 'RdYlGn', mostrar_valores, 0), "3mao", "t5_3")
        
        if 'CAA_pond' in resultados:
            st.subheader("Alianzas y Conflictos Clave")
            balance_mat = resultados['CAA_pond'] - resultados['DAA_pond']
            actores = data['actores']
            
            alianzas, conflictos = [], []
            for i in range(len(actores)):
                for j in range(i+1, len(actores)):
                    v = balance_mat.iloc[i, j]
                    if v > 2: alianzas.append((actores[i], actores[j], v))
                    elif v < -1: conflictos.append((actores[i], actores[j], v))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🟢 Alianzas:**")
                for a1, a2, v in sorted(alianzas, key=lambda x: -x[2])[:7]:
                    st.markdown(f"- {a1} ↔ {a2}: **+{v:.1f}**")
            with col2:
                st.markdown("**🔴 Conflictos:**")
                for a1, a2, v in sorted(conflictos, key=lambda x: x[2])[:7]:
                    st.markdown(f"- {a1} ↔ {a2}: **{v:.1f}**")
    else:
        st.warning("⚠️ Ejecuta análisis previo")

# TAB 6: SÍNTESIS
with tab6:
    st.header("📊 Síntesis")
    
    data = st.session_state.mactor_data
    resultados = st.session_state.mactor_resultados
    
    if data and resultados:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Actores Clave")
            if 'balance' in resultados and 'Ri_neto' in resultados['balance'].columns:
                bal = resultados['balance']
                st.markdown("**Top 3 Dominantes:**")
                for _, r in bal.nlargest(3, 'Ri_neto').iterrows():
                    st.markdown(f"- **{r['Actor']}**: +{r['Ri_neto']:.0f}")
                st.markdown("**Top 3 Dominados:**")
                for _, r in bal.nsmallest(3, 'Ri_neto').iterrows():
                    st.markdown(f"- **{r['Actor']}**: {r['Ri_neto']:.0f}")
        
        with col2:
            st.subheader("🎯 Objetivos Clave")
            if 'balance_objetivos' in resultados and isinstance(resultados['balance_objetivos'], pd.DataFrame):
                bal_obj = resultados['balance_objetivos']
                if 'Viabilidad' in bal_obj.columns:
                    st.markdown("**Más viables:**")
                    for _, r in bal_obj.nlargest(3, 'Viabilidad').iterrows():
                        st.markdown(f"- **{r['Objetivo']}**: {r['Viabilidad']:+.2f}")
                    st.markdown("**Bloqueados:**")
                    for _, r in bal_obj.nsmallest(3, 'Viabilidad').iterrows():
                        st.markdown(f"- **{r['Objetivo']}**: {r['Viabilidad']:+.2f}")
        
        st.subheader("💡 Recomendaciones")
        st.markdown("""
        <div class="info-box">
        <b>1.</b> Negociar primero con actores dominantes<br>
        <b>2.</b> Usar actores enlace como mediadores<br>
        <b>3.</b> Priorizar objetivos viables para coaliciones<br>
        <b>4.</b> Reformular objetivos bloqueados
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Ejecuta análisis")

# TAB 7: EXPORTAR
with tab7:
    st.header("📥 Exportar")
    
    data = st.session_state.mactor_data
    resultados = st.session_state.mactor_resultados
    
    if data and resultados:
        nombre = st.text_input("Nombre proyecto", "Analisis_MACTOR")
        
        datos_exp = {**data, **resultados, 'k': k_midi}
        
        buffer = generar_informe_excel(datos_exp, nombre)
        st.download_button("📥 Descargar Excel", buffer, f"{nombre}_MACTOR.xlsx", 
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        
        st.subheader("CSVs")
        for key, label in [('balance', 'Balance Actores'), ('balance_objetivos', 'Balance Objetivos')]:
            if key in resultados and isinstance(resultados[key], pd.DataFrame):
                st.download_button(f"📥 {label}", resultados[key].to_csv(index=False), f"{key}.csv", "text/csv", key=f"csv_{key}")
    else:
        st.warning("⚠️ Procesa datos primero")

# Footer
st.divider()
st.markdown('<div style="text-align:center;color:#666;"><b>MACTOR PRO v4.1</b> | Michel Godet (LIPSOR) | JETLEX | 2025</div>', unsafe_allow_html=True)
