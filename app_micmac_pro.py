"""
MICMAC PRO - Análisis Estructural con Conversor Integrado
Matriz de Impactos Cruzados - Multiplicación Aplicada a una Clasificación

Autor: JETLEX Strategic Consulting by Martín Pratto Chiarella
Basado en el método de Michel Godet (1990)
Versión: 5.5 - Metodología canónica Godet (umbral = MEDIA ARITMÉTICA)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import re
from datetime import datetime

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
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES DE GENERACIÓN DE CÓDIGOS INTELIGENTES
# ============================================================

STOPWORDS = {
    'de', 'del', 'la', 'las', 'el', 'los', 'en', 'y', 'a', 'con', 'por', 'para',
    'un', 'una', 'unos', 'unas', 'al', 'su', 'sus', 'que', 'se', 'es', 'son',
    'the', 'of', 'and', 'in', 'to', 'for', 'a', 'an', 'on', 'at', 'by', 'with',
    'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
}

def tiene_codigo_explicito(nombre):
    """Detecta si el nombre ya tiene un código explícito al inicio."""
    if pd.isna(nombre):
        return False, None
    
    nombre = str(nombre).strip()
    
    match = re.match(r'^([A-Za-z]+\d+)\s', nombre)
    if match:
        return True, match.group(1).upper()
    
    match = re.match(r'^([A-Z][A-Z0-9_]{1,12})$', nombre)
    if match:
        return True, match.group(1)
    
    match = re.match(r'^([A-Z][A-Z0-9_]{1,12})\s', nombre)
    if match and len(match.group(1)) <= 12:
        return True, match.group(1)
    
    return False, None

def generar_abreviatura_inteligente(nombre, max_chars=10):
    """Genera una abreviatura inteligente a partir de un nombre largo."""
    if pd.isna(nombre):
        return "VAR"
    
    nombre = str(nombre).strip()
    nombre_limpio = re.sub(r'^[^a-zA-Z]+', '', nombre)
    palabras = re.findall(r'[A-Za-záéíóúñÁÉÍÓÚÑ]+', nombre_limpio)
    palabras_significativas = [p for p in palabras if p.lower() not in STOPWORDS]
    
    if not palabras_significativas:
        palabras_significativas = palabras
    
    if not palabras_significativas:
        return "VAR"
    
    n_palabras = len(palabras_significativas)
    
    if n_palabras == 1:
        return palabras_significativas[0][:max_chars].upper()
    elif n_palabras == 2:
        chars_cada = max_chars // 2
        abrev = palabras_significativas[0][:chars_cada] + palabras_significativas[1][:chars_cada]
        return abrev.upper()[:max_chars]
    else:
        if n_palabras <= 4:
            chars_cada = max(2, max_chars // n_palabras)
        else:
            chars_cada = max(1, max_chars // min(n_palabras, 5))
        
        abrev = ''.join([p[:chars_cada] for p in palabras_significativas[:5]])
        return abrev.upper()[:max_chars]

def extraer_codigo(nombre_variable, max_chars=10):
    """Extrae o genera un código corto para una variable."""
    if pd.isna(nombre_variable):
        return "VAR"
    
    nombre = str(nombre_variable).strip()
    tiene_codigo, codigo = tiene_codigo_explicito(nombre)
    if tiene_codigo:
        return codigo[:max_chars]
    
    return generar_abreviatura_inteligente(nombre, max_chars)

def generar_codigos_y_mapeo(nombres_variables, max_chars=10):
    """Genera códigos únicos para cada variable."""
    codigos = []
    mapeo = {}
    codigos_usados = {}
    
    for i, nombre in enumerate(nombres_variables):
        codigo_base = extraer_codigo(nombre, max_chars)
        
        if codigo_base in codigos_usados:
            codigos_usados[codigo_base] += 1
            sufijo = str(codigos_usados[codigo_base])
            max_base = max_chars - len(sufijo)
            codigo = codigo_base[:max_base] + sufijo
        else:
            codigos_usados[codigo_base] = 0
            codigo = codigo_base
        
        codigos.append(codigo)
        mapeo[codigo] = nombre
    
    return codigos, mapeo

def truncar_texto(texto, max_chars=30):
    """Trunca texto largo agregando '...' si excede el máximo"""
    if pd.isna(texto):
        return ""
    texto = str(texto)
    if len(texto) <= max_chars:
        return texto
    return texto[:max_chars-3] + "..."

# ============================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================

def mostrar_grafico_con_descargas(fig, nombre_base, key_suffix=""):
    """Muestra un gráfico Plotly con opciones de descarga"""
    
    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': nombre_base,
            'height': 800,
            'width': 1200,
            'scale': 2
        },
        'displaylogo': False,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config, key=f"chart_{nombre_base}_{key_suffix}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        html_buffer = BytesIO()
        html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html_buffer.write(html_content.encode())
        html_buffer.seek(0)
        
        st.download_button(
            label="📥 HTML Interactivo",
            data=html_buffer,
            file_name=f"{nombre_base}.html",
            mime="text/html",
            key=f"html_{nombre_base}_{key_suffix}"
        )
    
    with col2:
        try:
            svg_str = fig.to_image(format="svg").decode('utf-8')
            st.download_button(
                label="📥 SVG (Vector)",
                data=svg_str,
                file_name=f"{nombre_base}.svg",
                mime="image/svg+xml",
                key=f"svg_{nombre_base}_{key_suffix}"
            )
        except Exception:
            st.info("📷 Usa el ícono de cámara en el gráfico")
    
    with col3:
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            st.download_button(
                label="📥 PNG (Imagen)",
                data=img_bytes,
                file_name=f"{nombre_base}.png",
                mime="image/png",
                key=f"png_{nombre_base}_{key_suffix}"
            )
        except Exception:
            st.markdown("📷 **PNG:** Clic en 📷 del gráfico")

# ============================================================
# FUNCIONES DE CÁLCULO MICMAC - DIRECTO E INDIRECTO
# ============================================================

def calcular_mid_directo(M):
    """
    Análisis DIRECTO (MID)
    Usa la matriz original M sin transformaciones.
    Valores en escala original (0-3 × n variables).
    """
    M = np.array(M, dtype=float)
    np.fill_diagonal(M, 0)
    return M

def calcular_motricidad_dependencia_directa(M):
    """
    Calcula Motricidad y Dependencia DIRECTAS
    - Motricidad (M): suma de la fila = influencia que EJERCE la variable
    - Dependencia (D): suma de la columna = influencia que RECIBE la variable
    """
    M = np.array(M, dtype=float)
    np.fill_diagonal(M, 0)
    motricidad = M.sum(axis=1)  # Suma de filas
    dependencia = M.sum(axis=0)  # Suma de columnas
    return motricidad, dependencia

def clasificar_variables_directas(motricidad, dependencia):
    """
    Clasifica variables según análisis DIRECTO
    IMPORTANTE: Usa MEDIA ARITMÉTICA como umbral (metodología canónica de Godet)
    Referencia: Godet, M. (2007) - "el umbral de clasificación es la media aritmética 
    de motricidad y dependencia del sistema"
    
    Nomenclatura clásica de Godet para análisis directo:
    - Motrices: Alta M, Baja D (palancas del sistema)
    - Enlace/Relé: Alta M, Alta D (nudos críticos)
    - Resultado/Dependientes: Baja M, Alta D (indicadores)
    - Autónomas/Excluidas: Baja M, Baja D (poco relevantes)
    """
    # MEDIA ARITMÉTICA (metodología Godet/LIPSOR canónica)
    med_mot = np.mean(motricidad)
    med_dep = np.mean(dependencia)
    
    clasificacion = []
    for mot, dep in zip(motricidad, dependencia):
        if mot >= med_mot and dep < med_dep:
            clasificacion.append("Motrices")
        elif mot >= med_mot and dep >= med_dep:
            clasificacion.append("Enlace")
        elif mot < med_mot and dep >= med_dep:
            clasificacion.append("Resultado")
        else:
            clasificacion.append("Autónomas")
    
    return clasificacion, med_mot, med_dep

def calcular_midi(M, alpha=0.5, K=3):
    """
    Análisis INDIRECTO (MIDI)
    Calcula la Matriz de Influencias Directas e Indirectas.
    MIDI = M + αM² + α²M³ + ... + α^(K-1)M^K
    Valores en escala amplificada (pueden ser millones).
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
    """Calcula motricidad y dependencia de MIDI (análisis indirecto)"""
    motricidad = MIDI.sum(axis=1)
    dependencia = MIDI.sum(axis=0)
    return motricidad, dependencia

def clasificar_variables(motricidad, dependencia):
    """
    Clasifica variables según análisis INDIRECTO
    IMPORTANTE: Usa MEDIA ARITMÉTICA como umbral (metodología canónica de Godet)
    Referencia: Godet, M. (2007)
    
    Nomenclatura para análisis indirecto:
    - Determinantes: Alta M, Baja D
    - Clave: Alta M, Alta D
    - Variables resultado: Baja M, Alta D
    - Autónomas: Baja M, Baja D
    """
    # MEDIA ARITMÉTICA (metodología Godet/LIPSOR canónica)
    med_mot = np.mean(motricidad)
    med_dep = np.mean(dependencia)
    
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
# FUNCIONES DE ANÁLISIS ADICIONALES
# ============================================================

def identificar_relaciones_fuertes(M, nombres, umbral=2):
    """Identifica las relaciones más fuertes de la matriz"""
    relaciones = []
    n = len(nombres)
    for i in range(n):
        for j in range(n):
            if i != j and M[i, j] >= umbral:
                relaciones.append({
                    'Origen': nombres[i],
                    'Destino': nombres[j],
                    'Intensidad': M[i, j]
                })
    return sorted(relaciones, key=lambda x: x['Intensidad'], reverse=True)

def comparar_directo_indirecto(df_directo, df_indirecto):
    """Compara clasificaciones entre análisis directo e indirecto"""
    df_comp = pd.merge(
        df_directo[['Código', 'Variable', 'Clasificación']].rename(columns={'Clasificación': 'Clasif_Directa'}),
        df_indirecto[['Código', 'Clasificación']].rename(columns={'Clasificación': 'Clasif_Indirecta'}),
        on='Código'
    )
    df_comp['Cambio'] = df_comp['Clasif_Directa'] != df_comp['Clasif_Indirecta']
    return df_comp

# ============================================================
# PROCESADOR ROBUSTO DE ARCHIVOS EXCEL
# ============================================================

def es_celda_excluir(valor):
    """Detecta si una celda contiene valores a excluir"""
    if pd.isna(valor):
        return False
    valor_str = str(valor).upper().strip()
    palabras_excluir = ['SUMA', 'TOTAL', 'SUM', 'PROMEDIO', 'AVERAGE', 'MEAN', 'Σ', 'MOTRI', 'DEPEN']
    return any(palabra in valor_str for palabra in palabras_excluir)

def es_valor_diagonal(valor):
    """Detecta si un valor es marcador de diagonal"""
    if pd.isna(valor):
        return False
    valor_str = str(valor).upper().strip()
    return valor_str in ['X', '-', 'N/A', 'NA', 'DIAG', '*']

def convertir_valor_numerico(valor):
    """Convierte un valor a numérico"""
    if pd.isna(valor):
        return 0.0
    if es_valor_diagonal(valor):
        return 0.0
    try:
        return float(valor)
    except (ValueError, TypeError):
        return 0.0

def detectar_inicio_matriz(df):
    """Detecta automáticamente dónde empieza la matriz de datos"""
    for i in range(min(20, len(df))):
        fila = df.iloc[i, 1:min(10, len(df.columns))]
        
        valores_validos = 0
        for val in fila:
            if pd.isna(val):
                continue
            if es_valor_diagonal(val):
                valores_validos += 1
            else:
                try:
                    float(val)
                    valores_validos += 1
                except:
                    pass
        
        if valores_validos >= len(fila) * 0.5:
            return i
    
    return 0

def procesar_archivo_excel(uploaded_file, nombre_hoja=None):
    """Procesa archivo Excel de forma robusta"""
    try:
        xl = pd.ExcelFile(uploaded_file)
        hojas_disponibles = xl.sheet_names
        
        if nombre_hoja and nombre_hoja in hojas_disponibles:
            hoja_usar = nombre_hoja
        else:
            hoja_usar = hojas_disponibles[0]
            for hoja in hojas_disponibles:
                if 'MID' in hoja.upper() or 'MATRIZ' in hoja.upper():
                    hoja_usar = hoja
                    break
        
        df = pd.read_excel(xl, sheet_name=hoja_usar, header=None)
        
        if df.empty:
            return None, None, None, "El archivo está vacío"
        
        fila_inicio = detectar_inicio_matriz(df)
        
        if fila_inicio > 0:
            headers = df.iloc[fila_inicio - 1, :].tolist()
        else:
            headers = df.iloc[0, :].tolist()
            fila_inicio = 1
        
        columnas_validas = []
        for j in range(1, len(df.columns)):
            header = headers[j] if j < len(headers) else None
            if not es_celda_excluir(header):
                columnas_validas.append(j)
        
        filas_validas = []
        nombres_variables = []
        
        for i in range(fila_inicio, len(df)):
            nombre_fila = df.iloc[i, 0]
            
            if es_celda_excluir(nombre_fila):
                continue
            
            fila_datos = df.iloc[i, columnas_validas[:5]]
            valores_numericos = 0
            for val in fila_datos:
                if es_valor_diagonal(val) or (not pd.isna(val) and isinstance(val, (int, float))):
                    valores_numericos += 1
                else:
                    try:
                        float(val)
                        valores_numericos += 1
                    except:
                        pass
            
            if valores_numericos >= 2:
                filas_validas.append(i)
                nombres_variables.append(str(nombre_fila).strip() if pd.notna(nombre_fila) else f"V{len(nombres_variables)+1}")
        
        if len(filas_validas) == 0:
            return None, None, None, "No se encontraron filas con datos válidos"
        
        n = min(len(filas_validas), len(columnas_validas))
        filas_validas = filas_validas[:n]
        columnas_validas = columnas_validas[:n]
        nombres_variables = nombres_variables[:n]
        
        matriz = np.zeros((n, n))
        for i, fila_idx in enumerate(filas_validas):
            for j, col_idx in enumerate(columnas_validas):
                valor = df.iloc[fila_idx, col_idx]
                matriz[i, j] = convertir_valor_numerico(valor)
        
        np.fill_diagonal(matriz, 0)
        
        df_matriz = pd.DataFrame(matriz, index=nombres_variables, columns=nombres_variables)
        
        return df_matriz, nombres_variables, hojas_disponibles, f"✅ Matriz {n}x{n} de hoja '{hoja_usar}' procesada correctamente"
            
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

# ============================================================
# GENERACIÓN DE GRÁFICOS
# ============================================================

def crear_grafico_subsistemas(df_res, med_mot, med_dep, motricidad, dependencia, titulo, usar_codigos=True, mostrar_etiquetas=True, tamaño_fuente=10, M_original=None):
    """
    Crea gráfico de subsistemas profesional estilo LIPSOR/Godet
    Con cuadrantes coloreados, estadísticas y diseño HD
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Calcular límites
    max_mot = max(motricidad) * 1.15
    max_dep = max(dependencia) * 1.15
    min_mot = min(0, min(motricidad) * 0.9)
    min_dep = min(0, min(dependencia) * 0.9)
    
    # Contar clasificaciones
    from collections import Counter
    conteo = Counter(df_res['Clasificación'].tolist())
    
    # Colores por clasificación
    colores_puntos = {
        'Motrices': '#1E3A8A',      # Azul oscuro
        'Determinantes': '#1E3A8A',
        'Enlace': '#B91C1C',         # Rojo oscuro
        'Clave': '#B91C1C',
        'Resultado': '#DC2626',      # Rojo
        'Variables resultado': '#DC2626',
        'Autónomas': '#1E40AF'       # Azul
    }
    
    # Colores de fondo de cuadrantes (muy suaves)
    fig = go.Figure()
    
    # Agregar rectángulos de fondo para cada cuadrante
    # Cuadrante MOTRICES (arriba-izquierda): verde muy suave
    fig.add_shape(type="rect", x0=min_dep, y0=med_mot, x1=med_dep, y1=max_mot,
                  fillcolor="rgba(220, 252, 231, 0.5)", line=dict(width=0), layer="below")
    
    # Cuadrante ENLACE (arriba-derecha): rojo muy suave
    fig.add_shape(type="rect", x0=med_dep, y0=med_mot, x1=max_dep, y1=max_mot,
                  fillcolor="rgba(254, 226, 226, 0.5)", line=dict(width=0), layer="below")
    
    # Cuadrante AUTÓNOMAS (abajo-izquierda): amarillo muy suave
    fig.add_shape(type="rect", x0=min_dep, y0=min_mot, x1=med_dep, y1=med_mot,
                  fillcolor="rgba(254, 249, 195, 0.5)", line=dict(width=0), layer="below")
    
    # Cuadrante RESULTADO (abajo-derecha): azul muy suave
    fig.add_shape(type="rect", x0=med_dep, y0=min_mot, x1=max_dep, y1=med_mot,
                  fillcolor="rgba(219, 234, 254, 0.5)", line=dict(width=0), layer="below")
    
    # Agregar títulos de cuadrantes
    font_cuadrante = dict(size=14, color='rgba(0,0,0,0.7)', family='Arial Black')
    font_subtitulo = dict(size=9, color='rgba(0,0,0,0.5)', family='Arial')
    
    # MOTRICES / PALANCAS
    fig.add_annotation(x=(min_dep + med_dep)/2, y=max_mot * 0.92,
                      text="<b>MOTRICES / PALANCAS</b>", showarrow=False,
                      font=dict(size=13, color='#166534'), opacity=0.8)
    fig.add_annotation(x=(min_dep + med_dep)/2, y=max_mot * 0.85,
                      text="Alta motricidad · Baja dependencia", showarrow=False,
                      font=font_subtitulo)
    
    # ENLACE / CLAVE
    fig.add_annotation(x=(med_dep + max_dep)/2, y=max_mot * 0.92,
                      text="<b>ENLACE / CLAVE</b>", showarrow=False,
                      font=dict(size=13, color='#991B1B'), opacity=0.8)
    fig.add_annotation(x=(med_dep + max_dep)/2, y=max_mot * 0.85,
                      text="Alta motricidad · Alta dependencia", showarrow=False,
                      font=font_subtitulo)
    
    # AUTÓNOMAS
    fig.add_annotation(x=(min_dep + med_dep)/2, y=min_mot + (med_mot - min_mot) * 0.08,
                      text="<b>AUTÓNOMAS</b>", showarrow=False,
                      font=dict(size=13, color='#92400E'), opacity=0.8)
    fig.add_annotation(x=(min_dep + med_dep)/2, y=min_mot + (med_mot - min_mot) * 0.02,
                      text="Baja motricidad · Baja dependencia", showarrow=False,
                      font=font_subtitulo)
    
    # RESULTADO / DEPENDIENTES
    fig.add_annotation(x=(med_dep + max_dep)/2, y=min_mot + (med_mot - min_mot) * 0.08,
                      text="<b>RESULTADO / DEPENDIENTES</b>", showarrow=False,
                      font=dict(size=13, color='#1E40AF'), opacity=0.8)
    fig.add_annotation(x=(med_dep + max_dep)/2, y=min_mot + (med_mot - min_mot) * 0.02,
                      text="Baja motricidad · Alta dependencia", showarrow=False,
                      font=font_subtitulo)
    
    # Agregar puntos por clasificación
    for clasif in df_res['Clasificación'].unique():
        color = colores_puntos.get(clasif, '#666666')
        mask = df_res['Clasificación'] == clasif
        df_temp = df_res[mask]
        n_vars = len(df_temp)
        
        if n_vars > 0:
            etiquetas_temp = df_temp['Código'].tolist() if usar_codigos else df_temp['Variable'].tolist()
            hover_text = [f"<b>{c}</b><br>{v}<br>M={m:.0f}, D={d:.0f}" 
                         for c, v, m, d in zip(df_temp['Código'], df_temp['Variable'], 
                                               df_temp['Motricidad'], df_temp['Dependencia'])]
            
            fig.add_trace(go.Scatter(
                x=df_temp['Dependencia'],
                y=df_temp['Motricidad'],
                mode='markers+text' if mostrar_etiquetas else 'markers',
                name=f"{clasif} ({n_vars})",
                text=etiquetas_temp if mostrar_etiquetas else None,
                textposition='top center',
                textfont=dict(size=tamaño_fuente, color='#1f2937'),
                marker=dict(size=12, color=color, line=dict(width=1.5, color='white'),
                           opacity=0.9),
                hovertext=hover_text,
                hoverinfo='text'
            ))
    
    # Líneas de media (más visibles)
    fig.add_hline(y=med_mot, line_dash="dash", line_color="#9CA3AF", line_width=2, opacity=0.8)
    fig.add_vline(x=med_dep, line_dash="dash", line_color="#9CA3AF", line_width=2, opacity=0.8)
    
    # Calcular estadísticas para el panel
    n_vars = len(df_res)
    fill_rate = 0
    if M_original is not None:
        n = M_original.shape[0]
        # Fill rate: excluir diagonal (autoinfluencia no se cuenta en MICMAC)
        n_total = n * (n - 1)  # Celdas posibles sin diagonal
        n_nonzero = (M_original != 0).sum()
        fill_rate = (n_nonzero / n_total) * 100 if n_total > 0 else 0
    
    # Top 5 Motrices
    top5_mot = df_res.nlargest(5, 'Motricidad')[['Código', 'Motricidad']]
    # Top 5 Dependientes
    top5_dep = df_res.nlargest(5, 'Dependencia')[['Código', 'Dependencia']]
    
    # Crear texto de estadísticas para anotación
    stats_text = f"<b>ESTADÍSTICAS</b><br>"
    stats_text += f"Variables: {n_vars}<br>"
    if fill_rate > 0:
        stats_text += f"Fill rate: {fill_rate:.1f}%<br>"
    stats_text += f"Prom. M: {med_mot:.1f}<br>"
    stats_text += f"Prom. D: {med_dep:.1f}<br><br>"
    stats_text += f"<b>TOP 5 MOTRICES</b><br>"
    for _, row in top5_mot.iterrows():
        stats_text += f"{row['Código']}: M={row['Motricidad']:.0f}<br>"
    stats_text += f"<br><b>TOP 5 DEPENDIENTES</b><br>"
    for _, row in top5_dep.iterrows():
        stats_text += f"{row['Código']}: D={row['Dependencia']:.0f}<br>"
    
    # Subtítulo con metadata (más compacto)
    subtitulo = f"Matriz {n_vars}×{n_vars}"
    if fill_rate > 0:
        subtitulo += f" · Fill: {fill_rate:.1f}%"
    subtitulo += f" · Media: M={med_mot:.1f}, D={med_dep:.1f}"
    
    fig.update_layout(
        title=dict(
            text=f"<b>PLANO MOTRICIDAD-DEPENDENCIA</b><br><sup>Clasificación de {n_vars} variables MICMAC · {subtitulo}</sup>",
            x=0.45,  # Centrado considerando el margen derecho
            xanchor='center',
            font=dict(size=15)
        ),
        xaxis=dict(
            title=dict(text="<b>DEPENDENCIA →</b>", font=dict(size=12)),
            range=[min_dep, max_dep],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="<b>↑ MOTRICIDAD</b>", font=dict(size=12)),
            range=[min_mot, max_mot],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False
        ),
        height=700,
        showlegend=True,
        legend=dict(
            title=dict(text="<b>CLASIFICACIÓN</b>", font=dict(size=11)),
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(r=200),  # Espacio para la leyenda
        plot_bgcolor='white',
        paper_bgcolor='white',
        # Agregar anotación con estadísticas
        annotations=[
            dict(
                x=1.02,
                y=0.45,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                font=dict(size=9, family="Courier New"),
                align="left",
                bgcolor="rgba(249,250,251,0.95)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                borderpad=8
            ),
            # Pie de página
            dict(
                x=0.5,
                y=-0.12,
                xref="paper",
                yref="paper",
                text="Fuente: Elaboración propia · Metodología: Godet/LIPSOR · Software: MICMAC PRO v5.5",
                showarrow=False,
                font=dict(size=9, color='gray'),
                align="center"
            )
        ]
    )
    
    return fig

def crear_grafico_red_influencias(M, nombres, codigos, umbral=2, usar_codigos=True):
    """Crea un gráfico de red de influencias fuertes"""
    etiquetas = codigos if usar_codigos else nombres
    n = len(etiquetas)
    
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos_x = np.cos(angles)
    pos_y = np.sin(angles)
    
    edge_x = []
    edge_y = []
    
    for i in range(n):
        for j in range(n):
            if i != j and M[i, j] >= umbral:
                edge_x.extend([pos_x[i], pos_x[j], None])
                edge_y.extend([pos_y[i], pos_y[j], None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(150,150,150,0.5)'),
        hoverinfo='none',
        mode='lines',
        name='Influencias'
    )
    
    influencia_recibida = M.sum(axis=0)
    node_sizes = 15 + (influencia_recibida / max(influencia_recibida.max(), 1)) * 30
    motricidad = M.sum(axis=1)
    
    hover_text = [f"<b>{codigos[i]}</b><br>{nombres[i]}<br>Influencia recibida: {influencia_recibida[i]:.0f}" for i in range(n)]
    
    node_trace = go.Scatter(
        x=pos_x, y=pos_y,
        mode='markers+text',
        hoverinfo='text',
        text=etiquetas,
        textposition='top center',
        hovertext=hover_text,
        marker=dict(
            size=node_sizes,
            color=motricidad,
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title="Motricidad"),
            line=dict(width=2, color='white')
        ),
        name='Variables'
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Red de Influencias Fuertes (≥{umbral})',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        ))
    
    return fig

def crear_grafico_estabilidad(M, nombres, codigos, alpha=0.5, usar_codigos=True):
    """Gráfico de estabilidad del ranking por K"""
    etiquetas = codigos if usar_codigos else nombres
    
    data = []
    for K in range(1, 6):
        if K == 1:
            MIDI = M.copy()
        else:
            MIDI = calcular_midi(M, alpha=alpha, K=K)
        mot, _ = calcular_motricidad_dependencia(MIDI)
        
        rankings = len(mot) - np.argsort(np.argsort(mot))
        for i, (nombre, rank) in enumerate(zip(etiquetas, rankings)):
            data.append({'K': K, 'Variable': nombre, 'Ranking': rank, 'Motricidad': mot[i]})
    
    df = pd.DataFrame(data)
    
    top_vars = df[df['K'] == 5].nlargest(10, 'Motricidad')['Variable'].tolist()
    df_top = df[df['Variable'].isin(top_vars)]
    
    fig = px.line(df_top, x='K', y='Ranking', color='Variable',
        title='Estabilidad del Ranking (Top 10) según Profundidad K',
        labels={'K': 'Profundidad K', 'Ranking': 'Posición en Ranking'},
        markers=True)
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=500)
    
    return fig

def crear_grafico_distribucion_cuadrantes(clasificacion, titulo="Distribución de Variables"):
    """Gráfico de distribución por cuadrantes"""
    conteo = pd.Series(clasificacion).value_counts()
    
    colores = {
        'Motrices': '#FF4444', 'Determinantes': '#FF4444',
        'Enlace': '#1166CC', 'Clave': '#1166CC',
        'Resultado': '#66BBFF', 'Variables resultado': '#66BBFF',
        'Autónomas': '#FF9944'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=conteo.index,
        values=conteo.values,
        hole=0.4,
        marker_colors=[colores.get(c, '#999') for c in conteo.index],
        textinfo='label+percent+value'
    )])
    
    fig.update_layout(title=titulo, height=400)
    
    return fig

def crear_grafico_relaciones_fuertes(M, nombres, codigos, top_n=20, usar_codigos=True):
    """Gráfico de barras de las relaciones más fuertes"""
    etiquetas = codigos if usar_codigos else nombres
    
    relaciones = []
    for i in range(len(M)):
        for j in range(len(M)):
            if i != j and M[i, j] > 0:
                relaciones.append({
                    'Par': f"{etiquetas[i]} → {etiquetas[j]}",
                    'Intensidad': M[i, j],
                    'Detalle': f"{nombres[i]} → {nombres[j]}"
                })
    
    df = pd.DataFrame(relaciones).nlargest(top_n, 'Intensidad')
    
    fig = px.bar(df, x='Intensidad', y='Par', orientation='h',
        title=f'Top {top_n} Relaciones de Influencia Más Fuertes',
        color='Intensidad',
        color_continuous_scale='Reds',
        hover_data=['Detalle'])
    
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    
    return fig

def crear_grafico_inestabilidad(df_resultados):
    """Gráfico de índice de inestabilidad"""
    df = df_resultados.copy()
    df['Inestabilidad'] = df['Motricidad'] * df['Dependencia']
    df = df.nlargest(15, 'Inestabilidad')
    
    fig = px.bar(df, x='Código', y='Inestabilidad',
        title='Índice de Inestabilidad (Variables que amplifican cambios)',
        color='Clasificación',
        hover_data=['Variable'])
    
    fig.update_layout(height=400)
    
    return fig

def crear_grafico_desplazamiento_ranking(df_directo, df_indirecto, tipo='Motricidad'):
    """
    Crea gráfico de desplazamiento de ranking estilo LIPSOR
    Muestra cómo cambian las posiciones entre análisis directo e indirecto
    tipo: 'Motricidad' o 'Dependencia'
    """
    # Calcular rankings
    df_dir = df_directo.copy()
    df_ind = df_indirecto.copy()
    
    df_dir['Rank_Dir'] = df_dir[tipo].rank(ascending=False).astype(int)
    df_ind['Rank_Ind'] = df_ind[tipo].rank(ascending=False).astype(int)
    
    # Merge para comparar
    df_comp = pd.merge(
        df_dir[['Código', 'Variable', 'Rank_Dir']],
        df_ind[['Código', 'Rank_Ind']],
        on='Código'
    )
    
    # Calcular cambio (positivo = subió en ranking, negativo = bajó)
    df_comp['Cambio'] = df_comp['Rank_Dir'] - df_comp['Rank_Ind']
    
    # Ordenar por ranking directo
    df_comp = df_comp.sort_values('Rank_Dir')
    
    n = len(df_comp)
    
    fig = go.Figure()
    
    # Dibujar líneas de conexión
    for _, row in df_comp.iterrows():
        # Color según si subió (verde) o bajó (rojo)
        if row['Cambio'] > 0:
            color = 'green'
            width = min(1 + abs(row['Cambio']) * 0.3, 4)
        elif row['Cambio'] < 0:
            color = 'red'
            width = min(1 + abs(row['Cambio']) * 0.3, 4)
        else:
            color = 'gray'
            width = 1
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[row['Rank_Dir'], row['Rank_Ind']],
            mode='lines',
            line=dict(color=color, width=width),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Agregar puntos y etiquetas lado izquierdo (Directo)
    fig.add_trace(go.Scatter(
        x=[0] * n,
        y=df_comp['Rank_Dir'],
        mode='markers+text',
        marker=dict(size=8, color='#1E3A8A'),
        text=df_comp['Código'],
        textposition='middle left',
        textfont=dict(size=9),
        name='Directo',
        hovertemplate='<b>%{text}</b><br>Rank Directo: %{y}<extra></extra>'
    ))
    
    # Agregar puntos y etiquetas lado derecho (Indirecto)
    fig.add_trace(go.Scatter(
        x=[1] * n,
        y=df_comp['Rank_Ind'],
        mode='markers+text',
        marker=dict(size=8, color='#DC2626'),
        text=df_comp['Código'],
        textposition='middle right',
        textfont=dict(size=9),
        name='Indirecto',
        hovertemplate='<b>%{text}</b><br>Rank Indirecto: %{y}<extra></extra>'
    ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=f"<b>DESPLAZAMIENTO DE RANKING - {tipo.upper()}</b><br><sup>Directo → Indirecto | Verde=Sube, Rojo=Baja</sup>",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            tickvals=[0, 1],
            ticktext=['<b>DIRECTO</b>', '<b>INDIRECTO</b>'],
            range=[-0.3, 1.3],
            showgrid=False
        ),
        yaxis=dict(
            title='Ranking',
            autorange='reversed',  # Rank 1 arriba
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        height=max(500, n * 18),
        showlegend=False,
        plot_bgcolor='#FFFEF0',
        paper_bgcolor='white'
    )
    
    # Agregar leyenda manual
    fig.add_annotation(x=0.5, y=-0.08, xref='paper', yref='paper',
                      text="🟢 Variable sube en ranking indirecto | 🔴 Variable baja en ranking indirecto",
                      showarrow=False, font=dict(size=10))
    
    return fig

def crear_grafo_influencias(M, codigos, nombres, umbral_pct=20, mostrar_valores=True):
    """
    Crea grafo de influencias estilo LIPSOR
    umbral_pct: porcentaje mínimo del valor máximo para mostrar conexión
    """
    import numpy as np
    
    n = len(codigos)
    
    # Calcular umbral basado en percentil
    valores_no_cero = M[M > 0]
    if len(valores_no_cero) == 0:
        umbral = 0
    else:
        umbral = np.percentile(valores_no_cero, umbral_pct)
    
    max_val = M.max()
    
    # Posiciones de nodos en círculo
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos_x = np.cos(angles) * 10
    pos_y = np.sin(angles) * 10
    
    fig = go.Figure()
    
    # Clasificar y dibujar aristas por intensidad
    for i in range(n):
        for j in range(n):
            if i != j and M[i, j] > umbral:
                valor = M[i, j]
                ratio = valor / max_val if max_val > 0 else 0
                
                # Determinar estilo según intensidad
                if ratio > 0.8:  # Strongest
                    color = 'red'
                    width = 3
                    dash = 'solid'
                elif ratio > 0.6:  # Relatively strong
                    color = 'blue'
                    width = 2.5
                    dash = 'solid'
                elif ratio > 0.4:  # Moderate
                    color = 'lightblue'
                    width = 2
                    dash = 'solid'
                elif ratio > 0.2:  # Weak
                    color = 'gray'
                    width = 1
                    dash = 'solid'
                else:  # Weakest
                    color = 'lightgray'
                    width = 0.5
                    dash = 'dot'
                
                # Calcular punto medio para flecha
                mid_x = (pos_x[i] + pos_x[j]) / 2
                mid_y = (pos_y[i] + pos_y[j]) / 2
                
                # Dibujar línea
                fig.add_trace(go.Scatter(
                    x=[pos_x[i], pos_x[j]],
                    y=[pos_y[i], pos_y[j]],
                    mode='lines',
                    line=dict(color=color, width=width, dash=dash),
                    hoverinfo='skip',
                    showlegend=False
                ))
                
                # Agregar valor en el medio si está habilitado
                if mostrar_valores and ratio > 0.3:
                    fig.add_annotation(
                        x=mid_x, y=mid_y,
                        text=f"{valor:.0f}",
                        showarrow=False,
                        font=dict(size=7, color='gray'),
                        bgcolor='white',
                        opacity=0.7
                    )
    
    # Dibujar nodos
    # Calcular tamaño basado en motricidad
    motricidad = M.sum(axis=1)
    node_sizes = 20 + (motricidad / max(motricidad.max(), 1)) * 30
    
    fig.add_trace(go.Scatter(
        x=pos_x,
        y=pos_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color='white',
            line=dict(color='black', width=2)
        ),
        text=codigos,
        textposition='middle center',
        textfont=dict(size=9, color='black'),
        hovertext=[f"<b>{c}</b><br>{n}<br>Motricidad: {m:.0f}" 
                   for c, n, m in zip(codigos, nombres, motricidad)],
        hoverinfo='text',
        showlegend=False
    ))
    
    # Leyenda de intensidades
    leyenda_y = 0.98
    intensidades = [
        ('red', 'solid', 'Strongest influences'),
        ('blue', 'solid', 'Relatively strong'),
        ('lightblue', 'solid', 'Moderate influences'),
        ('gray', 'solid', 'Weak influences'),
        ('lightgray', 'dot', 'Weakest influences')
    ]
    
    for i, (color, dash, texto) in enumerate(intensidades):
        fig.add_annotation(
            x=0.02, y=leyenda_y - i*0.04,
            xref='paper', yref='paper',
            text=f"<span style='color:{color}'>━━</span> {texto}",
            showarrow=False,
            font=dict(size=9),
            align='left',
            xanchor='left'
        )
    
    fig.update_layout(
        title=dict(
            text="<b>GRAFO DE INFLUENCIAS POTENCIALES DIRECTAS</b>",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-15, 15]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-15, 15]),
        height=700,
        width=900,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def crear_grafico_comparativo_barras(df_resultados, top_n=15):
    """Gráfico de barras comparativo Motricidad vs Dependencia"""
    df = df_resultados.nlargest(top_n, 'Motricidad').copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Motricidad',
        x=df['Código'],
        y=df['Motricidad'],
        marker_color='#FF6B6B',
        hovertext=df['Variable'],
        hovertemplate='<b>%{hovertext}</b><br>Motricidad: %{y:.1f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Dependencia',
        x=df['Código'],
        y=df['Dependencia'],
        marker_color='#4ECDC4',
        hovertext=df['Variable'],
        hovertemplate='<b>%{hovertext}</b><br>Dependencia: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Comparativo Motricidad vs Dependencia (Top {top_n})',
        barmode='group',
        height=500,
        xaxis_title='Variable',
        yaxis_title='Valor'
    )
    
    return fig

# ============================================================
# GENERACIÓN DE INFORME
# ============================================================

def generar_informe_excel(res_directo, res_indirecto, nombres, codigos, M, nombre_proyecto):
    """Genera informe completo en Excel con análisis DIRECTO e INDIRECTO"""
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Resumen ejecutivo
        resumen = pd.DataFrame({
            'Parámetro': [
                'Fecha de análisis', 'Nombre del proyecto', 'Total de variables',
                'Alpha (α)', 'K (profundidad)',
                '--- ANÁLISIS DIRECTO ---', '',
                'Motrices (Directo)', 'Enlace (Directo)', 'Resultado (Directo)', 'Autónomas (Directo)',
                '--- ANÁLISIS INDIRECTO ---', '',
                'Determinantes (Indirecto)', 'Clave (Indirecto)', 'Resultado (Indirecto)', 'Autónomas (Indirecto)',
                '---', 'Densidad de la matriz (%)'
            ],
            'Valor': [
                datetime.now().strftime('%Y-%m-%d %H:%M'), nombre_proyecto, len(nombres),
                res_indirecto['alpha'], res_indirecto['K'],
                '', '',
                sum(c == 'Motrices' for c in res_directo['clasificacion']),
                sum(c == 'Enlace' for c in res_directo['clasificacion']),
                sum(c == 'Resultado' for c in res_directo['clasificacion']),
                sum(c == 'Autónomas' for c in res_directo['clasificacion']),
                '', '',
                sum(c == 'Determinantes' for c in res_indirecto['clasificacion']),
                sum(c == 'Clave' for c in res_indirecto['clasificacion']),
                sum(c == 'Variables resultado' for c in res_indirecto['clasificacion']),
                sum(c == 'Autónomas' for c in res_indirecto['clasificacion']),
                '', round((M != 0).sum() / M.size * 100, 1)
            ]
        })
        resumen.to_excel(writer, sheet_name='Resumen Ejecutivo', index=False)
        
        # Ranking DIRECTO
        res_directo['df_resultados'].to_excel(writer, sheet_name='Ranking DIRECTO', index=False)
        
        # Ranking INDIRECTO
        res_indirecto['df_resultados'].to_excel(writer, sheet_name='Ranking INDIRECTO', index=False)
        
        # Comparación Directo vs Indirecto
        df_comp = comparar_directo_indirecto(res_directo['df_resultados'], res_indirecto['df_resultados'])
        df_comp.to_excel(writer, sheet_name='Comparación Dir-Indir', index=False)
        
        # Variables por cuadrante DIRECTO
        for cuadrante in ['Motrices', 'Enlace', 'Resultado', 'Autónomas']:
            df_cuad = res_directo['df_resultados'][res_directo['df_resultados']['Clasificación'] == cuadrante]
            if len(df_cuad) > 0:
                df_cuad.to_excel(writer, sheet_name=f'Dir_{cuadrante[:10]}', index=False)
        
        # Variables por cuadrante INDIRECTO
        for cuadrante in ['Determinantes', 'Clave', 'Variables resultado', 'Autónomas']:
            df_cuad = res_indirecto['df_resultados'][res_indirecto['df_resultados']['Clasificación'] == cuadrante]
            if len(df_cuad) > 0:
                nombre_hoja = f'Ind_{cuadrante[:10]}'
                df_cuad.to_excel(writer, sheet_name=nombre_hoja, index=False)
        
        # Matriz MID Original
        pd.DataFrame(M, index=codigos, columns=codigos).to_excel(writer, sheet_name='Matriz MID Original')
        
        # Matriz MIDI
        pd.DataFrame(res_indirecto['MIDI'], index=codigos, columns=codigos).to_excel(writer, sheet_name='Matriz MIDI')
        
        # Relaciones fuertes
        relaciones = identificar_relaciones_fuertes(M, codigos, umbral=2)
        if relaciones:
            pd.DataFrame(relaciones).to_excel(writer, sheet_name='Relaciones Fuertes', index=False)
        
        # Diccionario de códigos
        pd.DataFrame({'Código': codigos, 'Variable': nombres}).to_excel(writer, sheet_name='Diccionario', index=False)
        
        # Análisis estratégico
        df_estrategico = res_indirecto['df_resultados'].copy()
        df_estrategico['Valor_Estratégico'] = df_estrategico['Motricidad'] + df_estrategico['Dependencia']
        df_estrategico['Índice_Inestabilidad'] = df_estrategico['Motricidad'] * df_estrategico['Dependencia']
        df_estrategico = df_estrategico.sort_values('Valor_Estratégico', ascending=False)
        df_estrategico.to_excel(writer, sheet_name='Análisis Estratégico', index=False)
    
    buffer.seek(0)
    return buffer

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

st.markdown('<div class="main-header">🎯 MICMAC PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis Estructural - Metodología Godet (LIPSOR)</div>', unsafe_allow_html=True)

# Inicializar session state
for key in ['matriz_procesada', 'nombres_variables', 'codigos_variables', 'mapeo_codigos', 
            'resultados_directo', 'resultados_indirecto', 'hojas_excel', 'M_original']:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("1. Cargar Matriz")
    uploaded_file = st.file_uploader(
        "Subir archivo Excel",
        type=['xlsx', 'xls'],
        help="Soporta archivos con múltiples hojas"
    )
    
    hoja_seleccionada = None
    if uploaded_file is not None:
        try:
            xl_temp = pd.ExcelFile(uploaded_file)
            st.session_state.hojas_excel = xl_temp.sheet_names
            uploaded_file.seek(0)
        except:
            st.session_state.hojas_excel = None
        
        if st.session_state.hojas_excel and len(st.session_state.hojas_excel) > 1:
            default_idx = 0
            for i, hoja in enumerate(st.session_state.hojas_excel):
                if 'MID' in hoja.upper() or 'MATRIZ' in hoja.upper():
                    default_idx = i
                    break
            
            hoja_seleccionada = st.selectbox(
                "📑 Seleccionar hoja:",
                st.session_state.hojas_excel,
                index=default_idx
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
    usar_codigos = st.checkbox("Usar códigos cortos", value=True)
    max_chars_codigo = st.slider("Longitud máx. códigos", min_value=4, max_value=15, value=10)
    tamaño_fuente = st.slider("Tamaño fuente", min_value=8, max_value=16, value=10)

# Pestañas principales - AHORA CON PESTAÑA DIRECTO
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📋 Datos",
    "📊 Análisis DIRECTO",
    "📈 Análisis INDIRECTO",
    "🔄 Comparación",
    "🎯 Eje Estratégico",
    "🔬 Avanzado",
    "📑 Informe",
    "📥 Exportar"
])

# ============================================================
# TAB 1: DATOS
# ============================================================
with tab1:
    st.header("📋 Carga y Visualización de Datos")
    
    if uploaded_file is not None:
        df_procesado, nombres, hojas, mensaje = procesar_archivo_excel(uploaded_file, hoja_seleccionada)
        
        if df_procesado is not None:
            st.success(mensaje)
            
            codigos, mapeo = generar_codigos_y_mapeo(nombres, max_chars=max_chars_codigo)
            
            st.session_state.matriz_procesada = df_procesado
            st.session_state.nombres_variables = nombres
            st.session_state.codigos_variables = codigos
            st.session_state.mapeo_codigos = mapeo
            st.session_state.M_original = df_procesado.values.astype(float)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Variables", len(nombres))
            col2.metric("Tamaño", f"{len(nombres)}x{len(nombres)}")
            
            M_temp = df_procesado.values.astype(float)
            densidad = (M_temp != 0).sum() / M_temp.size * 100
            col3.metric("Densidad", f"{densidad:.1f}%")
            
            st.subheader("🏷️ Tabla de Variables y Códigos")
            df_codigos = pd.DataFrame({
                'Código': codigos,
                'Variable (nombre completo)': nombres
            })
            st.dataframe(df_codigos, use_container_width=True, height=300)
            
            st.subheader("📊 Vista previa de la matriz MID")
            st.dataframe(df_procesado, use_container_width=True, height=400)
        else:
            st.error(f"❌ {mensaje}")
    else:
        st.info("👆 Sube un archivo Excel con tu matriz de influencias")

# ============================================================
# TAB 2: ANÁLISIS DIRECTO
# ============================================================
with tab2:
    st.header("📊 Análisis DIRECTO (MID)")
    
    st.markdown("""
    <div class="info-box">
    <b>Análisis DIRECTO:</b> Usa la matriz original M sin transformaciones.<br>
    • <b>Motricidad (M):</b> Suma de la fila = influencia que EJERCE la variable<br>
    • <b>Dependencia (D):</b> Suma de la columna = influencia que RECIBE la variable<br>
    • Valores en <b>escala original</b> (enteros pequeños, típicamente 0-100)
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.matriz_procesada is not None:
        M = st.session_state.M_original
        nombres = st.session_state.nombres_variables
        codigos = st.session_state.codigos_variables
        n = M.shape[0]
        
        # Calcular análisis DIRECTO
        motricidad_dir, dependencia_dir = calcular_motricidad_dependencia_directa(M)
        clasificacion_dir, med_mot_dir, med_dep_dir = clasificar_variables_directas(motricidad_dir, dependencia_dir)
        
        df_directo = pd.DataFrame({
            'Código': codigos,
            'Variable': nombres,
            'Motricidad': np.round(motricidad_dir, 2),
            'Dependencia': np.round(dependencia_dir, 2),
            'Clasificación': clasificacion_dir
        })
        df_directo['Ranking_M'] = df_directo['Motricidad'].rank(ascending=False).astype(int)
        df_directo = df_directo.sort_values('Motricidad', ascending=False)
        
        st.session_state.resultados_directo = {
            'df_resultados': df_directo,
            'motricidad': motricidad_dir,
            'dependencia': dependencia_dir,
            'clasificacion': clasificacion_dir,
            'med_mot': med_mot_dir,
            'med_dep': med_dep_dir
        }
        
        # Métricas
        st.subheader("📈 Distribución de Variables (Análisis DIRECTO)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total", n)
        col2.metric("🔴 Motrices", sum(c == 'Motrices' for c in clasificacion_dir))
        col3.metric("🔵 Enlace", sum(c == 'Enlace' for c in clasificacion_dir))
        col4.metric("💧 Resultado", sum(c == 'Resultado' for c in clasificacion_dir))
        col5.metric("🟠 Autónomas", sum(c == 'Autónomas' for c in clasificacion_dir))
        
        # Ranking DIRECTO
        st.subheader("🏆 Ranking de Variables (DIRECTO)")
        
        def color_clasif_dir(val):
            colors = {
                'Motrices': 'background-color: #ffcccc',
                'Enlace': 'background-color: #cce5ff',
                'Resultado': 'background-color: #cceeff',
                'Autónomas': 'background-color: #fff3cd'
            }
            return colors.get(val, '')
        
        st.dataframe(
            df_directo.style.applymap(color_clasif_dir, subset=['Clasificación']),
            use_container_width=True, height=400
        )
        
        # Gráfico de subsistemas DIRECTO - Versión HD
        st.subheader("🗺️ Plano de Subsistemas (DIRECTO)")
        
        fig_dir = crear_grafico_subsistemas(
            df_directo, med_mot_dir, med_dep_dir, motricidad_dir, dependencia_dir,
            titulo="Análisis DIRECTO",
            usar_codigos=usar_codigos, mostrar_etiquetas=mostrar_etiquetas, 
            tamaño_fuente=tamaño_fuente, M_original=M
        )
        mostrar_grafico_con_descargas(fig_dir, "subsistemas_directo", "tab2")
        
        # Distribución
        st.subheader("📊 Distribución por Cuadrantes (DIRECTO)")
        fig_pie_dir = crear_grafico_distribucion_cuadrantes(clasificacion_dir, "Distribución DIRECTA")
        mostrar_grafico_con_descargas(fig_pie_dir, "distribucion_directo", "tab2_pie")
        
        # Exportar CSV directo
        st.subheader("📥 Exportar Análisis DIRECTO")
        csv_directo = df_directo.to_csv(index=False)
        st.download_button(
            "📥 Descargar Ranking DIRECTO (CSV)",
            csv_directo,
            "ranking_DIRECTO.csv",
            "text/csv",
            key="csv_directo"
        )
        
    else:
        st.warning("⚠️ Primero carga una matriz en 'Datos'")

# ============================================================
# TAB 3: ANÁLISIS INDIRECTO
# ============================================================
with tab3:
    st.header("📈 Análisis INDIRECTO (MIDI)")
    
    st.markdown("""
    <div class="info-box">
    <b>Análisis INDIRECTO:</b> Usa la matriz potenciada MIDI = M + αM² + α²M³ + ...<br>
    • Revela <b>relaciones ocultas</b> y <b>estructurales</b> del sistema<br>
    • Valores en <b>escala amplificada</b> (pueden ser millones)<br>
    • Muestra cómo las influencias se <b>propagan</b> a través del sistema
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.matriz_procesada is not None:
        M = st.session_state.M_original
        nombres = st.session_state.nombres_variables
        codigos = st.session_state.codigos_variables
        n = M.shape[0]
        
        if K_auto:
            K_usado = detectar_convergencia(M)
            st.success(f"🔍 K óptimo detectado: **{K_usado}**")
        else:
            K_usado = K_manual
        
        # Calcular MIDI
        MIDI = calcular_midi(M, alpha=alpha, K=K_usado)
        motricidad_ind, dependencia_ind = calcular_motricidad_dependencia(MIDI)
        clasificacion_ind, med_mot_ind, med_dep_ind = clasificar_variables(motricidad_ind, dependencia_ind)
        
        df_indirecto = pd.DataFrame({
            'Código': codigos,
            'Variable': nombres,
            'Motricidad': np.round(motricidad_ind, 2),
            'Dependencia': np.round(dependencia_ind, 2),
            'Clasificación': clasificacion_ind
        })
        df_indirecto['Ranking_M'] = df_indirecto['Motricidad'].rank(ascending=False).astype(int)
        df_indirecto = df_indirecto.sort_values('Motricidad', ascending=False)
        
        st.session_state.resultados_indirecto = {
            'df_resultados': df_indirecto,
            'MIDI': MIDI,
            'motricidad': motricidad_ind,
            'dependencia': dependencia_ind,
            'clasificacion': clasificacion_ind,
            'med_mot': med_mot_ind,
            'med_dep': med_dep_ind,
            'alpha': alpha,
            'K': K_usado
        }
        
        # Métricas
        st.subheader(f"📈 Distribución de Variables (INDIRECTO, α={alpha}, K={K_usado})")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total", n)
        col2.metric("🔴 Determinantes", sum(c == 'Determinantes' for c in clasificacion_ind))
        col3.metric("🔵 Clave", sum(c == 'Clave' for c in clasificacion_ind))
        col4.metric("💧 Resultado", sum(c == 'Variables resultado' for c in clasificacion_ind))
        col5.metric("🟠 Autónomas", sum(c == 'Autónomas' for c in clasificacion_ind))
        
        # Ranking INDIRECTO
        st.subheader("🏆 Ranking de Variables (INDIRECTO)")
        
        def color_clasif_ind(val):
            colors = {
                'Determinantes': 'background-color: #ffcccc',
                'Clave': 'background-color: #cce5ff',
                'Variables resultado': 'background-color: #cceeff',
                'Autónomas': 'background-color: #fff3cd'
            }
            return colors.get(val, '')
        
        st.dataframe(
            df_indirecto.style.applymap(color_clasif_ind, subset=['Clasificación']),
            use_container_width=True, height=400
        )
        
        # Gráfico de subsistemas INDIRECTO - Versión HD
        st.subheader("🗺️ Plano de Subsistemas (INDIRECTO)")
        
        fig_ind = crear_grafico_subsistemas(
            df_indirecto, med_mot_ind, med_dep_ind, motricidad_ind, dependencia_ind,
            titulo="Análisis INDIRECTO",
            usar_codigos=usar_codigos, mostrar_etiquetas=mostrar_etiquetas, 
            tamaño_fuente=tamaño_fuente, M_original=M
        )
        mostrar_grafico_con_descargas(fig_ind, "subsistemas_indirecto", "tab3")
        
        # Distribución
        st.subheader("📊 Distribución por Cuadrantes (INDIRECTO)")
        fig_pie_ind = crear_grafico_distribucion_cuadrantes(clasificacion_ind, "Distribución INDIRECTA")
        mostrar_grafico_con_descargas(fig_pie_ind, "distribucion_indirecto", "tab3_pie")
        
        # Matriz MIDI
        st.subheader("🔢 Matriz MIDI")
        etiquetas = codigos if usar_codigos else [truncar_texto(n, 20) for n in nombres]
        fig_midi = go.Figure(data=go.Heatmap(
            z=MIDI, x=etiquetas, y=etiquetas, colorscale='Blues',
            hovertemplate='%{x} → %{y}<br>Valor: %{z:.1f}<extra></extra>'
        ))
        fig_midi.update_layout(height=600, title=f"MIDI (α={alpha}, K={K_usado})")
        mostrar_grafico_con_descargas(fig_midi, "matriz_midi", "tab3_midi")
        
        # Exportar
        st.subheader("📥 Exportar Análisis INDIRECTO")
        csv_indirecto = df_indirecto.to_csv(index=False)
        st.download_button(
            "📥 Descargar Ranking INDIRECTO (CSV)",
            csv_indirecto,
            "ranking_INDIRECTO.csv",
            "text/csv",
            key="csv_indirecto"
        )
        
    else:
        st.warning("⚠️ Primero carga una matriz en 'Datos'")

# ============================================================
# TAB 4: COMPARACIÓN DIRECTO vs INDIRECTO
# ============================================================
with tab4:
    st.header("🔄 Comparación: DIRECTO vs INDIRECTO")
    
    st.markdown("""
    <div class="info-box">
    <b>¿Por qué comparar?</b><br>
    • El análisis <b>DIRECTO</b> muestra relaciones <b>inmediatas y visibles</b><br>
    • El análisis <b>INDIRECTO</b> revela relaciones <b>ocultas y estructurales</b><br>
    • Las variables que <b>cambian de cuadrante</b> son especialmente interesantes
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.resultados_directo is not None and st.session_state.resultados_indirecto is not None:
        res_dir = st.session_state.resultados_directo
        res_ind = st.session_state.resultados_indirecto
        
        # Comparación de distribuciones
        st.subheader("📊 Comparación de Distribuciones")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**DIRECTO**")
            st.write(f"- 🔴 Motrices: {sum(c == 'Motrices' for c in res_dir['clasificacion'])}")
            st.write(f"- 🔵 Enlace: {sum(c == 'Enlace' for c in res_dir['clasificacion'])}")
            st.write(f"- 💧 Resultado: {sum(c == 'Resultado' for c in res_dir['clasificacion'])}")
            st.write(f"- 🟠 Autónomas: {sum(c == 'Autónomas' for c in res_dir['clasificacion'])}")
        
        with col2:
            st.markdown("**INDIRECTO**")
            st.write(f"- 🔴 Determinantes: {sum(c == 'Determinantes' for c in res_ind['clasificacion'])}")
            st.write(f"- 🔵 Clave: {sum(c == 'Clave' for c in res_ind['clasificacion'])}")
            st.write(f"- 💧 Resultado: {sum(c == 'Variables resultado' for c in res_ind['clasificacion'])}")
            st.write(f"- 🟠 Autónomas: {sum(c == 'Autónomas' for c in res_ind['clasificacion'])}")
        
        # Tabla de comparación
        st.subheader("📋 Tabla Comparativa")
        df_comp = comparar_directo_indirecto(res_dir['df_resultados'], res_ind['df_resultados'])
        
        # Resaltar cambios
        n_cambios = df_comp['Cambio'].sum()
        if n_cambios > 0:
            st.warning(f"⚠️ **{n_cambios} variables** cambiaron de cuadrante entre análisis directo e indirecto")
        else:
            st.success("✅ Todas las variables mantienen su clasificación en ambos análisis")
        
        def highlight_cambio(row):
            if row['Cambio']:
                return ['background-color: #ffe6e6'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            df_comp.style.apply(highlight_cambio, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Variables que cambiaron
        if n_cambios > 0:
            st.subheader("🔄 Variables que Cambiaron de Cuadrante")
            df_cambios = df_comp[df_comp['Cambio']]
            for _, row in df_cambios.iterrows():
                st.markdown(f"- **{row['Código']}** ({row['Variable'][:40]}...): {row['Clasif_Directa']} → {row['Clasif_Indirecta']}")
        
        # Gráficos lado a lado
        st.subheader("📈 Planos de Subsistemas Comparados")
        col1, col2 = st.columns(2)
        
        M = st.session_state.M_original
        
        with col1:
            fig_dir = crear_grafico_subsistemas(
                res_dir['df_resultados'], res_dir['med_mot'], res_dir['med_dep'],
                res_dir['motricidad'], res_dir['dependencia'],
                titulo="DIRECTO",
                usar_codigos=usar_codigos, mostrar_etiquetas=mostrar_etiquetas, 
                tamaño_fuente=tamaño_fuente-2, M_original=M
            )
            fig_dir.update_layout(height=500, margin=dict(r=50))
            st.plotly_chart(fig_dir, use_container_width=True, key="comp_dir")
        
        with col2:
            fig_ind = crear_grafico_subsistemas(
                res_ind['df_resultados'], res_ind['med_mot'], res_ind['med_dep'],
                res_ind['motricidad'], res_ind['dependencia'],
                titulo="INDIRECTO",
                usar_codigos=usar_codigos, mostrar_etiquetas=mostrar_etiquetas, 
                tamaño_fuente=tamaño_fuente-2, M_original=M
            )
            fig_ind.update_layout(height=500, margin=dict(r=50))
            st.plotly_chart(fig_ind, use_container_width=True, key="comp_ind")
        
        # Exportar comparación
        csv_comp = df_comp.to_csv(index=False)
        st.download_button(
            "📥 Descargar Comparación (CSV)",
            csv_comp,
            "comparacion_directo_indirecto.csv",
            "text/csv",
            key="csv_comp"
        )
        
        st.divider()
        
        # === GRÁFICOS DE DESPLAZAMIENTO DE RANKING (estilo LIPSOR) ===
        st.subheader("📊 Desplazamiento de Rankings (Directo → Indirecto)")
        st.markdown("""
        <div class="info-box">
        Estos gráficos muestran cómo cambia la posición de cada variable al pasar del análisis 
        <b>directo</b> al <b>indirecto</b>. Las líneas <span style="color:green"><b>verdes</b></span> 
        indican variables que <b>suben</b> en el ranking indirecto, las <span style="color:red"><b>rojas</b></span> 
        indican que <b>bajan</b>.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ranking por MOTRICIDAD**")
            fig_desp_mot = crear_grafico_desplazamiento_ranking(
                res_dir['df_resultados'], 
                res_ind['df_resultados'], 
                tipo='Motricidad'
            )
            st.plotly_chart(fig_desp_mot, use_container_width=True, key="desp_mot")
        
        with col2:
            st.markdown("**Ranking por DEPENDENCIA**")
            fig_desp_dep = crear_grafico_desplazamiento_ranking(
                res_dir['df_resultados'], 
                res_ind['df_resultados'], 
                tipo='Dependencia'
            )
            st.plotly_chart(fig_desp_dep, use_container_width=True, key="desp_dep")
        
    else:
        st.warning("⚠️ Ejecuta primero los análisis DIRECTO e INDIRECTO")

# ============================================================
# TAB 5: EJE ESTRATÉGICO
# ============================================================
with tab5:
    st.header("🎯 Eje Estratégico")
    
    if st.session_state.resultados_indirecto is not None:
        res = st.session_state.resultados_indirecto
        df_res = res['df_resultados'].copy()
        nombres = st.session_state.nombres_variables
        codigos = st.session_state.codigos_variables
        
        st.markdown("""
        <div class="info-box">
        <b>Eje Estratégico:</b> La diagonal donde Motricidad = Dependencia.<br>
        Variables cerca de esta línea tienen <b>máximo valor estratégico</b>.
        </div>
        """, unsafe_allow_html=True)
        
        df_res['Valor_Estrategico'] = df_res['Motricidad'] + df_res['Dependencia']
        df_res['Distancia_Eje'] = np.abs(df_res['Motricidad'] - df_res['Dependencia'])
        
        etiquetas = df_res['Código'].tolist() if usar_codigos else df_res['Variable'].tolist()
        hover_text = [f"<b>{c}</b><br>{v}" for c, v in zip(df_res['Código'], df_res['Variable'])]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_res['Dependencia'], y=df_res['Motricidad'],
            mode='markers+text' if mostrar_etiquetas else 'markers',
            text=etiquetas if mostrar_etiquetas else None,
            textposition='top center',
            textfont=dict(size=tamaño_fuente),
            marker=dict(size=12, color=df_res['Valor_Estrategico'], colorscale='YlOrRd',
                showscale=True, colorbar=dict(title="Valor<br>Estratégico")),
            hovertemplate="%{customdata}<br>Mot: %{y:.1f}<br>Dep: %{x:.1f}<extra></extra>",
            customdata=hover_text
        ))
        
        max_val = max(max(res['motricidad']), max(res['dependencia'])) * 1.1
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', name='Eje Estratégico',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(title="Eje Estratégico (Análisis Indirecto)", height=600)
        mostrar_grafico_con_descargas(fig, "eje_estrategico", "tab5")
        
        st.subheader("🏆 Top 10 Variables Estratégicas")
        top10 = df_res.nlargest(10, 'Valor_Estrategico')[
            ['Código', 'Variable', 'Motricidad', 'Dependencia', 'Valor_Estrategico', 'Clasificación']
        ]
        st.dataframe(top10, use_container_width=True)
        
    else:
        st.warning("⚠️ Ejecuta primero el análisis")

# ============================================================
# TAB 6: ANÁLISIS AVANZADO
# ============================================================
with tab6:
    st.header("🔬 Análisis Avanzado")
    
    if st.session_state.resultados_indirecto is not None and st.session_state.M_original is not None:
        res = st.session_state.resultados_indirecto
        M = st.session_state.M_original
        nombres = st.session_state.nombres_variables
        codigos = st.session_state.codigos_variables
        
        # === GRAFO DE INFLUENCIAS (estilo LIPSOR) ===
        st.subheader("🕸️ Grafo de Influencias Potenciales Directas")
        st.markdown("""
        <div class="info-box">
        Visualización de la red de influencias entre variables. El grosor y color de las líneas 
        indica la intensidad de la influencia:
        <span style="color:red"><b>━━ Muy fuerte</b></span> | 
        <span style="color:blue"><b>━━ Fuerte</b></span> | 
        <span style="color:lightblue">━━ Moderada</span> | 
        <span style="color:gray">━━ Débil</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            umbral_grafo = st.slider(
                "Umbral de visualización (%)", 
                min_value=0, max_value=80, value=30,
                help="Porcentaje mínimo del valor máximo para mostrar una conexión"
            )
            mostrar_valores_grafo = st.checkbox("Mostrar valores en aristas", value=False)
        
        fig_grafo = crear_grafo_influencias(M, codigos, nombres, umbral_pct=umbral_grafo, mostrar_valores=mostrar_valores_grafo)
        st.plotly_chart(fig_grafo, use_container_width=True, key="grafo_inf")
        
        st.divider()
        
        st.subheader("🕸️ Red de Influencias Fuertes (Vista alternativa)")
        umbral_red = st.slider("Umbral de intensidad", min_value=1, max_value=3, value=2, key="umbral_red")
        fig_red = crear_grafico_red_influencias(M, nombres, codigos, umbral=umbral_red, usar_codigos=usar_codigos)
        mostrar_grafico_con_descargas(fig_red, "red_influencias", "tab6_red")
        
        st.divider()
        
        st.subheader("📊 Comparativo Motricidad vs Dependencia")
        fig_comp = crear_grafico_comparativo_barras(res['df_resultados'], top_n=15)
        mostrar_grafico_con_descargas(fig_comp, "comparativo_mot_dep", "tab6_comp")
        
        st.divider()
        
        st.subheader("📈 Estabilidad del Ranking según K")
        fig_estab = crear_grafico_estabilidad(M, nombres, codigos, alpha=res['alpha'], usar_codigos=usar_codigos)
        mostrar_grafico_con_descargas(fig_estab, "estabilidad_ranking", "tab6_estab")
        
        st.divider()
        
        st.subheader("💪 Top Relaciones de Influencia")
        fig_rel = crear_grafico_relaciones_fuertes(M, nombres, codigos, top_n=20, usar_codigos=usar_codigos)
        mostrar_grafico_con_descargas(fig_rel, "relaciones_fuertes", "tab6_rel")
        
    else:
        st.warning("⚠️ Ejecuta primero el análisis")

# ============================================================
# TAB 7: INFORME
# ============================================================
with tab7:
    st.header("📑 Generación de Informe")
    
    if st.session_state.resultados_directo is not None and st.session_state.resultados_indirecto is not None:
        res_dir = st.session_state.resultados_directo
        res_ind = st.session_state.resultados_indirecto
        nombres = st.session_state.nombres_variables
        codigos = st.session_state.codigos_variables
        M = st.session_state.M_original
        
        nombre_proyecto = st.text_input("Nombre del proyecto", value="Analisis_MICMAC")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Informe Excel Completo")
            st.markdown("""
            **Incluye:**
            - Resumen ejecutivo
            - Ranking DIRECTO (valores originales)
            - Ranking INDIRECTO (valores MIDI)
            - Comparación Directo vs Indirecto
            - Variables por cuadrante (ambos análisis)
            - Matriz MID Original
            - Matriz MIDI
            - Relaciones fuertes
            - Diccionario de códigos
            """)
            
            buffer = generar_informe_excel(res_dir, res_ind, nombres, codigos, M, nombre_proyecto)
            st.download_button(
                label="📥 Descargar Informe Excel Completo",
                data=buffer,
                file_name=f"{nombre_proyecto}_informe_completo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        
        with col2:
            st.subheader("📋 Resumen del Análisis")
            st.markdown(f"""
            **Proyecto:** {nombre_proyecto}
            
            **Parámetros:**
            - Variables: {len(nombres)}
            - Alpha (α): {res_ind['alpha']}
            - K: {res_ind['K']}
            
            **DIRECTO:**
            - 🔴 Motrices: {sum(c == 'Motrices' for c in res_dir['clasificacion'])}
            - 🔵 Enlace: {sum(c == 'Enlace' for c in res_dir['clasificacion'])}
            - 💧 Resultado: {sum(c == 'Resultado' for c in res_dir['clasificacion'])}
            - 🟠 Autónomas: {sum(c == 'Autónomas' for c in res_dir['clasificacion'])}
            
            **INDIRECTO:**
            - 🔴 Determinantes: {sum(c == 'Determinantes' for c in res_ind['clasificacion'])}
            - 🔵 Clave: {sum(c == 'Clave' for c in res_ind['clasificacion'])}
            - 💧 Resultado: {sum(c == 'Variables resultado' for c in res_ind['clasificacion'])}
            - 🟠 Autónomas: {sum(c == 'Autónomas' for c in res_ind['clasificacion'])}
            """)
        
    else:
        st.warning("⚠️ Ejecuta primero ambos análisis")

# ============================================================
# TAB 8: EXPORTAR
# ============================================================
with tab8:
    st.header("📥 Exportar Resultados Individuales")
    
    if st.session_state.resultados_directo is not None and st.session_state.resultados_indirecto is not None:
        res_dir = st.session_state.resultados_directo
        res_ind = st.session_state.resultados_indirecto
        codigos = st.session_state.codigos_variables
        nombres = st.session_state.nombres_variables
        M = st.session_state.M_original
        
        st.subheader("📋 Exportaciones CSV")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Análisis DIRECTO:**")
            csv_dir = res_dir['df_resultados'].to_csv(index=False)
            st.download_button("📥 Ranking DIRECTO (CSV)", csv_dir, "ranking_DIRECTO.csv", "text/csv", key="exp_dir")
        
        with col2:
            st.markdown("**Análisis INDIRECTO:**")
            csv_ind = res_ind['df_resultados'].to_csv(index=False)
            st.download_button("📥 Ranking INDIRECTO (CSV)", csv_ind, "ranking_INDIRECTO.csv", "text/csv", key="exp_ind")
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Matriz MID Original:**")
            df_mid = pd.DataFrame(M, index=codigos, columns=codigos)
            csv_mid = df_mid.to_csv()
            st.download_button("📥 Matriz MID (CSV)", csv_mid, "matriz_MID.csv", "text/csv", key="exp_mid")
        
        with col2:
            st.markdown("**Matriz MIDI:**")
            df_midi = pd.DataFrame(res_ind['MIDI'], index=codigos, columns=codigos)
            csv_midi = df_midi.to_csv()
            st.download_button("📥 Matriz MIDI (CSV)", csv_midi, "matriz_MIDI.csv", "text/csv", key="exp_midi")
        
        with col3:
            st.markdown("**Diccionario:**")
            df_dict = pd.DataFrame({'Código': codigos, 'Variable': nombres})
            csv_dict = df_dict.to_csv(index=False)
            st.download_button("📥 Diccionario (CSV)", csv_dict, "diccionario.csv", "text/csv", key="exp_dict")
        
        st.divider()
        
        st.markdown("**Comparación Directo vs Indirecto:**")
        df_comp = comparar_directo_indirecto(res_dir['df_resultados'], res_ind['df_resultados'])
        csv_comp = df_comp.to_csv(index=False)
        st.download_button("📥 Comparación Dir-Ind (CSV)", csv_comp, "comparacion_dir_ind.csv", "text/csv", key="exp_comp")
        
    else:
        st.warning("⚠️ Ejecuta primero los análisis")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
<b>MICMAC PRO v5.5</b> | Metodología Godet (umbral = Media Aritmética) | JETLEX Strategic Consulting by Horacio Martin Pratto Chiarella | 2025
</div>
""", unsafe_allow_html=True)
