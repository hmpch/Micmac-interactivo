# 📚 Metodología MICMAC - Documentación Técnica Detallada

## Índice

1. [Fundamentos Teóricos](#fundamentos-teóricos)
2. [Algoritmo Matemático](#algoritmo-matemático)
3. [Interpretación de Resultados](#interpretación-de-resultados)
4. [Configuración de Parámetros](#configuración-de-parámetros)
5. [Validación y Confiabilidad](#validación-y-confiabilidad)
6. [Casos de Uso](#casos-de-uso)
7. [Limitaciones y Consideraciones](#limitaciones-y-consideraciones)

---

## 1. Fundamentos Teóricos

### 1.1 Origen del Método MICMAC

El método **MICMAC** (Matriz de Impactos Cruzados - Multiplicación Aplicada a una Clasificación) fue desarrollado por **Michel Godet** en 1990 en el contexto de la **prospectiva estratégica francesa**.

#### Contexto Histórico

- **Década de 1970:** Primeras técnicas de análisis estructural (Duperrin & Godet, 1973)
- **1990:** Formalización del método MICMAC en "From Anticipation to Action"
- **2000s:** Digitalización y desarrollo del software propietario
- **2025:** Esta implementación open-source

#### Objetivo Principal

Identificar **variables clave** en sistemas complejos mediante el análisis sistemático de influencias **directas** e **indirectas** entre elementos del sistema.

### 1.2 Principios Fundamentales

#### Principio 1: Análisis Sistémico

> "El todo es más que la suma de las partes"

El método MICMAC considera que:
- Las variables están **interconectadas**
- Existen **efectos en cascada** y **retroalimentaciones**
- Las influencias **indirectas** pueden ser tan o más importantes que las directas

#### Principio 2: Propagación de Influencias

La influencia de una variable A sobre una variable C puede ocurrir:
- **Directamente:** A → C
- **Indirectamente (orden 2):** A → B → C
- **Indirectamente (orden 3):** A → B → D → C
- **Indirectamente (orden K):** Cadenas de K pasos

#### Principio 3: Atenuación Exponencial

Las influencias indirectas se **atenúan** con la distancia:
- Una influencia de 2 pasos tiene menos peso que una directa
- Se usa factor α para controlar esta atenuación
- Típicamente: α ∈ [0.3, 0.8]

---

## 2. Algoritmo Matemático

### 2.1 Notación Matemática

Sea:
- **n:** Número de variables del sistema
- **M:** Matriz de influencias directas (n×n)
- **M[i,j]:** Influencia directa de variable i sobre variable j
- **α:** Factor de atenuación (0 < α ≤ 1)
- **K:** Profundidad máxima de análisis

### 2.2 Construcción de la Matriz Directa (M)

#### Escala de Valoración Clásica

| Valor | Significado |
|-------|-------------|
| 0 | Sin influencia |
| 1 | Influencia débil |
| 2 | Influencia moderada |
| 3 | Influencia fuerte |
| 4 | Influencia muy fuerte |

#### Escala Alternativa (P = Potencial)

| Valor | Significado |
|-------|-------------|
| 0 | Sin influencia |
| 1 | Influencia débil |
| 2 | Influencia media |
| 3 | Influencia fuerte |
| P | Influencia potencial (futura) |

#### Propiedades de M

1. **Matriz cuadrada:** M ∈ ℝⁿˣⁿ
2. **Diagonal nula:** M[i,i] = 0 ∀i (una variable no se influye a sí misma)
3. **No necesariamente simétrica:** M[i,j] ≠ M[j,i] en general
4. **Valores no negativos:** M[i,j] ≥ 0 ∀i,j

### 2.3 Cálculo de la Matriz Total

#### Fórmula General

```
M_total = M + α·M² + α²·M³ + α³·M⁴ + ... + α^(K-1)·M^K
```

En notación compacta:

```
M_total = Σ(k=1 to K) α^(k-1)·M^k
```

#### Interpretación Física

- **M¹ = M:** Influencias directas (1 paso)
- **M²:** Influencias indirectas de orden 2 (2 pasos)
- **M³:** Influencias indirectas de orden 3 (3 pasos)
- **M^K:** Influencias indirectas de orden K (K pasos)

#### Ejemplo Numérico

Supongamos un sistema simple con 3 variables y α = 0.5, K = 2:

```
M = [0  2  1]
    [1  0  3]
    [0  1  0]

M² = M·M = [2  1  6]
           [0  5  1]
           [1  0  3]

M_total = M + 0.5·M² = [0  2  1]   [1.0  0.5  3.0]   [1.0  2.5  4.0]
                        [1  0  3] + [0.0  2.5  0.5] = [1.0  2.5  3.5]
                        [0  1  0]   [0.5  0.0  1.5]   [0.5  1.0  1.5]
```

### 2.4 Cálculo de Indicadores

#### Motricidad (Capacidad de Influir)

Para la variable i:

```
Motricidad_i = Σ(j=1 to n) M_total[i,j]
```

**Interpretación:** Suma de la fila i → Cuánto influye i sobre todas las demás

#### Dependencia (Susceptibilidad a ser Influido)

Para la variable j:

```
Dependencia_j = Σ(i=1 to n) M_total[i,j]
```

**Interpretación:** Suma de la columna j → Cuánto es influido j por todas las demás

#### Ejemplo Continuado

Del ejemplo anterior:

```
Motricidad = [Σfila1, Σfila2, Σfila3] = [7.5, 7.0, 3.0]
Dependencia = [Σcol1, Σcol2, Σcol3] = [2.5, 6.0, 9.0]
```

Variable 1: Alta motricidad (7.5), baja dependencia (2.5) → **DETERMINANTE**
Variable 2: Alta motricidad (7.0), alta dependencia (6.0) → **CRÍTICA**
Variable 3: Baja motricidad (3.0), alta dependencia (9.0) → **RESULTADO**

### 2.5 Algoritmo de Implementación

```python
def micmac_total(M, alpha, K):
    """
    Calcula matriz total MICMAC
    
    Parámetros:
    - M: matriz numpy (n×n) de influencias directas
    - alpha: float en (0,1] - factor de atenuación
    - K: int >= 2 - profundidad máxima
    
    Retorna:
    - M_total: matriz numpy (n×n) con influencias totales
    """
    M = M.astype(float)
    M_total = M.copy()
    M_power = M.copy()
    
    for k in range(2, K + 1):
        M_power = M_power @ M  # Multiplicación matricial
        M_total += (alpha ** (k - 1)) * M_power
    
    # Forzar diagonal a 0
    np.fill_diagonal(M_total, 0.0)
    
    return M_total
```

---

## 3. Interpretación de Resultados

### 3.1 El Plano Motricidad-Dependencia

#### Construcción del Plano

1. **Eje X (horizontal):** Dependencia total
2. **Eje Y (vertical):** Motricidad total
3. **Umbrales:** Media o mediana de motricidad/dependencia
4. **Cuadrantes:** Divididos por los umbrales

```
        Motricidad
            ^
            |
     Z2     |     Z1
  CRÍTICO   |  DETERMINANTES
            |
------------|-------------> Dependencia
            |
     Z3     |     Z4
 RESULTADO  |  AUTÓNOMAS
            |
```

### 3.2 Clasificación en Cuadrantes

#### Zona 1: Variables DETERMINANTES (Superior Izquierda)

**Características:**
- **Motricidad:** Alta (> umbral)
- **Dependencia:** Baja (< umbral)

**Interpretación Estratégica:**

✅ **PALANCAS DE ACCIÓN** del sistema
- Fáciles de controlar (baja dependencia)
- Gran capacidad de influir (alta motricidad)
- Bajo riesgo de efectos colaterales
- Puntos de intervención prioritarios

**Acción Recomendada:** **ACTUAR DIRECTAMENTE**
- Invertir recursos
- Implementar políticas
- Establecer control estricto

**Ejemplos Típicos:**
- Decisiones ejecutivas
- Políticas corporativas
- Inversiones estratégicas
- Recursos controlables

#### Zona 2: Variables CRÍTICAS/INESTABLES (Superior Derecha)

**Características:**
- **Motricidad:** Alta (> umbral)
- **Dependencia:** Alta (> umbral)

**Interpretación Estratégica:**

⚠️ **AMPLIFICADORES SISTÉMICOS**
- Gran capacidad de influir (alta motricidad)
- Muy influidas por otras (alta dependencia)
- Funcionan como "relays" o transmisores
- Potencial de efectos en cascada

**Acción Recomendada:** **MONITOREAR Y EQUILIBRAR**
- Sistema de alertas tempranas
- Análisis de sensibilidad continuo
- Planes de contingencia robustos
- Gestión de riesgos especializada

**Riesgo:** 🔴 **ALTO** - Pueden desestabilizar el sistema completo

**Ejemplos Típicos:**
- Mercados financieros
- Tecnologías emergentes
- Regulaciones cambiantes
- Factores geopolíticos

#### Zona 3: Variables RESULTADO (Inferior Derecha)

**Características:**
- **Motricidad:** Baja (< umbral)
- **Dependencia:** Alta (> umbral)

**Interpretación Estratégica:**

📊 **INDICADORES DE IMPACTO**
- Poca capacidad de influir (baja motricidad)
- Muy influidas por el sistema (alta dependencia)
- Reflejan el estado general del sistema
- Cambian como consecuencia, no como causa

**Acción Recomendada:** **MEDIR Y MONITOREAR**
- Usar como KPIs (Key Performance Indicators)
- Sistemas de dashboard
- Señales de alerta temprana
- NO intervenir directamente

**Riesgo:** 🟢 **BAJO** - No generan efectos sistémicos significativos

**Ejemplos Típicos:**
- Satisfacción del cliente
- Rentabilidad
- Cuota de mercado
- Clima laboral

#### Zona 4: Variables AUTÓNOMAS (Inferior Izquierda)

**Características:**
- **Motricidad:** Baja (< umbral)
- **Dependencia:** Baja (< umbral)

**Interpretación Estratégica:**

🔘 **FACTORES INDEPENDIENTES**
- Poca capacidad de influir
- Poco influidas por otras
- Operan de forma relativamente aislada
- Bajo impacto en la dinámica sistémica

**Acción Recomendada:** **GESTIÓN RUTINARIA**
- Prioridad baja
- Procedimientos estándar
- Revisión periódica no crítica

**Riesgo:** 🟢 **MUY BAJO** - Pueden ignorarse estratégicamente

**Ejemplos Típicos:**
- Variables periféricas
- Factores contextuales estables
- Elementos desconectados del sistema core

### 3.3 El Eje de Estrategia

#### Definición

El **eje de estrategia** es la diagonal del plano motricidad-dependencia (línea y = x).

#### Variables Estratégicas

Son aquellas **cercanas al eje** y con **valores absolutos altos**.

**Características:**
- Balance óptimo entre motricidad y dependencia
- Alto valor estratégico
- Influencia sostenible
- Menor vulnerabilidad que variables críticas

#### Cálculo de Proximidad al Eje

```python
# Normalizar coordenadas
x_norm = dependencia / max(dependencia)
y_norm = motricidad / max(motricidad)

# Distancia al eje (diagonal)
dist_to_axis = abs(y_norm - x_norm) / sqrt(2)

# Puntuación estratégica
strategic_score = (x_norm + y_norm) / 2 - factor * dist_to_axis
```

---

## 4. Configuración de Parámetros

### 4.1 Factor de Atenuación (α)

#### Interpretación Física

El factor α controla cuánto "peso" tienen las influencias indirectas lejanas.

**Valores recomendados según contexto:**

| α | Contexto | Interpretación |
|---|----------|----------------|
| **0.2-0.4** | Sistemas de respuesta rápida | Solo influencias cercanas importan |
| **0.5** | **RECOMENDADO general** | Balance entre corto y largo plazo |
| **0.6-0.8** | Sistemas complejos evolutivos | Influencias lejanas son relevantes |
| **0.9-1.0** | Sistemas fuertemente acoplados | Todas las rutas son importantes |

#### Guía de Selección

**Usar α BAJO (0.2-0.4) cuando:**
- Sistema de respuesta rápida
- Relaciones débiles entre elementos
- Interesa solo el corto plazo
- Alta incertidumbre en rutas largas

**Usar α MEDIO (0.4-0.6) cuando:**
- Sistema estándar (recomendado por defecto)
- Balance entre corto y largo plazo
- Primer análisis exploratorio

**Usar α ALTO (0.6-1.0) cuando:**
- Sistema fuertemente interconectado
- Importan efectos de largo plazo
- Retroalimentaciones significativas
- Cascadas de influencia conocidas

### 4.2 Profundidad de Análisis (K)

#### Interpretación

K define el número máximo de "saltos" en las cadenas de influencia.

**Convergencia:**
- Típicamente el ranking se estabiliza entre K=5 y K=9
- Valores K>10 raramente aportan información nueva
- K=2 es suficiente para análisis preliminar

#### Método de Selección Automática

```python
def first_stable_K(M, alpha, K_max=15):
    """
    Encuentra el K donde el ranking se estabiliza
    """
    prev_ranking = None
    
    for K in range(2, K_max + 1):
        M_tot = micmac_total(M, alpha, K)
        motricidad = M_tot.sum(axis=1)
        ranking = tuple(np.argsort(-motricidad))
        
        if prev_ranking == ranking:
            return K  # Estabilizado
        
        prev_ranking = ranking
    
    return K_max  # No estabilizó
```

**Recomendación:** Usar detección automática de K (incluida en la app)

### 4.3 Umbral de Clasificación

#### Opciones

**Opción 1: Media Aritmética** (Método MICMAC clásico)
```python
umbral_motricidad = np.mean(motricidad)
umbral_dependencia = np.mean(dependencia)
```

**Opción 2: Mediana** (Más robusta a outliers)
```python
umbral_motricidad = np.median(motricidad)
umbral_dependencia = np.median(dependencia)
```

#### Comparación

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| **Media** | Método oficial MICMAC | Sensible a valores extremos |
| **Mediana** | Robusta a outliers | Divide exactamente en 50%-50% |

**Recomendación:** 
- **Media** para comparar con MICMAC oficial
- **Mediana** si hay variables con valores muy extremos

---

## 5. Validación y Confiabilidad

### 5.1 Validación de la Matriz Directa (M)

#### Checklist de Calidad

✅ **Completitud:**
- Todas las variables relevantes están incluidas
- No hay variables redundantes
- Cobertura completa del sistema

✅ **Consistencia:**
- Valores en escala homogénea (0-3 o 0-4)
- Criterios de valoración claros y consistentes
- Diagonal en ceros

✅ **Validación Experta:**
- Consenso entre múltiples expertos (idealmente)
- Documentación de supuestos
- Justificación de influencias clave

#### Métodos de Construcción

**Método 1: Taller de Expertos**
1. Identificar stakeholders clave
2. Sesión de trabajo grupal (4-8 personas)
3. Discusión estructurada de cada relación
4. Votación o consenso
5. Documentar justificaciones

**Método 2: Método Delphi**
1. Cuestionario individual a expertos
2. Recopilación anónima de valoraciones
3. Análisis estadístico (media, dispersión)
4. Retroalimentación a expertos
5. Segunda ronda de valoración
6. Consenso final

**Método 3: Análisis Híbrido**
- Datos cuantitativos (correlaciones, causalidad estadística)
- Conocimiento experto para relaciones no medibles
- Validación cruzada con literatura

### 5.2 Análisis de Sensibilidad

#### Test de Robustez

Verificar que los resultados son **estables** ante:

**1. Variación de α:**
```python
alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
for alpha in alphas:
    ranking = compute_ranking(M, alpha, K)
    # ¿El top-5 cambia significativamente?
```

**2. Variación de K:**
```python
Ks = [3, 5, 7, 9]
for K in Ks:
    ranking = compute_ranking(M, alpha, K)
    # ¿Se estabiliza el ranking?
```

**3. Perturbaciones en M:**
```python
# Agregar ruido aleatorio ±5% a cada valor
M_perturbed = M + np.random.uniform(-0.15, 0.15, M.shape)
# ¿El ranking top-10 se mantiene?
```

#### Indicadores de Robustez

**Variables Robustas:**
- Mantienen posición en top-10 con variaciones de α
- Clasificación de cuadrante estable
- Poco sensibles a pequeños cambios en M

**Variables Sensibles:**
- Cambian de cuadrante con α diferentes
- Posición en ranking muy variable
- Requieren análisis adicional

### 5.3 Validación de Resultados

#### Coherencia con Conocimiento Experto

✅ ¿Las variables "determinantes" identificadas son realmente controlables?
✅ ¿Las variables "críticas" coinciden con puntos de riesgo conocidos?
✅ ¿Las variables "resultado" son efectivamente indicadores?
✅ ¿Hay sorpresas (variables inesperadas en lugares inesperados)?

#### Comparación con Literatura

- Contrastar con estudios previos del mismo dominio
- Verificar si variables clave coinciden
- Explicar diferencias si las hay

#### Validación Empírica

- Si es posible, contrastar con evolución histórica real
- ¿Las variables motrices identificadas efectivamente han sido influyentes?
- ¿Las predicciones del modelo se han cumplido?

---

## 6. Casos de Uso

### 6.1 Prospectiva Corporativa

**Contexto:** Identificar factores clave para estrategia empresarial 2030

**Variables típicas:**
- Tecnología
- Regulación
- Competencia
- Recursos humanos
- Financiación
- Mercados
- Innovación

**Pregunta clave:** ¿Dónde invertir recursos limitados para máximo impacto?

**Resultado esperado:** Priorización de iniciativas estratégicas

### 6.2 Análisis de Políticas Públicas

**Contexto:** Diseño de políticas de desarrollo sostenible

**Variables típicas:**
- Educación
- Salud
- Infraestructura
- Medio ambiente
- Economía
- Gobernanza
- Seguridad

**Pregunta clave:** ¿Qué políticas tienen mayor efecto multiplicador?

**Resultado esperado:** Roadmap de intervenciones priorizadas

### 6.3 Gestión de Riesgos

**Contexto:** Identificar riesgos sistémicos en cadena de suministro

**Variables típicas:**
- Proveedores clave
- Logística
- Regulaciones
- Geopolítica
- Tecnología
- Financiación
- Demanda

**Pregunta clave:** ¿Cuáles son los puntos críticos de vulnerabilidad?

**Resultado esperado:** Plan de mitigación de riesgos

### 6.4 Inteligencia Competitiva

**Contexto:** Análisis del ecosistema competitivo de una industria

**Variables típicas:**
- Competidores
- Tecnologías disruptivas
- Regulación
- Barreras de entrada
- Poder de negociación
- Substitutos
- Complementadores

**Pregunta clave:** ¿Dónde está el verdadero poder en la industria?

**Resultado esperado:** Estrategia de posicionamiento competitivo

---

## 7. Limitaciones y Consideraciones

### 7.1 Supuestos del Método

⚠️ **Linealidad:**
- Asume relaciones lineales (M^k)
- No captura sinergias o antagonismos complejos
- Efectos umbral no se modelan

⚠️ **Estabilidad Temporal:**
- La matriz M se asume constante
- En realidad, las relaciones evolucionan
- Requiere actualización periódica (3-6 meses)

⚠️ **Completitud:**
- Solo analiza variables incluidas
- Variables omitidas pueden ser críticas
- Requiere conocimiento exhaustivo del sistema

⚠️ **Independencia de Rutas:**
- Trata cada ruta de forma independiente
- No captura interacciones de orden superior
- Simplificación de la realidad compleja

### 7.2 Errores Comunes

#### Error 1: Matriz Incompleta

**Problema:** Faltan variables clave del sistema

**Consecuencia:** Resultados sesgados, variables importantes omitidas

**Solución:** Proceso riguroso de identificación de variables (brainstorming, literatura, stakeholders)

#### Error 2: Escala Heterogénea

**Problema:** Algunas relaciones valoradas en 0-3, otras en 0-10

**Consecuencia:** Pesos artificiales, dominancia de ciertas variables

**Solución:** Escala homogénea y normalización si es necesario

#### Error 3: Sobreinterpretación

**Problema:** Tomar resultados como "verdad absoluta"

**Consecuencia:** Decisiones erróneas basadas en modelo simplificado

**Solución:** Usar MICMAC como herramienta de exploración, no oráculo

#### Error 4: Ignorar Análisis de Sensibilidad

**Problema:** Confiar en un único conjunto de parámetros

**Consecuencia:** No detectar variables sensibles a supuestos

**Solución:** Siempre probar múltiples configuraciones α y K

### 7.3 Buenas Prácticas

✅ **Documentación:**
- Registrar todas las decisiones metodológicas
- Justificar valoraciones de influencias
- Mantener trazabilidad

✅ **Iteración:**
- Primera versión → revisión expertos → ajustes → versión final
- Actualizar periódicamente (trimestral/semestral)

✅ **Complementariedad:**
- Combinar MICMAC con otras herramientas (MACTOR, escenarios, SWOT)
- No usar como único método de análisis

✅ **Comunicación:**
- Presentar resultados con gráficos claros
- Explicar limitaciones a stakeholders
- Fomentar discusión, no imposición

### 7.4 Cuándo NO Usar MICMAC

❌ **Sistema con <10 variables:** Demasiado simple, análisis directo es suficiente

❌ **Relaciones bien modeladas matemáticamente:** Usar modelos cuantitativos específicos

❌ **Sistemas con dinámicas no lineales críticas:** Requiere simulación dinámica (System Dynamics)

❌ **Decisiones inmediatas:** MICMAC es para estrategia de medio-largo plazo

❌ **Falta de conocimiento experto:** Basura entra → Basura sale

---

## Referencias Completas

1. **Godet, M. (1990).** *From Anticipation to Action: A Handbook of Strategic Prospective.* UNESCO Publishing, Paris.

2. **Godet, M., & Durance, P. (2011).** *Strategic Foresight for Corporate and Regional Development.* Fondation Prospective et Innovation, UNESCO, Paris.

3. **Arcade, J., Godet, M., Meunier, F., & Roubelat, F. (2004).** *Structural analysis with the MICMAC method & Actor's strategy with MACTOR method.* In: Futures Research Methodology, Version 3.0. AC/UNU Millennium Project.

4. **Duperrin, J. C., & Godet, M. (1973).** *Méthode de hiérarchisation des éléments d'un système.* Rapport Économique du CEA, Paris.

5. **Godet, M. (2000).** *The Art of Scenarios and Strategic Planning: Tools and Pitfalls.* Technological Forecasting and Social Change, 65(1), 3-22.

6. **Asan, S. S., & Asan, U. (2007).** *Qualitative cross-impact analysis with time consideration.* Technological Forecasting and Social Change, 74(5), 627-644.

---

*Documento revisado: Enero 2025*  
*Versión: 1.0*
