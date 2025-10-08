# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto MICMAC Interactivo! Este documento te guiará en el proceso.

---

## 📋 Tabla de Contenidos

1. [Código de Conducta](#código-de-conducta)
2. [¿Cómo puedo contribuir?](#cómo-puedo-contribuir)
3. [Configuración del Entorno de Desarrollo](#configuración-del-entorno-de-desarrollo)
4. [Proceso de Contribución](#proceso-de-contribución)
5. [Estándares de Código](#estándares-de-código)
6. [Testing](#testing)
7. [Documentación](#documentación)
8. [Roadmap y Prioridades](#roadmap-y-prioridades)

---

## Código de Conducta

Este proyecto se adhiere a los principios de respeto, inclusión y colaboración profesional. Al participar, te comprometes a:

- ✅ Ser respetuoso con todos los contribuidores
- ✅ Aceptar críticas constructivas con mente abierta
- ✅ Enfocarte en lo mejor para la comunidad y el proyecto
- ✅ Mostrar empatía hacia otros miembros de la comunidad

❌ No se toleran:
- Lenguaje o imágenes sexualizadas
- Ataques personales o políticos
- Acoso público o privado
- Publicación de información privada de terceros

---

## ¿Cómo puedo contribuir?

### 🐛 Reportar Bugs

¿Encontraste un error? Ayúdanos creando un issue detallado:

**Antes de reportar:**
1. Verifica que no exista ya un issue similar
2. Asegúrate de usar la última versión
3. Intenta reproducir el error de forma consistente

**Información a incluir:**
- **Descripción clara:** ¿Qué esperabas que pasara? ¿Qué pasó realmente?
- **Pasos para reproducir:** Lista numerada de acciones
- **Entorno:** OS, versión de Python, versión de librerías
- **Screenshots:** Si es relevante
- **Logs de error:** Mensaje de error completo

**Ejemplo de buen reporte:**
```markdown
**Bug:** Gráfico de subsistemas no se renderiza con matrices >50 variables

**Pasos para reproducir:**
1. Cargar archivo con 60 variables
2. Configurar α=0.5, K=6
3. Navegar a tab "Gráfico de Subsistemas"

**Resultado esperado:** Gráfico se muestra correctamente
**Resultado actual:** Error "Timeout" y gráfico en blanco

**Entorno:**
- OS: Windows 11
- Python: 3.10.5
- Matplotlib: 3.7.1

**Error log:**
```
TimeoutError: Rendering exceeded 60 seconds
```

---

### 💡 Proponer Mejoras

¿Tienes una idea para mejorar el proyecto? Abre un issue con la etiqueta `enhancement`:

**Plantilla de propuesta:**
```markdown
**Título:** [MEJORA] Descripción breve

**Problema que resuelve:**
[Explica el problema o necesidad actual]

**Solución propuesta:**
[Describe tu idea en detalle]

**Alternativas consideradas:**
[¿Qué otras opciones hay?]

**Impacto:**
- Usuarios beneficiados: [Estimación]
- Complejidad de implementación: [Baja/Media/Alta]
- Breaking changes: [Sí/No]

**Disposición a implementar:**
[¿Estás dispuesto a trabajar en esto?]
```

---

### 📝 Mejorar Documentación

La documentación siempre necesita amor. Contribuciones bienvenidas:

- Corregir errores tipográficos o gramaticales
- Clarificar explicaciones confusas
- Agregar ejemplos prácticos
- Traducir a otros idiomas
- Mejorar comentarios en el código

**Archivos de documentación:**
- `README.md` - Documentación principal
- `docs/METODOLOGIA.md` - Teoría y algoritmos
- `docs/CASOS_DE_USO.md` - Ejemplos prácticos
- Docstrings en `app.py` - Documentación inline

---

### 🔧 Contribuir con Código

Ver sección [Proceso de Contribución](#proceso-de-contribución) más abajo.

---

## Configuración del Entorno de Desarrollo

### 1. Fork y Clone

```bash
# Fork en GitHub (botón "Fork" en la página del repo)

# Clonar tu fork
git clone https://github.com/TU-USUARIO/micmac-interactivo.git
cd micmac-interactivo

# Agregar el repo original como "upstream"
git remote add upstream https://github.com/USUARIO-ORIGINAL/micmac-interactivo.git
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Linux/Mac)
source venv/bin/activate

# Activar (Windows)
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
# Dependencias de producción
pip install -r requirements.txt

# Dependencias de desarrollo (opcional)
pip install black flake8 pytest pytest-cov
```

### 4. Verificar Instalación

```bash
# Ejecutar la app
streamlit run app.py

# Si funciona, ¡estás listo para contribuir!
```

---

## Proceso de Contribución

### Flujo Estándar

```
1. Crear issue (si no existe)
   ↓
2. Asignarte el issue (comenta "Me asigno")
   ↓
3. Crear rama feature
   ↓
4. Escribir código + tests
   ↓
5. Commit con mensajes claros
   ↓
6. Push a tu fork
   ↓
7. Abrir Pull Request
   ↓
8. Code review e iteración
   ↓
9. Merge! 🎉
```

### Detalle de Pasos

#### 1. Crear Rama

```bash
# Actualizar tu fork
git checkout main
git pull upstream main

# Crear rama descriptiva
git checkout -b feature/nombre-descriptivo
# Ejemplos:
# - feature/analisis-mactor
# - fix/graficos-grandes-matrices
# - docs/traduccion-espanol
```

#### 2. Hacer Cambios

```bash
# Hacer tus cambios en el código

# Verificar que funciona
streamlit run app.py

# Agregar tests si es aplicable
# (ver sección Testing más abajo)
```

#### 3. Commit

Usa **mensajes claros y descriptivos**:

```bash
# Formato: <tipo>: <descripción breve>

git add .
git commit -m "feat: agregar análisis MACTOR básico"

# Tipos de commit:
# - feat: Nueva funcionalidad
# - fix: Corrección de bug
# - docs: Cambios en documentación
# - style: Formato, no afecta lógica
# - refactor: Refactorización sin cambiar funcionalidad
# - test: Agregar o modificar tests
# - chore: Mantenimiento (actualizar dependencias, etc.)
```

**Ejemplos de buenos commits:**
```
feat: agregar exportación a PDF de informes
fix: corregir error en cálculo de dependencia total
docs: actualizar README con instrucciones de instalación
refactor: simplificar función micmac_total
test: agregar tests para clasificación de cuadrantes
```

#### 4. Push

```bash
# Push a tu fork
git push origin feature/nombre-descriptivo
```

#### 5. Pull Request

**En GitHub:**
1. Ve a tu fork
2. Haz clic en "Compare & pull request"
3. Completa la plantilla de PR (ver abajo)
4. Espera review

**Plantilla de Pull Request:**

```markdown
## Descripción
[Describe claramente qué hace este PR]

## Issue relacionado
Closes #[número de issue]

## Tipo de cambio
- [ ] Bug fix (non-breaking change)
- [ ] Nueva funcionalidad (non-breaking change)
- [ ] Breaking change (fix o feature que rompe funcionalidad existente)
- [ ] Mejora de documentación

## ¿Cómo se ha probado?
[Describe los tests realizados]

## Checklist
- [ ] Mi código sigue los estándares del proyecto
- [ ] He realizado self-review de mi código
- [ ] He comentado código complejo/no obvio
- [ ] He actualizado la documentación
- [ ] Mis cambios no generan nuevos warnings
- [ ] He agregado tests que prueban mi fix/feature
- [ ] Tests nuevos y existentes pasan localmente

## Screenshots (si aplica)
[Agregar capturas de pantalla si hay cambios visuales]
```

---

## Estándares de Código

### Estilo de Código Python

Seguimos **PEP 8** con algunas adaptaciones:

```python
# ✅ BIEN

def calcular_motricidad_total(M: np.ndarray, alpha: float, K: int) -> np.ndarray:
    """
    Calcula la motricidad total usando propagación MICMAC.
    
    Args:
        M: Matriz de influencias directas (n×n)
        alpha: Factor de atenuación (0 < α ≤ 1)
        K: Profundidad máxima de análisis
    
    Returns:
        Array con motricidad total de cada variable
    """
    M_total = M.copy()
    M_power = M.copy()
    
    for k in range(2, K + 1):
        M_power = M_power @ M
        M_total += (alpha ** (k - 1)) * M_power
    
    return M_total.sum(axis=1)


# ❌ MAL

def calcMotr(m,a,k):  # Nombres no descriptivos, sin tipos
    mt=m.copy()  # Sin espacios alrededor de =
    mp=m.copy()
    for k in range(2,k+1):  # Variable k reutilizada
        mp=mp@m  # Sin espacios
        mt+=(a**(k-1))*mp
    return mt.sum(axis=1)
```

### Reglas Generales

1. **Nombres descriptivos:**
   - Variables: `snake_case`
   - Funciones: `snake_case`
   - Clases: `PascalCase`
   - Constantes: `UPPER_SNAKE_CASE`

2. **Type hints:**
   - Usar type hints en funciones públicas
   - Especialmente importante para parámetros complejos

3. **Docstrings:**
   - Todas las funciones públicas deben tener docstring
   - Formato: Google style o NumPy style

4. **Comentarios:**
   - Explicar "por qué", no "qué"
   - Comentar lógica compleja o no obvia

5. **Longitud de línea:**
   - Máximo 100 caracteres (más tolerante que PEP 8's 79)

### Formateo Automático

```bash
# Instalar black
pip install black

# Formatear todo el código
black app.py

# Verificar sin modificar
black --check app.py
```

### Linting

```bash
# Instalar flake8
pip install flake8

# Verificar código
flake8 app.py --max-line-length=100
```

---

## Testing

### Estructura de Tests

```
tests/
├── __init__.py
├── test_micmac_core.py      # Tests de funciones core
├── test_validations.py       # Tests de validaciones
└── test_integration.py       # Tests de integración
```

### Escribir Tests

```python
# tests/test_micmac_core.py

import numpy as np
import pytest
from app import micmac_total, classify_quadrant


def test_micmac_total_convergence():
    """Verifica que M_total converge con K suficiente"""
    M = np.array([[0, 2, 1],
                  [1, 0, 3],
                  [0, 1, 0]], dtype=float)
    
    # Calcular con K=5 y K=10
    M_total_5 = micmac_total(M, alpha=0.5, K=5)
    M_total_10 = micmac_total(M, alpha=0.5, K=10)
    
    # Deberían ser muy similares
    assert np.allclose(M_total_5, M_total_10, rtol=1e-3)


def test_classify_quadrant():
    """Verifica clasificación correcta de cuadrantes"""
    # Determinante: alta motricidad, baja dependencia
    assert classify_quadrant(100, 20, 50, 50) == 'Determinantes'
    
    # Crítico: alta motricidad, alta dependencia
    assert classify_quadrant(100, 100, 50, 50) == 'Crítico/inestable'
    
    # Resultado: baja motricidad, alta dependencia
    assert classify_quadrant(20, 100, 50, 50) == 'Variables resultado'
    
    # Autónoma: baja motricidad, baja dependencia
    assert classify_quadrant(20, 20, 50, 50) == 'Autónomas'


def test_micmac_total_diagonal_zero():
    """Verifica que la diagonal de M_total sea siempre 0"""
    M = np.random.rand(5, 5) * 4
    np.fill_diagonal(M, 0)
    
    M_total = micmac_total(M, alpha=0.5, K=6)
    
    assert np.allclose(np.diag(M_total), 0)
```

### Ejecutar Tests

```bash
# Instalar pytest
pip install pytest pytest-cov

# Ejecutar todos los tests
pytest

# Con reporte de cobertura
pytest --cov=app tests/

# Ejecutar tests específicos
pytest tests/test_micmac_core.py::test_micmac_total_convergence
```

---

## Documentación

### Docstrings de Funciones

Usa formato **Google style**:

```python
def micmac_total(M: np.ndarray, alpha: float, K: int) -> np.ndarray:
    """Calcula la matriz total MICMAC con propagación de influencias.
    
    La matriz total incluye influencias directas e indirectas hasta orden K,
    atenuadas exponencialmente por factor alpha según:
    M_total = M + α·M² + α²·M³ + ... + α^(K-1)·M^K
    
    Args:
        M: Matriz de influencias directas (n×n). Debe ser cuadrada con
           valores no negativos y diagonal en ceros.
        alpha: Factor de atenuación exponencial (0 < α ≤ 1). Valores típicos:
               0.3-0.4 (corto plazo), 0.5 (recomendado), 0.6-0.8 (largo plazo).
        K: Profundidad máxima de análisis (K >= 2). Número de órdenes indirectos
           a considerar. Típicamente converge entre K=5 y K=9.
    
    Returns:
        Matriz numpy (n×n) con influencias totales (directas + indirectas).
        La diagonal será forzada a 0.
    
    Raises:
        ValueError: Si M no es cuadrada o contiene valores negativos.
        ValueError: Si alpha no está en (0, 1].
        ValueError: Si K < 2.
    
    Examples:
        >>> M = np.array([[0, 2, 1], [1, 0, 3], [0, 1, 0]], dtype=float)
        >>> M_total = micmac_total(M, alpha=0.5, K=3)
        >>> M_total.shape
        (3, 3)
    
    References:
        Godet, M. (1990). From Anticipation to Action: A Handbook of 
        Strategic Prospective. UNESCO Publishing.
    """
    # Implementación...
```

### Comentarios en Código

```python
# ✅ BIEN: Explica el "por qué"

# Forzamos la diagonal a 0 para evitar que una variable se influya a sí misma,
# lo cual no tiene sentido en el modelo MICMAC y puede generar inestabilidad numérica
np.fill_diagonal(M_total, 0.0)

# El factor alpha^(k-1) atenúa exponencialmente las influencias lejanas.
# Por ejemplo, con alpha=0.5: M² se pondera 0.5, M³ se pondera 0.25, etc.
M_total += (alpha ** (k - 1)) * M_power


# ❌ MAL: Repite lo obvio del código

# Poner diagonal en cero
np.fill_diagonal(M_total, 0.0)

# Sumar alpha a la k menos uno por M power
M_total += (alpha ** (k - 1)) * M_power
```

---

## Roadmap y Prioridades

### Alta Prioridad (Help Wanted!)

🔥 **Análisis MACTOR**
- Implementar matriz de actores
- Análisis de convergencias/divergencias
- Gráfico de relaciones entre actores

🔥 **Tests Automatizados**
- Cobertura >80% de funciones core
- Tests de regresión
- CI/CD con GitHub Actions

🔥 **Optimización de Performance**
- Soporte para matrices >100 variables
- Caching de cálculos intermedios
- Paralelización para análisis de estabilidad

### Prioridad Media

⭐ **Mejoras de Visualización**
- Plotly en lugar de Matplotlib (interactividad)
- Animaciones de propagación de influencias
- Grafos de redes de influencia

⭐ **Exportación Avanzada**
- Exportación a PDF de informes
- Plantillas personalizables
- Integración con PowerPoint

⭐ **Internacionalización**
- Soporte multiidioma (inglés, portugués, francés)
- Detección automática de idioma

### Prioridad Baja (Nice to Have)

💡 **API REST**
- Endpoint para análisis MICMAC vía HTTP
- Documentación con Swagger
- Rate limiting y autenticación

💡 **Base de Datos**
- Guardar análisis históricos
- Comparación temporal de análisis
- Colaboración multi-usuario

💡 **Machine Learning**
- Sugerencias automáticas de relaciones
- Validación de matrices con históricos
- Predicción de evolución de variables

---

## Preguntas Frecuentes

### ¿Cuánto tiempo toma revisar un PR?

Típicamente **2-5 días hábiles**. PRs simples (docs, fixes pequeños) pueden ser más rápidos.

### ¿Qué pasa si mi PR no es aceptado?

No te desanimes! Explicaremos las razones y, si es posible, sugeriremos cómo modificarlo para que sea aceptable.

### ¿Puedo trabajar en algo que no está en el roadmap?

¡Claro! Pero **abre un issue primero** para discutir la idea. Así evitamos trabajo duplicado o features que no alineen con la visión del proyecto.

### ¿Necesito saber Streamlit para contribuir?

No necesariamente! Puedes contribuir en:
- Documentación
- Tests
- Algoritmos core (NumPy/Pandas)
- Casos de uso y ejemplos

### ¿Cómo puedo obtener más ayuda?

- Abre un issue con la etiqueta `question`
- Revisa la documentación en `/docs`
- Consulta ejemplos en `/examples`

---

## Agradecimientos

Gracias por contribuir a hacer MICMAC Interactivo mejor para toda la comunidad de prospectiva estratégica! 🎉

---

*Última actualización: Enero 2025*
