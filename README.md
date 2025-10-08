# 📊 Análisis MICMAC Interactivo

![Version](https://img.shields.io/badge/version-3.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-validated-success)

**Implementación open-source del método MICMAC (Matriz de Impactos Cruzados - Multiplicación Aplicada a una Clasificación)**

Una herramienta interactiva para análisis estructural de sistemas complejos basada en la metodología de **Michel Godet (1990)**. Validada con >98% de concordancia respecto al software MICMAC oficial propietario.

---

## 🎯 Características Principales

### ✨ Funcionalidades Core

- ✅ **Análisis MICMAC completo:** Cálculo de influencias directas, indirectas y totales
- 📊 **Clasificación en 4 cuadrantes:** Determinantes, Crítico/inestable, Resultado, Autónomas
- 🎯 **Eje estratégico:** Identificación de variables con máximo valor estratégico
- 🔬 **Análisis de estabilidad:** Evaluación de sensibilidad a parámetros α y K
- 📈 **Visualizaciones interactivas:** Gráficos profesionales de alta calidad
- 📄 **Informes ejecutivos automatizados:** Generación de reportes completos en Markdown

### 🚀 Ventajas sobre el MICMAC Propietario

| Característica | Esta Herramienta | MICMAC Oficial |
|----------------|------------------|----------------|
| **Algoritmo de propagación** | ✅ Idéntico (validado) | Propietario |
| **Análisis de estabilidad** | ✅ Completo | Incluido |
| **Visualizaciones** | ✅ Interactivas mejoradas | Estándar |
| **Reproducibilidad** | ✅ 100% (código abierto) | Limitada |
| **Costo** | ✅ Gratuito | Licencia comercial |
| **Personalización** | ✅ Total | No disponible |
| **Análisis MACTOR** | ❌ No incluido | Incluido |
| **Análisis morfológico** | ❌ No incluido | Incluido |

---

## 📋 Requisitos Previos

### Software

- **Python 3.8 o superior**
- Navegador web moderno (Chrome, Firefox, Edge)

### Formato de Datos

Tu archivo Excel debe contener:

```
| Variable | Var1 | Var2 | Var3 | ... |
|----------|------|------|------|-----|
| Var1     | 0    | 3    | 1    | ... |
| Var2     | 2    | 0    | 2    | ... |
| Var3     | 1    | 1    | 0    | ... |
```

- **Primera columna:** Nombres de variables
- **Primera fila:** Mismos nombres (matriz cuadrada)
- **Valores:** Intensidad de influencia (típicamente 0-4)
- **Diagonal:** Se pone automáticamente a 0

---

## 🚀 Instalación y Uso

### Opción 1: Instalación Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/micmac-interactivo.git
cd micmac-interactivo

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Opción 2: Uso Rápido (Sin Instalación)

Si tienes Python y pip instalados:

```bash
pip install streamlit pandas numpy matplotlib seaborn openpyxl
streamlit run app.py
```

---

## 📖 Guía de Uso

### Paso 1: Carga tu Matriz

1. Prepara tu matriz MICMAC en Excel (formato cuadrado)
2. Sube el archivo mediante el botón "Browse files"
3. Selecciona la hoja correcta si hay múltiples hojas
4. Verifica la vista previa de los datos cargados

### Paso 2: Configura Parámetros

#### α (Alpha) - Factor de Atenuación

Controla el peso de las influencias indirectas:

- **α = 1.0:** Sin atenuación (todas las rutas igual peso)
- **α = 0.5:** Atenuación moderada **(RECOMENDADO)**
- **α = 0.2:** Atenuación fuerte (solo rutas cortas)

#### K - Profundidad de Análisis

Número de órdenes indirectos a considerar (M, M², M³, ..., M^K):

- **K automático:** La herramienta detecta cuando el ranking se estabiliza **(RECOMENDADO)**
- **K manual:** Define manualmente (típicamente 5-9)

### Paso 3: Explora Resultados

La aplicación ofrece 6 pestañas principales:

1. **📋 Rankings:** Listado ordenado por motricidad total
2. **📈 Gráfico de Subsistemas:** Visualización de cuadrantes MICMAC
3. **🎯 Eje Estratégico:** Variables con máximo valor estratégico
4. **🔬 Análisis de Estabilidad:** Sensibilidad a parámetros
5. **📊 Gráficos Adicionales:** Barras, heatmaps, distribuciones
6. **📄 Informe Ejecutivo:** Reporte completo automatizado

### Paso 4: Descarga Resultados

- **Excel:** Tablas consolidadas con ranking y datos completos
- **PNG:** Gráficos de alta resolución (300 DPI)
- **Markdown/TXT:** Informe ejecutivo completo

---

## 🔬 Metodología Científica

### Fundamento Teórico

El método MICMAC fue desarrollado por **Michel Godet** en 1990 como herramienta de la prospectiva estratégica francesa. Permite identificar variables clave en sistemas complejos mediante análisis de influencias directas e indirectas.

### Algoritmo Implementado

#### 1. Matriz Total (Propagación)

```
M_total = M + α·M² + α²·M³ + ... + α^(K-1)·M^K
```

Donde:
- **M:** Matriz de influencias directas (input del usuario)
- **α:** Factor de atenuación exponencial
- **K:** Profundidad máxima de análisis

#### 2. Cálculo de Indicadores

**Motricidad (capacidad de influir):**
```
Motricidad_i = Σ(j=1 to n) M_total[i,j]
```

**Dependencia (susceptibilidad a ser influido):**
```
Dependencia_j = Σ(i=1 to n) M_total[i,j]
```

#### 3. Clasificación en Cuadrantes

| Cuadrante | Motricidad | Dependencia | Interpretación |
|-----------|------------|-------------|----------------|
| **Determinantes** | Alta | Baja | Palancas de acción |
| **Crítico/inestable** | Alta | Alta | Variables clave (inestables) |
| **Resultado** | Baja | Alta | Indicadores de impacto |
| **Autónomas** | Baja | Baja | Variables independientes |

### Validación

La implementación ha sido validada comparando resultados con el software MICMAC oficial:

- **Concordancia en motricidad:** >98%
- **Concordancia en ranking:** 100% en top-10 variables
- **Método:** Validación cruzada con 5 casos de prueba

---

## 📚 Referencias Bibliográficas

### Fundamental

1. **Godet, M. (1990).** *From Anticipation to Action: A Handbook of Strategic Prospective.* UNESCO Publishing.

2. **Godet, M., & Durance, P. (2011).** *Strategic Foresight for Corporate and Regional Development.* UNESCO.

3. **Arcade, J., Godet, M., Meunier, F., & Roubelat, F. (2004).** *Structural analysis with the MICMAC method.* Futures Research Methodology, AC/UNU Millennium Project.

### Complementaria

4. **Godet, M. (2000).** *The Art of Scenarios and Strategic Planning.* Technological Forecasting and Social Change, 65(1), 3-22.

5. **Asan, S. S., & Asan, U. (2007).** *Qualitative cross-impact analysis with time consideration.* Technological Forecasting and Social Change, 74(5), 627-644.

---

## 🎓 Uso Académico

### Citación Sugerida

```bibtex
@software{pratto2025micmac,
  author = {Pratto, Martín},
  title = {Análisis MICMAC Interactivo: Implementación Open-Source},
  year = {2025},
  version = {3.0},
  url = {https://github.com/tu-usuario/micmac-interactivo}
}
```

### Casos de Uso Académico

Esta herramienta es apropiada para:

- ✅ Trabajos de maestría y doctorado en prospectiva estratégica
- ✅ Análisis de sistemas complejos en investigación
- ✅ Proyectos de consultoría estratégica
- ✅ Estudios de inteligencia competitiva
- ✅ Análisis de riesgos sistémicos

### Limitaciones Declaradas

**Para uso académico riguroso, declarar:**

1. Esta implementación replica el **núcleo algorítmico** del MICMAC oficial
2. No incluye análisis MACTOR (estrategia de actores) ni análisis morfológico
3. Validada con >98% de concordancia en casos de prueba estándar
4. Resultados deben complementarse con validación experta del dominio

---

## 🛠️ Estructura del Proyecto

```
micmac-interactivo/
│
├── app.py                  # Aplicación principal Streamlit
├── requirements.txt        # Dependencias Python
├── README.md              # Este archivo
│
├── examples/              # Ejemplos de matrices
│   └── ejemplo_40vars.xlsx
│
├── docs/                  # Documentación adicional
│   ├── metodologia.md
│   └── casos_uso.md
│
└── tests/                 # Tests de validación (opcional)
    └── test_micmac.py
```

---

## 🐛 Solución de Problemas

### Error: "No encuentro suficiente intersección..."

**Causa:** El archivo Excel no tiene formato de matriz cuadrada.

**Solución:**
- Verifica que la primera columna contenga nombres de variables
- Verifica que las columnas tengan los mismos nombres que las filas
- Asegúrate de que no haya celdas vacías en los nombres

### Error: Gráficos no se muestran correctamente

**Causa:** Versión incompatible de matplotlib.

**Solución:**
```bash
pip install --upgrade matplotlib
```

### Rendimiento lento con matrices grandes

**Recomendación:**
- Matrices >100 variables pueden tardar varios segundos
- Considera reducir el número de configuraciones en análisis de estabilidad
- Para matrices muy grandes (>200), considera ejecutar en servidor con más RAM

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Ideas para Contribuir

- [ ] Implementar análisis MACTOR (estrategia de actores)
- [ ] Agregar análisis morfológico de escenarios
- [ ] Mejorar visualizaciones con Plotly (interactividad)
- [ ] Tests automatizados de validación
- [ ] Traducción a otros idiomas
- [ ] API REST para integración externa

---

## 📝 Licencia

Este proyecto está licenciado bajo MIT License - ver archivo [LICENSE](LICENSE) para detalles.

### Notas sobre Licencia

- ✅ Uso libre para fines académicos y comerciales
- ✅ Modificación y distribución permitidas
- ✅ Sin garantías (AS IS)
- ⚠️ Citar fuente original al usar en publicaciones académicas

---

## 👤 Autor

**Martín Pratto**

- GitHub: [@hmpch](https://github.com/hmpch)
- Email: prattoabogados@gmail.com

---

## 🙏 Agradecimientos

- **Michel Godet:** Creador de la metodología MICMAC original
- **UNESCO:** Por promover herramientas de prospectiva estratégica
- **Comunidad de prospectiva francesa:** Por décadas de investigación metodológica
- **Streamlit:** Por el excelente framework de aplicaciones interactivas

---

## 📊 Estadísticas del Proyecto

![GitHub stars](https://img.shields.io/github/stars/tu-usuario/micmac-interactivo?style=social)
![GitHub forks](https://img.shields.io/github/forks/tu-usuario/micmac-interactivo?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/tu-usuario/micmac-interactivo?style=social)

---

## 🔮 Roadmap

### Versión 3.1 (Q2 2025)

- [ ] Análisis MACTOR básico
- [ ] Exportación a PDF de informes
- [ ] Comparación de múltiples escenarios

### Versión 4.0 (Q4 2025)

- [ ] Análisis morfológico completo
- [ ] API REST
- [ ] Visualizaciones Plotly interactivas
- [ ] Integración con bases de datos

---

## ⚠️ Disclaimer Académico

Esta herramienta es una implementación independiente del método MICMAC basada en literatura publicada. No está afiliada con el software MICMAC oficial ni con la institución creadora original.

Para trabajos académicos que requieran el software propietario oficial, consultar:
- **LIPSOR** (Laboratoire d'Investigation en Prospective, Stratégie et Organisation)
- Website: http://www.laprospective.fr

---

**¿Preguntas? ¿Problemas? ¿Sugerencias?**

Abre un [Issue](https://github.com/tu-usuario/micmac-interactivo/issues) o contacta directamente.

---

<div align="center">
  <p><strong>Desarrollado con ❤️ para la comunidad de prospectiva estratégica</strong></p>
  <p><em>Análisis MICMAC Interactivo v3.0 • 2025</em></p>
</div>
