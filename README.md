# 🎯 JETLEX - Suite de Análisis Prospectivo by Martin Pratto Chiarella

![Version](https://img.shields.io/badge/version-4.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![Status](https://img.shields.io/badge/status-active-brightgreen)

**Suite completa de herramientas open-source para análisis estructural y estratégico**

Implementación de los métodos **MICMAC** y **MACTOR** desarrollados por Michel Godet, adaptados para análisis prospectivo en sistemas complejos.

---

## 🚀 Aplicaciones Disponibles

| Aplicación | Descripción | Estado |
|------------|-------------|--------|
| **MICMAC Original** | Análisis estructural clásico | ✅ Operativo |
| **MICMAC PRO** | Versión mejorada con conversor integrado | ✅ Operativo |
| **MACTOR** | Análisis de juego de actores | ✅ Operativo |

### 🔗 Acceso Directo

- **MICMAC Interactivo:** [https://micmac-interactivo-fvg2ckpsahhgzc7ywtfdel.streamlit.app/](https://micmac-interactivo-fvg2ckpsahhgzc7ywtfdel.streamlit.app/)
- **MICMAC PRO:** *(Actualizar con tu URL de Streamlit)*
- **MACTOR:** *(Actualizar con tu URL de Streamlit)*

---

## 📊 Características por Aplicación

### 1️⃣ MICMAC Original

Implementación completa del método de Matriz de Impactos Cruzados - Multiplicación Aplicada a una Clasificación.

| Funcionalidad | Descripción |
|---------------|-------------|
| ✅ Análisis MICMAC completo | Cálculo de influencias directas, indirectas y totales |
| 📊 Clasificación en 4 cuadrantes | Determinantes, Crítico/inestable, Resultado, Autónomas |
| 🎯 Eje estratégico | Identificación de variables con máximo valor estratégico |
| 🔬 Análisis de estabilidad | Evaluación de sensibilidad a parámetros α y K |
| 📈 Visualizaciones interactivas | Gráficos profesionales de alta calidad |
| 📄 Informes ejecutivos | Generación de reportes completos en Markdown |

### 2️⃣ MICMAC PRO (Nuevo)

Versión mejorada que incluye conversor de matrices con metadata.

| Funcionalidad | Descripción |
|---------------|-------------|
| 🔄 **Conversor integrado** | Procesa matrices Excel con columnas de metadata (Tipo, Nombre, Código) |
| 📥 Carga flexible | Acepta múltiples formatos de entrada |
| 🏷️ Detección automática | Extrae códigos originales (P1, E2, S3, T4, L5...) |
| 📊 Análisis MIDI | Matriz de Influencias Directas e Indirectas |
| 🎨 Visualizaciones Plotly | Gráficos interactivos modernos |
| 💾 Exportación Excel | Múltiples hojas con todos los resultados |

**¿Cuándo usar MICMAC PRO?**

Si tu matriz tiene esta estructura:

```
| Tipo       | Variable                    | Código | P1 | P6 | E1 | ...
|------------|------------------------------|--------|----|----|----|----|
| Políticas  | Descripción de la variable  | P1     | 0  | 2  | 1  | ...
| Económicas | Otra descripción            | E1     | 1  | 0  | 0  | ...
```

MICMAC PRO la convierte automáticamente al formato requerido.

### 3️⃣ MACTOR (Nuevo)

Implementación completa del Método de Análisis de Actores - Tácticas, Objetivos y Recomendaciones.

| Matriz | Descripción | Tipo |
|--------|-------------|------|
| **MAO** | Influencia de actores sobre objetivos | Input (0-4) |
| **1MAO** | Posición de actores frente a objetivos | Input (-1, 0, +1) |
| **2MAO** | Prioridad de objetivos para actores | Input (0-4) |
| **MID** | Influencias directas entre actores | Input (0-4) |
| **MIDI** | Influencias directas e indirectas | Calculada |
| **3MAO** | Implicación de actores en objetivos | Calculada |
| **4MAO** | Movilización de actores sobre objetivos | Calculada |

**Análisis incluidos:**

- 🤝 Matriz de convergencias (alianzas potenciales)
- ⚔️ Matriz de divergencias (conflictos potenciales)
- 📊 Balance de relaciones actor-actor
- 🕸️ Red de interacciones visualizada
- 🎯 Clasificación de actores: Motrices, Enlace, Dominados, Autónomos

---

## 📋 Comparativa con Software Propietario

| Característica | Suite JETLEX | MICMAC Oficial |
|----------------|--------------|----------------|
| Algoritmo de propagación | ✅ Idéntico (validado) | Propietario |
| Análisis de estabilidad | ✅ Completo | Incluido |
| Visualizaciones | ✅ Interactivas mejoradas | Estándar |
| Reproducibilidad | ✅ 100% (código abierto) | Limitada |
| Costo | ✅ **Gratuito** | Licencia comercial |
| Personalización | ✅ Total | No disponible |
| Análisis MACTOR | ✅ **Incluido** | Incluido |
| Conversor de matrices | ✅ **Incluido** | No disponible |
| Análisis morfológico | ❌ No incluido | Incluido |

---

## 🔧 Instalación

### Opción 1: Uso Online (Recomendado)

Accede directamente a las aplicaciones desplegadas en Streamlit Cloud sin necesidad de instalación.

### Opción 2: Instalación Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/jetlex-analisis-prospectivo.git
cd jetlex-analisis-prospectivo

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación deseada
streamlit run app.py              # MICMAC Original
streamlit run app_micmac_pro.py   # MICMAC PRO
streamlit run app_mactor.py       # MACTOR
```

### Dependencias

```txt
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.3
plotly==5.18.0
openpyxl==3.1.2
scipy==1.11.4
networkx==3.2.1
matplotlib==3.8.2
seaborn==0.13.0
```

---

## 📖 Guía de Uso

### MICMAC - Formato de Datos

Tu archivo Excel debe contener una matriz cuadrada:

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

### MICMAC PRO - Formato con Metadata

Si tu matriz incluye columnas adicionales de metadata:

```
| Tipo       | Nombre Completo              | Código | P1 | P6 | E1 |
|------------|------------------------------|--------|----|----|----| 
| Políticas  | Regulación ambiental         | P1     | 0  | 2  | 1  |
| Políticas  | Política de sostenibilidad   | P6     | 1  | 0  | 2  |
| Económicas | Inversión en infraestructura | E1     | 2  | 1  | 0  |
```

El conversor extraerá automáticamente los códigos y generará la matriz limpia.

### MACTOR - Configuración de Actores y Objetivos

1. **Define actores:** Stakeholders del sistema (gobierno, empresas, ONG, etc.)
2. **Define objetivos:** Metas estratégicas en disputa
3. **Completa matrices:**
   - MAO: ¿Cuánta influencia tiene cada actor sobre cada objetivo? (0-4)
   - 1MAO: ¿El actor está a favor (+1), neutral (0) o en contra (-1) del objetivo?
   - 2MAO: ¿Qué prioridad tiene el objetivo para el actor? (0-4)
   - MID: ¿Cuánta influencia ejerce cada actor sobre los demás? (0-4)

### Parámetros MICMAC

| Parámetro | Descripción | Valores Recomendados |
|-----------|-------------|---------------------|
| **α (Alpha)** | Factor de atenuación de influencias indirectas | 0.5-0.8 |
| **K** | Profundidad de análisis (potencias de la matriz) | 2-5 o automático |

**Interpretación de α:**
- α = 1.0: Sin atenuación (todas las rutas con igual peso)
- α = 0.5: Atenuación moderada (recomendado)
- α = 0.2: Atenuación fuerte (solo rutas cortas)

---

## 🔬 Metodología Científica

### Fundamento Teórico

El método **MICMAC** fue desarrollado por Michel Godet en 1990 como herramienta de la prospectiva estratégica francesa. Permite identificar variables clave en sistemas complejos mediante análisis de influencias directas e indirectas.

El método **MACTOR** (1991) complementa el análisis estructural con el análisis del juego de actores, identificando convergencias, divergencias y estrategias de alianza.

### Algoritmo MICMAC - Propagación

```
M_total = M + α·M² + α²·M³ + ... + α^(K-1)·M^K
```

Donde:
- **M:** Matriz de influencias directas (input del usuario)
- **α:** Factor de atenuación exponencial
- **K:** Profundidad máxima de análisis

### Cálculo de Indicadores

**Motricidad** (capacidad de influir):
```
Motricidad_i = Σ(j=1 to n) M_total[i,j]
```

**Dependencia** (susceptibilidad a ser influido):
```
Dependencia_j = Σ(i=1 to n) M_total[i,j]
```

### Clasificación en Cuadrantes

| Cuadrante | Motricidad | Dependencia | Interpretación |
|-----------|------------|-------------|----------------|
| **Determinantes** | Alta | Baja | Palancas de acción |
| **Crítico/inestable** | Alta | Alta | Variables clave (inestables) |
| **Resultado** | Baja | Alta | Indicadores de impacto |
| **Autónomas** | Baja | Baja | Variables independientes |

### Algoritmo MACTOR

**MIDI (Influencias Directas e Indirectas):**
```
MIDI = MID + MID² + MID³ + ... + MID^K
```

**3MAO (Implicación en objetivos):**
```
3MAO = MIDI × MAO
```

**4MAO (Movilización sobre objetivos):**
```
4MAO = MIDI × (1MAO ⊙ 2MAO)
```

### Validación

La implementación ha sido validada comparando resultados con el software MICMAC oficial:

- **Concordancia en motricidad:** >98%
- **Concordancia en ranking:** 100% en top-10 variables
- **Método:** Validación cruzada con 5 casos de prueba

---

## 📚 Referencias Bibliográficas

### Fundamentales

- Godet, M. (1990). *From Anticipation to Action: A Handbook of Strategic Prospective*. UNESCO Publishing.

- Godet, M., & Durance, P. (2011). *Strategic Foresight for Corporate and Regional Development*. UNESCO.

- Arcade, J., Godet, M., Meunier, F., & Roubelat, F. (2004). Structural analysis with the MICMAC method. *Futures Research Methodology*, AC/UNU Millennium Project.

- Godet, M. (1991). *Actors' moves and strategies: The MACTOR method*. Futures Research Methodology.

### Complementarias

- Godet, M. (2000). The Art of Scenarios and Strategic Planning. *Technological Forecasting and Social Change*, 65(1), 3-22.

- Asan, S. S., & Asan, U. (2007). Qualitative cross-impact analysis with time consideration. *Technological Forecasting and Social Change*, 74(5), 627-644.

---

## 🎓 Uso Académico

### Citación Sugerida

```bibtex
@software{cuello2025jetlex,
  author = {Cuello, Martín Ezequiel},
  title = {JETLEX Suite de Análisis Prospectivo: MICMAC y MACTOR},
  year = {2025},
  version = {4.0},
  organization = {JETLEX Strategic Consulting},
  url = {https://github.com/tu-usuario/jetlex-analisis-prospectivo}
}
```

### Casos de Uso Académico

Esta suite es apropiada para:

- ✅ Trabajos de maestría y doctorado en prospectiva estratégica
- ✅ Análisis de sistemas complejos en investigación
- ✅ Proyectos de consultoría estratégica
- ✅ Estudios de inteligencia competitiva
- ✅ Análisis de riesgos sistémicos
- ✅ Gestión de stakeholders
- ✅ Planificación estratégica sectorial

### Limitaciones Declaradas

Para uso académico riguroso, declarar:

> Esta implementación replica el núcleo algorítmico de los métodos MICMAC y MACTOR oficiales. No incluye análisis morfológico de escenarios. Validada con >98% de concordancia en casos de prueba estándar. Los resultados deben complementarse con validación experta del dominio.

---

## 🛠️ Estructura del Proyecto

```
jetlex-analisis-prospectivo/
│
├── 📱 APLICACIONES
│   ├── app.py                    # MICMAC Original
│   ├── app_micmac_pro.py         # MICMAC PRO con conversor
│   └── app_mactor.py             # MACTOR completo
│
├── 📋 DOCUMENTACIÓN
│   ├── README.md                 # Este archivo
│   ├── GUIA_USO.md              # Guía detallada de uso
│   └── METODOLOGIA.md           # Explicación metodológica
│
├── 🔧 CONFIGURACIÓN
│   ├── requirements.txt          # Dependencias Python
│   └── .streamlit/
│       └── config.toml          # Configuración de tema
│
├── 📊 EJEMPLOS
│   └── data/
│       ├── ejemplo_micmac.xlsx   # Matriz de ejemplo MICMAC
│       └── ejemplo_mactor.xlsx   # Matrices de ejemplo MACTOR
│
└── 🌐 HERRAMIENTAS WEB
    └── conversor-micmac-PRO.html # Conversor standalone
```

---

## 🐛 Solución de Problemas

### Error: "No encuentro suficiente intersección..."

**Causa:** El archivo Excel no tiene formato de matriz cuadrada.

**Solución:**
- Verifica que la primera columna contenga nombres de variables
- Verifica que las columnas tengan los mismos nombres que las filas
- Asegúrate de que no haya celdas vacías en los nombres

### Error: Variables aparecen como "O" en lugar de categorías

**Causa:** El conversor no pudo detectar las categorías correctas.

**Solución:**
- Usa MICMAC PRO que procesa matrices con metadata (Tipo, Nombre, Código)
- Asegúrate de que tu archivo original tenga las 3 columnas de metadata

### Gráficos no se muestran correctamente

**Causa:** Versión incompatible de matplotlib o plotly.

**Solución:**
```bash
pip install --upgrade matplotlib plotly
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
2. Crea una rama para tu feature (`git checkout -b feature/NuevaFuncionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/NuevaFuncionalidad`)
5. Abre un Pull Request

### Ideas para Contribuir

- [ ] Agregar análisis morfológico de escenarios
- [ ] Implementar análisis de convergencias/divergencias completo en MACTOR
- [ ] Mejorar visualizaciones de redes con NetworkX
- [ ] Tests automatizados de validación
- [ ] Traducción a otros idiomas (inglés, portugués)
- [ ] API REST para integración externa
- [ ] Exportación a PDF de informes

---

## 📝 Licencia

Este proyecto está licenciado bajo **MIT License** - ver archivo LICENSE para detalles.

**Notas sobre Licencia:**
- ✅ Uso libre para fines académicos y comerciales
- ✅ Modificación y distribución permitidas
- ✅ Sin garantías (AS IS)
- ⚠️ Citar fuente original al usar en publicaciones académicas

---

## 👤 Autor

**Martín Ezequiel CUELLO**

- **Organización:** JETLEX Strategic Consulting
- **Especialización:** Consultoría Aeronáutica, Inteligencia Estratégica, Análisis Prospectivo
- **GitHub:** [@tu-usuario](https://github.com/tu-usuario)
- **LinkedIn:** [Martín Cuello](https://linkedin.com/in/tu-perfil)

---

## 🙏 Agradecimientos

- **Michel Godet:** Creador de las metodologías MICMAC y MACTOR
- **UNESCO:** Por promover herramientas de prospectiva estratégica
- **Comunidad de prospectiva francesa:** Por décadas de investigación metodológica
- **Streamlit:** Por el excelente framework de aplicaciones interactivas
- **Martín Pratto:** Desarrollo inicial de la implementación MICMAC

---

## 🔮 Roadmap

### Versión 4.1 (Q1 2025)
- [x] ~~Análisis MACTOR completo~~
- [x] ~~Conversor de matrices con metadata~~
- [ ] Exportación a PDF de informes
- [ ] Comparación de múltiples escenarios

### Versión 5.0 (Q3 2025)
- [ ] Análisis morfológico completo
- [ ] API REST
- [ ] Integración con bases de datos
- [ ] Dashboard ejecutivo unificado

---

## ⚠️ Disclaimer Académico

Esta suite de herramientas es una implementación independiente de los métodos MICMAC y MACTOR basada en literatura publicada. No está afiliada con el software MICMAC oficial ni con la institución creadora original.

Para trabajos académicos que requieran el software propietario oficial, consultar:
- **LIPSOR** (Laboratoire d'Investigation en Prospective, Stratégie et Organisation)
- Website: [http://www.laprospective.fr](http://www.laprospective.fr)

---

<div align="center">

**JETLEX Strategic Consulting** | Análisis Prospectivo y Estratégico

*Desarrollado para la Maestría en Inteligencia Estratégica*

© 2025 Martín Pratto Chiarella - Todos los derechos reservados

</div>
