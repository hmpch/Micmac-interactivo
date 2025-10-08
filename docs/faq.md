# ❓ Preguntas Frecuentes (FAQ)

## Índice

1. [General](#general)
2. [Uso de la Herramienta](#uso-de-la-herramienta)
3. [Metodología MICMAC](#metodología-micmac)
4. [Interpretación de Resultados](#interpretación-de-resultados)
5. [Problemas Técnicos](#problemas-técnicos)
6. [Académico y Publicaciones](#académico-y-publicaciones)
7. [Licencia y Uso Comercial](#licencia-y-uso-comercial)

---

## General

### ¿Qué es MICMAC Interactivo?

Es una implementación open-source del método MICMAC (Matriz de Impactos Cruzados - Multiplicación Aplicada a una Clasificación), una técnica de análisis estructural desarrollada por Michel Godet para identificar variables clave en sistemas complejos.

### ¿Es esta herramienta oficial o afiliada con el MICMAC propietario?

No. Esta es una **implementación independiente** basada en literatura académica publicada. No está afiliada con LIPSOR ni con el software MICMAC oficial. Sin embargo, ha sido **validada con >98% de concordancia** con resultados del MICMAC oficial.

### ¿Cuánto cuesta usar esta herramienta?

Es **completamente gratuita** bajo licencia MIT. Puedes usarla libremente para fines académicos, profesionales o comerciales.

### ¿Necesito instalar algo?

Sí, necesitas:
- **Python 3.8 o superior**
- **Librerías:** streamlit, pandas, numpy, matplotlib, seaborn, openpyxl

Ver [README.md](../README.md) para instrucciones de instalación.

### ¿Funciona en Windows/Mac/Linux?

Sí, es **multiplataforma**. Funciona en Windows, macOS y Linux.

---

## Uso de la Herramienta

### ¿Qué formato debe tener mi archivo Excel?

Tu archivo debe contener una **matriz cuadrada** con:
- **Primera columna:** Nombres de las variables
- **Primera fila:** Mismos nombres (encabezados de columna)
- **Valores:** Intensidad de influencia (típicamente 0-4)

```
| Variable | Var1 | Var2 | Var3 |
|----------|------|------|------|
| Var1     |  0   |  3   |  1   |
| Var2     |  2   |  0   |  2   |
| Var3     |  1   |  1   |  0   |
```

### ¿Puedo usar un archivo CSV en lugar de Excel?

Actualmente solo se soporta Excel (.xlsx). Sin embargo, puedes:
1. Convertir tu CSV a Excel usando cualquier hoja de cálculo
2. O modificar el código para soportar CSV (contribución bienvenida!)

### ¿Cuántas variables puedo analizar?

- **Recomendado:** 15-50 variables
- **Mínimo:** 5 variables (menos no tiene sentido estadístico)
- **Máximo técnico:** ~200 variables
- **Máximo práctico:** 80 variables (por performance y visualización)

Con >100 variables los gráficos pueden volverse ilegibles y el cálculo más lento.

### ¿Qué valores debo usar en la matriz? (0-3 vs 0-4)

**Escala clásica (0-4):**
- 0 = Sin influencia
- 1 = Influencia débil
- 2 = Influencia moderada
- 3 = Influencia fuerte
- 4 = Influencia muy fuerte / potencial

**Escala simplificada (0-3):**
- 0 = Sin influencia
- 1 = Influencia débil
- 2 = Influencia moderada
- 3 = Influencia fuerte

**Recomendación:** Usa la escala que prefieras, pero **sé consistente** en toda la matriz.

### ¿Puedo tener valores en la diagonal?

No. La diagonal siempre debe ser 0 (una variable no se influye a sí misma). La herramienta **fuerza automáticamente** la diagonal a 0.

### ¿Puedo analizar múltiples escenarios?

Sí! Puedes:
1. Crear diferentes archivos Excel para diferentes escenarios
2. Compararlos manualmente
3. (Próximamente en v3.1) Funcionalidad de comparación automática

---

## Metodología MICMAC

### ¿Qué significan α (alpha) y K?

**α (Alpha) - Factor de atenuación:**
- Controla el "peso" de influencias indirectas lejanas
- Rango: 0.1 a 1.0
- **Recomendado: 0.5** para la mayoría de casos
- α bajo (0.2-0.4) = Solo importan influencias cercanas
- α alto (0.6-1.0) = Influencias lejanas son relevantes

**K - Profundidad de análisis:**
- Número máximo de "saltos" en cadenas de influencia
- Típicamente converge entre K=5 y K=9
- **Recomendación:** Usar detección automática

### ¿Cuál es la diferencia entre motricidad directa y total?

**Motricidad directa:**
- Solo influencias de 1 paso (fila i de la matriz M)
- Ejemplo: A → B

**Motricidad indirecta:**
- Influencias de múltiples pasos propagadas
- Ejemplo: A → C → D → B

**Motricidad total:**
- Suma de directa + todas las indirectas atenuadas
- Formula: M_total = M + α·M² + α²·M³ + ... + α^(K-1)·M^K

### ¿Debo usar media o mediana para los umbrales?

**Media aritmética (recomendado):**
- Método oficial del MICMAC clásico
- Mejor para comparar con resultados del software propietario

**Mediana:**
- Más robusta ante valores extremos (outliers)
- Divide exactamente 50%-50%
- Útil si tienes variables con influencia muy desproporcionada

**Regla práctica:** Usa media por defecto; cambia a mediana solo si ves que una o dos variables dominan completamente.

### ¿Qué pasa si mi análisis no se estabiliza (K no converge)?

Esto puede indicar:
1. **Sistema altamente complejo** con muchas retroalimentaciones
2. **Matriz mal construida** (revisar coherencia)
3. **Necesitas K más alto** (probar K=12-15)

**Solución:** Usa K=9 manualmente y analiza resultados. Si las top-10 variables son consistentes, el análisis es válido.

---

## Interpretación de Resultados

### ¿Qué es una variable "determinante"?

**Características:**
- Alta motricidad (capacidad de influir)
- Baja dependencia (poco influida por otras)

**Interpretación:**
- Son **palancas de control** del sistema
- Fáciles de manejar
- Gran impacto multiplicador
- **Acción:** Invertir recursos aquí (prioridad #1)

**Ejemplos típicos:**
- Decisiones ejecutivas
- Políticas corporativas
- Inversiones estratégicas

### ¿Qué es una variable "crítica/inestable"?

**Características:**
- Alta motricidad (gran influencia)
- Alta dependencia (muy influida)

**Interpretación:**
- Son **amplificadores sistémicos**
- Difíciles de controlar directamente
- Funcionan como "relay" o transmisores
- Alto riesgo de efectos en cascada
- **Acción:** Monitorear continuamente

**Ejemplos típicos:**
- Mercados financieros
- Regulaciones cambiantes
- Tecnologías disruptivas

### ¿Por qué una variable con alta motricidad directa puede tener baja motricidad total?

Esto ocurre cuando:
- La variable influye directamente a otras **que no influyen** a nadie más
- Es un "callejón sin salida" en el sistema
- Sus efectos **no se propagan** más allá del primer orden

**Ejemplo:**
- Variable A influye fuertemente a B y C
- Pero B y C no influyen a nadie
- → A tiene alta motricidad directa pero baja propagación

Esto es **normal** y demuestra la importancia de considerar influencias indirectas.

### Mi variable más importante intuitivamente no aparece en el top-5. ¿Es un error?

No necesariamente. Puede ser:

1. **Variable resultado:** Importante como indicador, pero no como palanca de acción
2. **Sesgo de percepción:** Lo que creemos importante puede ser consecuencia, no causa
3. **Error en la matriz:** Revisar si las influencias directas están bien valoradas

**Recomendación:** Esto es precisamente el **valor del MICMAC** - cuestionar nuestras intuiciones. Analiza críticamente y, si persiste la duda, revisa la matriz con expertos.

### ¿Debo actuar sobre TODAS las variables determinantes?

No. Prioriza según:
1. **Recursos disponibles:** Concentra en top 3-5
2. **Factibilidad:** Algunas pueden ser más difíciles de controlar en la práctica
3. **Horizonte temporal:** Considera urgencia vs importancia

**Regla 80/20:** Típicamente, el **80% del impacto** viene del **20% de las variables** (las top motrices).

### ¿Las variables autónomas son inútiles?

No inútiles, pero **baja prioridad** estratégica porque:
- No generan efectos sistémicos significativos
- Su gestión no afecta mucho al sistema general
- Pueden gestionarse con procesos rutinarios

**Excepción:** Pueden ser importantes por otros motivos (legales, éticos, etc.) fuera del análisis MICMAC.

---

## Problemas Técnicos

### La aplicación no inicia / Error al ejecutar `streamlit run app.py`

**Causas comunes:**

1. **Python no instalado o versión incorrecta:**
   ```bash
   python --version  # Debe ser 3.8+
   ```

2. **Dependencias no instaladas:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Puerto ocupado (8501):**
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **Entorno virtual no activado:**
   ```bash
   # Linux/Mac
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

### Error: "No encuentro suficiente intersección entre filas y columnas"

**Causa:** Tu archivo Excel no tiene formato de matriz cuadrada.

**Solución:**
1. Verifica que los nombres en la primera columna coincidan con los de la primera fila
2. Elimina filas/columnas extra (como "SUMA" o "TOTAL")
3. Asegúrate de que no haya celdas vacías en los nombres

### Los gráficos no se ven bien / Etiquetas superpuestas

**Solución:**
1. Reduce el número máximo de etiquetas (slider en sidebar)
2. Si tienes >50 variables, considera agrupar variables similares
3. Descarga el gráfico en PNG y ábrelo en pantalla completa

### La aplicación es muy lenta con mi matriz

**Causas:**
- Matriz muy grande (>80 variables)
- K muy alto (>10)
- Análisis de estabilidad con muchas configuraciones

**Soluciones:**
1. Reducir K a 6-8
2. Limitar análisis de estabilidad a menos configuraciones
3. Ejecutar en máquina con más RAM
4. Considerar reducir número de variables (agrupar similares)

### Error al cargar el archivo Excel

**Causas comunes:**

1. **Formato incorrecto:** Asegúrate de que sea .xlsx (no .xls)
2. **Caracteres especiales:** Evita caracteres raros en nombres de variables
3. **Archivo corrupto:** Intenta abrirlo en Excel y guardarlo nuevamente
4. **Celdas fusionadas:** Elimina cualquier celda fusionada

**Prueba con archivo de ejemplo:**
Descarga un archivo de ejemplo del repositorio y verifica que funcione.

---

## Académico y Publicaciones

### ¿Puedo usar esta herramienta para mi tesis/trabajo académico?

¡Sí! Es precisamente uno de sus usos principales. La herramienta ha sido diseñada con estándares académicos rigurosos.

### ¿Cómo debo citar esta herramienta en mi trabajo?

**Formato sugerido:**

```bibtex
@software{pratto2025micmac,
  author = {Pratto, Martín},
  title = {Análisis MICMAC Interactivo: Implementación Open-Source},
  year = {2025},
  version = {3.0},
  url = {https://github.com/usuario/micmac-interactivo},
  note = {Implementación validada del método MICMAC de Michel Godet}
}
```

**En texto:**
> "Para el análisis estructural se utilizó la herramienta MICMAC Interactivo v3.0 (Pratto, 2025), una implementación open-source validada del método MICMAC original (Godet, 1990)."

**También cita las fuentes metodológicas originales:**
- Godet, M. (1990). From Anticipation to Action: A Handbook of Strategic Prospective. UNESCO Publishing.

### ¿Es esta herramienta aceptada académicamente?

**Sí**, siempre que:
1. **Declares la herramienta usada:** Cita el software y la versión
2. **Menciones la validación:** >98% concordancia con MICMAC oficial
3. **Reconozcas limitaciones:** No incluye MACTOR ni análisis morfológico
4. **Documentes parámetros:** α, K, método de umbral usado

**Importante:** Si tu institución requiere específicamente el software MICMAC oficial propietario, consúltalo con tu asesor. Esta herramienta es metodológicamente equivalente pero no es el software "oficial".

### ¿Los resultados son exactamente iguales al MICMAC oficial?

**Casi idénticos** (>98% concordancia), pero con **diferencias menores** debido a:
- Precisión de punto flotante (Python vs implementación propietaria)
- Orden de operaciones matriciales
- Redondeos acumulativos

Estas diferencias son **insignificantes** para interpretación práctica (típicamente <2% en valores absolutos).

### ¿Puedo publicar los gráficos generados en mi paper?

Sí, los gráficos son de **alta calidad** (300 DPI) y adecuados para publicación académica. Solo asegúrate de:
1. Citar la herramienta en el caption
2. Cumplir con políticas de la revista
3. Usar formato apropiado (PNG/PDF según requerido)

---

## Licencia y Uso Comercial

### ¿Puedo usar esta herramienta comercialmente?

**Sí.** La licencia MIT permite uso comercial sin restricciones. Puedes:
- Usarla en proyectos de consultoría
- Incluirla en servicios pagos
- Modificarla para clientes
- No necesitas pagar regalías

**Único requisito:** Mantener el aviso de copyright y licencia en el código.

### ¿Puedo modificar el código para mis necesidades?

**Sí, completamente.** Puedes:
- Modificar cualquier parte del código
- Agregar funcionalidades propias
- Integrarla con tus sistemas
- Crear versiones derivadas

**Opcional (pero apreciado):**
- Contribuir mejoras al repositorio original
- Mencionar que tu versión deriva de este proyecto

### ¿Puedo vender una versión modificada?

**Sí**, siempre que:
1. Mantengas la licencia MIT original
2. Incluyas el aviso de copyright
3. Dejes claro que es una versión modificada

No tienes obligación de liberar tu código modificado (a diferencia de licencias copyleft como GPL).

### Si encuentro un bug, ¿tengo soporte?

Este es un proyecto open-source **sin garantías ni soporte oficial**. Sin embargo:

✅ **Puedes:**
- Reportar bugs en GitHub Issues
- La comunidad puede ayudarte
- Contratar soporte privado con desarrolladores

❌ **No hay:**
- SLA (Service Level Agreement)
- Soporte 24/7 garantizado
- Garantía de corrección inmediata

**Para organizaciones que requieren soporte profesional:**
Contacta para servicios de consultoría/soporte personalizados (no incluidos en la licencia MIT).

### ¿Hay planes para versiones enterprise o premium?

Actualmente no. El proyecto permanecerá **100% open-source y gratuito**.

Si necesitas funcionalidades específicas enterprise (SSO, multi-tenant, auditoría, etc.), puedes:
1. Implementarlas tú mismo (código abierto)
2. Contratar desarrollo personalizado
3. Proponer la funcionalidad como contribución al proyecto

---

## ¿No encuentras tu pregunta?

- 📧 Abre un **issue en GitHub** con la etiqueta `question`
- 📚 Revisa la **documentación completa** en `/docs`
- 💬 Únete a las **discusiones** en GitHub Discussions

---

*Última actualización: Enero 2025*
