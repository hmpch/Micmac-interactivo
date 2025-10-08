@echo off
REM ============================================================
REM Script de Instalación Rápida - MICMAC Interactivo (Windows)
REM ============================================================

echo ==========================================
echo   MICMAC Interactivo - Quick Start
echo   by Martin Pratto
echo ==========================================
echo.

REM Verificar Python
echo [1/5] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Por favor instala Python 3.8 o superior.
    echo Descarga desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo [OK] Python encontrado
echo.

REM Preguntar por entorno virtual
echo [2/5] Crear entorno virtual? (recomendado)
set /p CREATE_VENV="Crear entorno virtual (S/N): "

if /i "%CREATE_VENV%"=="S" (
    echo Creando entorno virtual...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [OK] Entorno virtual creado y activado
) else (
    echo Continuando sin entorno virtual...
)
echo.

REM Actualizar pip
echo [3/5] Actualizando pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip actualizado
echo.

REM Instalar dependencias
echo [4/5] Instalando dependencias...
if exist requirements.txt (
    pip install -r requirements.txt --quiet
    echo [OK] Dependencias instaladas desde requirements.txt
) else (
    echo [AVISO] requirements.txt no encontrado
    echo Instalando dependencias basicas...
    pip install streamlit pandas numpy matplotlib seaborn openpyxl --quiet
    echo [OK] Dependencias basicas instaladas
)
echo.

REM Crear directorios
echo [5/5] Configurando estructura de proyecto...
if not exist examples mkdir examples
if not exist docs mkdir docs
if not exist .streamlit mkdir .streamlit

REM Crear configuración de Streamlit
(
echo [theme]
echo primaryColor = "#1166CC"
echo backgroundColor = "#FFFFFF"
echo secondaryBackgroundColor = "#F0F2F6"
echo textColor = "#262730"
echo font = "sans serif"
echo.
echo [server]
echo maxUploadSize = 50
echo.
echo [browser]
echo gatherUsageStats = false
) > .streamlit\config.toml

echo [OK] Configuracion completada
echo.

REM Verificar instalación
echo Verificando instalacion...
python -c "import streamlit; import pandas; import numpy; import matplotlib; import seaborn" 2>nul
if errorlevel 1 (
    echo [ERROR] Error en la instalacion de algunas librerias
    pause
    exit /b 1
)
echo [OK] Todas las librerias instaladas correctamente
echo.

REM Instrucciones finales
echo ==========================================
echo [OK] INSTALACION COMPLETADA
echo ==========================================
echo.
echo Para ejecutar la aplicacion:
echo.
if /i "%CREATE_VENV%"=="S" (
    echo   1. Activa el entorno virtual:
    echo      ^> venv\Scripts\activate.bat
    echo.
)
echo   2. Ejecuta la aplicacion:
echo      ^> streamlit run app.py
echo.
echo   3. La app se abrira automaticamente en tu navegador
echo      ^(normalmente en http://localhost:8501^)
echo.
echo ==========================================
echo.

set /p RUN_NOW="Deseas ejecutar la app ahora? (S/N): "

if /i "%RUN_NOW%"=="S" (
    echo.
    echo Iniciando MICMAC Interactivo...
    streamlit run app.py
) else (
    echo.
    echo Para iniciar la app mas tarde, ejecuta: streamlit run app.py
    pause
)
