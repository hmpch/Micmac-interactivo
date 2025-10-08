#!/bin/bash
# ============================================================
# Script de Instalación Rápida - MICMAC Interactivo
# ============================================================

echo "=========================================="
echo "  MICMAC Interactivo - Quick Start"
echo "  by Martín Pratto"
echo "=========================================="
echo ""

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Verificar Python
echo -e "${YELLOW}[1/5] Verificando Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 no encontrado. Por favor instala Python 3.8 o superior.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION encontrado${NC}"

# Crear entorno virtual (opcional pero recomendado)
echo -e "\n${YELLOW}[2/5] ¿Deseas crear un entorno virtual? (recomendado)${NC}"
read -p "Crear entorno virtual (s/n): " CREATE_VENV

if [[ "$CREATE_VENV" == "s" || "$CREATE_VENV" == "S" ]]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
    
    # Activar entorno virtual
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    echo -e "${GREEN}✓ Entorno virtual creado y activado${NC}"
else
    echo "Continuando sin entorno virtual..."
fi

# Actualizar pip
echo -e "\n${YELLOW}[3/5] Actualizando pip...${NC}"
python3 -m pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip actualizado${NC}"

# Instalar dependencias
echo -e "\n${YELLOW}[4/5] Instalando dependencias...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Dependencias instaladas${NC}"
else
    echo -e "${RED}✗ Archivo requirements.txt no encontrado${NC}"
    echo "Instalando dependencias básicas..."
    pip install streamlit pandas numpy matplotlib seaborn openpyxl --quiet
    echo -e "${GREEN}✓ Dependencias básicas instaladas${NC}"
fi

# Crear directorios necesarios
echo -e "\n${YELLOW}[5/5] Configurando estructura de proyecto...${NC}"
mkdir -p examples
mkdir -p docs
mkdir -p .streamlit

# Crear archivo de configuración de Streamlit
cat > .streamlit/config.toml << EOF
[theme]
primaryColor = "#1166CC"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 50

[browser]
gatherUsageStats = false
EOF

echo -e "${GREEN}✓ Configuración completada${NC}"

# Verificar instalación
echo -e "\n${YELLOW}Verificando instalación...${NC}"
python3 -c "import streamlit; import pandas; import numpy; import matplotlib; import seaborn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Todas las librerías instaladas correctamente${NC}"
else
    echo -e "${RED}✗ Error en la instalación de algunas librerías${NC}"
    exit 1
fi

# Instrucciones finales
echo ""
echo "=========================================="
echo -e "${GREEN}✓ INSTALACIÓN COMPLETADA${NC}"
echo "=========================================="
echo ""
echo "Para ejecutar la aplicación:"
echo ""
echo "  1. Si creaste entorno virtual, actívalo:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "     > venv\\Scripts\\activate"
else
    echo "     $ source venv/bin/activate"
fi
echo ""
echo "  2. Ejecuta la aplicación:"
echo "     $ streamlit run app.py"
echo ""
echo "  3. La app se abrirá en tu navegador automáticamente"
echo "     (normalmente en http://localhost:8501)"
echo ""
echo "=========================================="
echo ""
echo "¿Deseas ejecutar la app ahora? (s/n)"
read -p "> " RUN_NOW

if [[ "$RUN_NOW" == "s" || "$RUN_NOW" == "S" ]]; then
    echo ""
    echo "Iniciando MICMAC Interactivo..."
    streamlit run app.py
fi
