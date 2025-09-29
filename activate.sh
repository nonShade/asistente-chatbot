#!/bin/bash
# Script de conveniencia para usar el chatbot

source .venv/bin/activate
export USE_QDRANT=true
export QDRANT_HOST=localhost
export QDRANT_PORT=6333

echo "🤖 Entorno activado - Variables configuradas:"
echo "   USE_QDRANT=true"
echo "   QDRANT_HOST=localhost"
echo ""
echo "📋 Comandos disponibles:"
echo "   python app.py                    # Chatbot interactivo"
echo "   python scripts/build_index.py   # Reconstruir índices"
echo "   python app.py --mode batch      # Evaluación batch"
echo ""
echo "🔹 Para salir: deactivate"
echo ""

# Mantener la shell activa con las variables de entorno
exec "$SHELL"