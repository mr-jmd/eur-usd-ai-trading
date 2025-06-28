
"""Dashboard en Tiempo Real - Archivo independiente"""
import sys
from pathlib import Path

# Agregar directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Importar y ejecutar dashboard
from realtime_dashboard import main

if __name__ == "__main__":
    main()
