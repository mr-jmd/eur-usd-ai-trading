#!/usr/bin/env python3
"""Script para ejecutar el dashboard"""

import subprocess
import sys
import os

def run_dashboard():
    """Ejecuta el dashboard de Streamlit"""
    dashboard_path = os.path.join("src", "dashboard", "app.py")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        dashboard_path, "--server.port", "8501"
    ])

if __name__ == "__main__":
    run_dashboard()
