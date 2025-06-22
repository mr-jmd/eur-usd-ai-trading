#!/usr/bin/env python3
"""Script de instalación automática"""

import subprocess
import sys

def install_requirements():
    """Instala dependencias"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

if __name__ == "__main__":
    install_requirements()
