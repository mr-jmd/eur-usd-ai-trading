#!/usr/bin/env python3
"""Script para configurar la base de datos"""

import sqlite3
from pathlib import Path

def setup_database():
    """Configura la base de datos inicial"""
    db_path = Path("data/trading_data.db")
    db_path.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    # Crear tablas aquí...
    conn.close()
    print("✅ Base de datos configurada")

if __name__ == "__main__":
    setup_database()
