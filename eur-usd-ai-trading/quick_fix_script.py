"""
Script de Corrección Rápida para Errores Detectados
Ejecutar este script para corregir los problemas automáticamente
"""

import sqlite3
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_database_schema():
    """Corrige el esquema de la base de datos"""
    logger.info("🔧 Corrigiendo esquema de base de datos...")
    
    db_path = "trading_data.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar si la tabla existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'")
        if not cursor.fetchone():
            logger.info("Tabla price_data no existe, creándola...")
            cursor.execute('''
                CREATE TABLE price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    dividends REAL DEFAULT 0,
                    stock_splits REAL DEFAULT 0,
                    repaired INTEGER DEFAULT 0,
                    sma_20 REAL,
                    sma_50 REAL,
                    sma_200 REAL,
                    ema_12 REAL,
                    ema_26 REAL,
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    bb_width REAL,
                    bb_position REAL,
                    stoch_k REAL,
                    stoch_d REAL,
                    atr REAL
                )
            ''')
            logger.info("✅ Tabla price_data creada")
        else:
            # Agregar columnas faltantes
            cursor.execute("PRAGMA table_info(price_data)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            new_columns = {
                'repaired': 'INTEGER DEFAULT 0',
                'dividends': 'REAL DEFAULT 0',
                'stock_splits': 'REAL DEFAULT 0'
            }
            
            for col_name, col_type in new_columns.items():
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE price_data ADD COLUMN {col_name} {col_type}")
                        logger.info(f"✅ Agregada columna {col_name}")
                    except sqlite3.OperationalError as e:
                        logger.warning(f"No se pudo agregar columna {col_name}: {e}")
        
        conn.commit()
        conn.close()
        logger.info("✅ Base de datos corregida")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error corrigiendo base de datos: {e}")
        return False

def update_config_file():
    """Actualiza el archivo config.json"""
    logger.info("📝 Actualizando config.json...")
    
    config = {
        "data_sources": {
            "price_symbol": "EURUSD=X",
            "use_mock_data": False,
            "news_api_key": None,
            "twitter_api_key": None,
            "update_interval_minutes": 60
        },
        "model_settings": {
            "ensemble_weights": {"lstm": 0.4, "gru": 0.4, "rf": 0.2},
            "min_confidence_threshold": 0.65,
            "retrain_interval_days": 7,
            "sequence_length": 60
        },
        "trading_settings": {
            "auto_trading_enabled": False,
            "max_position_size": 0.1,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "risk_per_trade": 0.02,
            "demo_mode": True
        },
        "sentiment_settings": {
            "sentiment_weight": 0.3,
            "news_lookback_hours": 24,
            "min_impact_score": 0.5,
            "use_mock_sentiment": True
        },
        "dashboard": {
            "enable_realtime": True,
            "auto_refresh_interval_seconds": 30,
            "max_prediction_history": 100,
            "show_advanced_metrics": True,
            "enable_live_charts": True,
            "cache_predictions": True,
            "prediction_cache_ttl_seconds": 120,
            "host": "0.0.0.0",
            "port": 8501,
            "theme": "light"
        },
        "database": {
            "path": "trading_data.db",
            "backup_interval_hours": 6,
            "auto_backup": True
        },
        "alerts": {
            "email_enabled": False,
            "console_alerts": True,
            "dashboard_notifications": True
        },
        "development": {
            "debug_mode": True,
            "log_level": "INFO",
            "save_predictions": True
        }
    }
    
    try:
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info("✅ config.json actualizado")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error actualizando config.json: {e}")
        return False

def cleanup_old_database():
    """Limpia registros problemáticos de la base de datos"""
    logger.info("🧹 Limpiando base de datos...")
    
    try:
        conn = sqlite3.connect("trading_data.db")
        cursor = conn.cursor()
        
        # Eliminar registros con datos problemáticos si existen
        cursor.execute("DELETE FROM price_data WHERE timestamp IS NULL")
        deleted = cursor.rowcount
        
        if deleted > 0:
            logger.info(f"🗑️ Eliminados {deleted} registros problemáticos")
        
        conn.commit()
        conn.close()
        logger.info("✅ Base de datos limpia")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Error limpiando base de datos: {e}")
        return False

def main():
    """Ejecuta todas las correcciones"""
    print("🔧 EUR/USD AI Trading System - Corrección Rápida")
    print("=" * 60)
    
    fixes = [
        ("Base de Datos", fix_database_schema),
        ("Configuración", update_config_file),
        ("Limpieza", cleanup_old_database)
    ]
    
    success_count = 0
    
    for name, fix_func in fixes:
        print(f"\n🔧 Aplicando corrección: {name}...")
        try:
            if fix_func():
                success_count += 1
                print(f"✅ {name} corregido")
            else:
                print(f"❌ Error en {name}")
        except Exception as e:
            print(f"❌ Error en {name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 CORRECCIONES COMPLETADAS: {success_count}/{len(fixes)}")
    
    if success_count == len(fixes):
        print("🎉 ¡Todas las correcciones aplicadas exitosamente!")
        print("\n🚀 Ahora puedes ejecutar:")
        print("   python run_system_complete.py")
        print("   O")
        print("   streamlit run realtime_dashboard.py")
    else:
        print("⚠️ Algunas correcciones fallaron. Revisar logs.")
    
    return success_count == len(fixes)

if __name__ == "__main__":
    main()
