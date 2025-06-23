"""
Utilidades del sistema para verificación y diagnóstico
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yfinance as yf

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemDiagnostics:
    """Herramientas de diagnóstico del sistema"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {}
    
    def check_dependencies(self):
        """Verifica que todas las dependencias estén instaladas"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'pandas', 'numpy', 'tensorflow', 'sklearn', 'yfinance',
            'requests', 'sqlite3', 'asyncio', 'aiohttp', 'streamlit',
            'plotly', 'joblib'
        ]
        
        missing_packages = []
        installed_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                installed_packages.append(package)
                logger.info(f"✅ {package} - OK")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"❌ {package} - MISSING")
        
        self.results['dependencies'] = {
            'installed': installed_packages,
            'missing': missing_packages,
            'status': 'OK' if not missing_packages else 'ERROR'
        }
        
        return len(missing_packages) == 0
    
    def check_data_connectivity(self):
        """Verifica conectividad con fuentes de datos"""
        logger.info("Checking data connectivity...")
        
        # Test Yahoo Finance
        try:
            ticker = yf.Ticker("EURUSD=X")
            data = ticker.history(period="5d", interval="1d")
            
            if not data.empty:
                logger.info("✅ Yahoo Finance - OK")
                yf_status = 'OK'
                yf_samples = len(data)
            else:
                logger.warning("⚠️ Yahoo Finance - No data returned")
                yf_status = 'WARNING'
                yf_samples = 0
                
        except Exception as e:
            logger.error(f"❌ Yahoo Finance - ERROR: {e}")
            yf_status = 'ERROR'
            yf_samples = 0
        
        # Test mock data generator
        try:
            sys.path.append(str(self.project_root / 'src'))
            from mock_data_generator import MockDataGenerator
            
            generator = MockDataGenerator()
            mock_data = generator.generate_price_data(periods=100, freq='1H')
            
            if not mock_data.empty:
                logger.info("✅ Mock Data Generator - OK")
                mock_status = 'OK'
                mock_samples = len(mock_data)
            else:
                logger.error("❌ Mock Data Generator - No data generated")
                mock_status = 'ERROR'
                mock_samples = 0
                
        except Exception as e:
            logger.error(f"❌ Mock Data Generator - ERROR: {e}")
            mock_status = 'ERROR'
            mock_samples = 0
        
        self.results['data_connectivity'] = {
            'yahoo_finance': {'status': yf_status, 'samples': yf_samples},
            'mock_generator': {'status': mock_status, 'samples': mock_samples},
            'overall_status': 'OK' if (yf_status == 'OK' or mock_status == 'OK') else 'ERROR'
        }
        
        return yf_status == 'OK' or mock_status == 'OK'
    
    def check_directories(self):
        """Verifica y crea directorios necesarios"""
        logger.info("Checking directories...")
        
        required_dirs = [
            'data', 'models', 'logs', 'backups', 'src', 'utils'
        ]
        
        created_dirs = []
        existing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            
            if dir_path.exists():
                existing_dirs.append(dir_name)
                logger.info(f"✅ Directory {dir_name} - EXISTS")
            else:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_name)
                    logger.info(f"✅ Directory {dir_name} - CREATED")
                except Exception as e:
                    logger.error(f"❌ Directory {dir_name} - ERROR: {e}")
        
        self.results['directories'] = {
            'existing': existing_dirs,
            'created': created_dirs,
            'status': 'OK'
        }
        
        return True
    
    def check_config_file(self):
        """Verifica el archivo de configuración"""
        logger.info("Checking configuration file...")
        
        config_path = self.project_root / 'config.json'
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                logger.info("✅ Configuration file - OK")
                
                # Verificar secciones importantes
                required_sections = [
                    'data_sources', 'model_settings', 'trading_settings',
                    'sentiment_settings', 'database', 'alerts'
                ]
                
                missing_sections = [section for section in required_sections 
                                  if section not in config]
                
                if missing_sections:
                    logger.warning(f"⚠️ Missing config sections: {missing_sections}")
                    status = 'WARNING'
                else:
                    status = 'OK'
                
                self.results['config'] = {
                    'exists': True,
                    'valid': True,
                    'missing_sections': missing_sections,
                    'status': status
                }
                
            except Exception as e:
                logger.error(f"❌ Configuration file - INVALID: {e}")
                self.results['config'] = {
                    'exists': True,
                    'valid': False,
                    'error': str(e),
                    'status': 'ERROR'
                }
                return False
        else:
            logger.warning("⚠️ Configuration file - NOT FOUND (will use defaults)")
            self.results['config'] = {
                'exists': False,
                'status': 'WARNING'
            }
        
        return True
    
    def check_model_files(self):
        """Verifica archivos de modelos pre-entrenados"""
        logger.info("Checking model files...")
        
        models_dir = self.project_root / 'models'
        model_files = [
            'lstm_model.h5',
            'gru_model.h5', 
            'rf_model.pkl',
            'feature_engineer.pkl'
        ]
        
        existing_models = []
        missing_models = []
        
        for model_file in model_files:
            model_path = models_dir / model_file
            if model_path.exists():
                existing_models.append(model_file)
                logger.info(f"✅ Model {model_file} - EXISTS")
            else:
                missing_models.append(model_file)
                logger.info(f"ℹ️ Model {model_file} - NOT FOUND (will be trained)")
        
        self.results['models'] = {
            'existing': existing_models,
            'missing': missing_models,
            'status': 'OK' if existing_models else 'INFO'
        }
        
        return True
    
    def test_mock_data_generation(self):
        """Prueba la generación de datos mock"""
        logger.info("Testing mock data generation...")
        
        try:
            sys.path.append(str(self.project_root / 'src'))
            from mock_data_generator import MockDataGenerator
            
            generator = MockDataGenerator()
            
            # Generar datos de precios
            price_data = generator.generate_price_data(periods=200, freq='1H')
            
            # Generar datos de noticias
            news_data = generator.generate_news_data(num_articles=20)
            
            # Verificar datos de precios
            price_checks = {
                'not_empty': not price_data.empty,
                'has_ohlc': all(col in price_data.columns for col in ['Open', 'High', 'Low', 'Close']),
                'has_indicators': all(col in price_data.columns for col in ['SMA_20', 'RSI', 'MACD']),
                'valid_prices': price_data['Close'].isna().sum() == 0,
                'price_range': (price_data['Close'].min() > 1.05 and price_data['Close'].max() < 1.15)
            }
            
            # Verificar datos de noticias
            news_checks = {
                'not_empty': not news_data.empty,
                'has_columns': all(col in news_data.columns for col in ['timestamp', 'title', 'sentiment_score']),
                'valid_sentiment': news_data['sentiment_score'].between(-1, 1).all(),
                'has_content': news_data['title'].notna().all()
            }
            
            all_price_checks = all(price_checks.values())
            all_news_checks = all(news_checks.values())
            
            if all_price_checks and all_news_checks:
                logger.info("✅ Mock data generation - OK")
                status = 'OK'
            else:
                logger.warning("⚠️ Mock data generation - Some issues found")
                status = 'WARNING'
            
            self.results['mock_data_test'] = {
                'price_data': {
                    'samples': len(price_data),
                    'checks': price_checks,
                    'status': 'OK' if all_price_checks else 'WARNING'
                },
                'news_data': {
                    'samples': len(news_data),
                    'checks': news_checks,
                    'status': 'OK' if all_news_checks else 'WARNING'
                },
                'overall_status': status
            }
            
            return all_price_checks and all_news_checks
            
        except Exception as e:
            logger.error(f"❌ Mock data generation - ERROR: {e}")
            self.results['mock_data_test'] = {
                'error': str(e),
                'status': 'ERROR'
            }
            return False
    
    def run_full_diagnostic(self):
        """Ejecuta diagnóstico completo del sistema"""
        logger.info("🔍 Starting system diagnostics...")
        logger.info("=" * 50)
        
        checks = [
            ("Dependencies", self.check_dependencies),
            ("Directories", self.check_directories), 
            ("Configuration", self.check_config_file),
            ("Data Connectivity", self.check_data_connectivity),
            ("Model Files", self.check_model_files),
            ("Mock Data Generation", self.test_mock_data_generation)
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_func in checks:
            logger.info(f"\n🔎 {check_name}...")
            try:
                if check_func():
                    passed_checks += 1
            except Exception as e:
                logger.error(f"❌ {check_name} check failed: {e}")
        
        # Resumen final
        logger.info("=" * 50)
        logger.info("📊 DIAGNOSTIC SUMMARY")
        logger.info("=" * 50)
        
        if passed_checks == total_checks:
            logger.info("✅ ALL CHECKS PASSED - System ready!")
            overall_status = 'READY'
        elif passed_checks >= total_checks * 0.7:  # 70% o más
            logger.info("⚠️ MOST CHECKS PASSED - System functional with warnings")
            overall_status = 'FUNCTIONAL'
        else:
            logger.info("❌ MULTIPLE ISSUES FOUND - System may not work properly")
            overall_status = 'ISSUES'
        
        logger.info(f"Passed: {passed_checks}/{total_checks} checks")
        
        self.results['summary'] = {
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results
    
    def save_diagnostic_report(self, filename: str = None):
        """Guarda reporte de diagnóstico en archivo"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnostic_report_{timestamp}.json"
        
        report_path = self.project_root / 'logs' / filename
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"📄 Diagnostic report saved: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save diagnostic report: {e}")
            return None

def create_sample_config():
    """Crea archivo de configuración de ejemplo"""
    config = {
        "data_sources": {
            "price_symbol": "EURUSD=X",
            "news_api_key": None,
            "twitter_api_key": None,
            "update_interval_minutes": 60,
            "use_mock_data": True
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
            "risk_per_trade": 0.02
        },
        "sentiment_settings": {
            "sentiment_weight": 0.3,
            "news_lookback_hours": 24,
            "min_impact_score": 0.5
        },
        "database": {
            "path": "trading_data.db",
            "backup_interval_hours": 6
        },
        "alerts": {
            "email_enabled": False,
            "console_alerts": True
        }
    }
    
    config_path = Path.cwd() / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Sample configuration created: {config_path}")

def main():
    """Función principal para ejecutar diagnósticos"""
    print("🤖 EUR/USD AI Trading System - Diagnostics")
    print("=" * 50)
    
    diagnostics = SystemDiagnostics()
    results = diagnostics.run_full_diagnostic()
    
    # Guardar reporte
    report_path = diagnostics.save_diagnostic_report()
    
    # Sugerencias basadas en resultados
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    
    if results['summary']['overall_status'] == 'ISSUES':
        print("1. Install missing dependencies with: pip install -r requirements.txt")
        print("2. Check internet connectivity for data sources")
        print("3. Ensure all files are in correct directories")
    elif results['summary']['overall_status'] == 'FUNCTIONAL':
        print("1. System should work with mock data")
        print("2. Consider configuring real data sources for production")
        print("3. Review warnings in diagnostic log")
    else:
        print("✅ System is ready to run!")
        print("Run: python src/main.py")
    
    return results

if __name__ == "__main__":
    main()