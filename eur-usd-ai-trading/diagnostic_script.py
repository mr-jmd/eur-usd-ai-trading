"""
Script de Diagnóstico para Sistema de Trading en Tiempo Real
Verifica todas las conexiones y componentes
"""

import sys
import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
import importlib.util

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemDiagnostic:
    """Diagnóstico completo del sistema"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {}
    
    def check_file_structure(self):
        """Verifica estructura de archivos"""
        logger.info("🗂️ Verificando estructura de archivos...")
        
        required_files = [
            'src/main.py',
            'src/models/ml_models.py',
            'src/data_collection/trading_architecture.py',
            'config.json',
            'requirements.txt'
        ]
        
        optional_files = [
            'realtime_dashboard.py',
            'run_system_complete.py',
            '.env'
        ]
        
        missing_required = []
        missing_optional = []
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_required.append(file_path)
                logger.error(f"❌ {file_path} - FALTANTE")
            else:
                logger.info(f"✅ {file_path}")
        
        for file_path in optional_files:
            if not (self.project_root / file_path).exists():
                missing_optional.append(file_path)
                logger.warning(f"⚠️ {file_path} - OPCIONAL")
            else:
                logger.info(f"✅ {file_path}")
        
        self.results['file_structure'] = {
            'missing_required': missing_required,
            'missing_optional': missing_optional,
            'status': 'OK' if not missing_required else 'ERROR'
        }
        
        return len(missing_required) == 0
    
    def check_dependencies(self):
        """Verifica dependencias Python"""
        logger.info("📦 Verificando dependencias...")
        
        required_modules = [
            'streamlit', 'pandas', 'numpy', 'plotly', 'yfinance',
            'tensorflow', 'sklearn', 'asyncio', 'threading'
        ]
        
        missing = []
        installed = []
        
        for module in required_modules:
            try:
                spec = importlib.util.find_spec(module)
                if spec is not None:
                    installed.append(module)
                    logger.info(f"✅ {module}")
                else:
                    missing.append(module)
                    logger.error(f"❌ {module}")
            except ImportError:
                missing.append(module)
                logger.error(f"❌ {module}")
        
        self.results['dependencies'] = {
            'installed': installed,
            'missing': missing,
            'status': 'OK' if not missing else 'ERROR'
        }
        
        return len(missing) == 0
    
    def check_system_imports(self):
        """Verifica importaciones del sistema"""
        logger.info("🔧 Verificando importaciones del sistema...")
        
        try:
            sys.path.insert(0, str(self.project_root / 'src'))
            
            # Intentar importar componentes principales
            components = {}
            
            try:
                from main import TradingSystemManager
                components['main_system'] = True
                logger.info("✅ Sistema principal importable")
            except Exception as e:
                components['main_system'] = False
                logger.error(f"❌ Sistema principal: {e}")
            
            try:
                from models.ml_models import EnsembleModel
                components['ml_models'] = True
                logger.info("✅ Modelos ML importables")
            except Exception as e:
                components['ml_models'] = False
                logger.error(f"❌ Modelos ML: {e}")
            
            try:
                from data_collection.trading_architecture import DataPipeline
                components['data_pipeline'] = True
                logger.info("✅ Pipeline de datos importable")
            except Exception as e:
                components['data_pipeline'] = False
                logger.error(f"❌ Pipeline de datos: {e}")
            
            self.results['system_imports'] = {
                'components': components,
                'status': 'OK' if all(components.values()) else 'WARNING'
            }
            
            return all(components.values())
            
        except Exception as e:
            logger.error(f"❌ Error general en importaciones: {e}")
            self.results['system_imports'] = {
                'error': str(e),
                'status': 'ERROR'
            }
            return False
    
    async def test_system_initialization(self):
        """Prueba inicialización del sistema"""
        logger.info("🚀 Probando inicialización del sistema...")
        
        try:
            sys.path.insert(0, str(self.project_root / 'src'))
            from main import TradingSystemManager
            
            # Crear instancia del sistema
            system = TradingSystemManager()
            
            # Verificar inicialización
            if system.is_initialized:
                logger.info("✅ Sistema inicializado correctamente")
                
                # Probar generación de predicción
                try:
                    prediction_data = await system.generate_prediction()
                    if prediction_data:
                        logger.info("✅ Predicción generada exitosamente")
                        logger.info(f"📊 Señal: {prediction_data['signal']}")
                        logger.info(f"🎯 Confianza: {prediction_data['prediction'].confidence:.1%}")
                        
                        self.results['system_test'] = {
                            'initialized': True,
                            'prediction_generated': True,
                            'signal': prediction_data['signal'],
                            'confidence': prediction_data['prediction'].confidence,
                            'status': 'OK'
                        }
                        return True
                    else:
                        logger.warning("⚠️ No se pudo generar predicción")
                        self.results['system_test'] = {
                            'initialized': True,
                            'prediction_generated': False,
                            'status': 'WARNING'
                        }
                        return False
                
                except Exception as e:
                    logger.error(f"❌ Error generando predicción: {e}")
                    self.results['system_test'] = {
                        'initialized': True,
                        'prediction_error': str(e),
                        'status': 'WARNING'
                    }
                    return False
            else:
                logger.error("❌ Sistema no se pudo inicializar")
                self.results['system_test'] = {
                    'initialized': False,
                    'status': 'ERROR'
                }
                return False
                
        except Exception as e:
            logger.error(f"❌ Error en prueba del sistema: {e}")
            self.results['system_test'] = {
                'error': str(e),
                'status': 'ERROR'
            }
            return False
    
    def check_streamlit_installation(self):
        """Verifica instalación de Streamlit"""
        logger.info("🖥️ Verificando Streamlit...")
        
        try:
            import streamlit as st
            streamlit_version = st.__version__
            logger.info(f"✅ Streamlit {streamlit_version} disponible")
            
            # Verificar si se puede ejecutar comando streamlit
            import subprocess
            result = subprocess.run([sys.executable, '-m', 'streamlit', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info("✅ Comando streamlit funcional")
                self.results['streamlit'] = {
                    'installed': True,
                    'version': streamlit_version,
                    'command_works': True,
                    'status': 'OK'
                }
                return True
            else:
                logger.warning("⚠️ Comando streamlit no funciona")
                self.results['streamlit'] = {
                    'installed': True,
                    'version': streamlit_version,
                    'command_works': False,
                    'status': 'WARNING'
                }
                return False
                
        except Exception as e:
            logger.error(f"❌ Error con Streamlit: {e}")
            self.results['streamlit'] = {
                'error': str(e),
                'status': 'ERROR'
            }
            return False
    
    def check_configuration(self):
        """Verifica configuración del sistema"""
        logger.info("⚙️ Verificando configuración...")
        
        config_file = self.project_root / 'config.json'
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Verificar secciones importantes
                required_sections = [
                    'data_sources', 'model_settings', 'trading_settings',
                    'dashboard', 'database'
                ]
                
                missing_sections = [s for s in required_sections if s not in config]
                
                if not missing_sections:
                    logger.info("✅ Configuración completa")
                    
                    # Verificar configuración del dashboard
                    dashboard_config = config.get('dashboard', {})
                    if dashboard_config.get('enable_realtime', False):
                        logger.info("✅ Dashboard en tiempo real habilitado")
                    else:
                        logger.warning("⚠️ Dashboard en tiempo real deshabilitado")
                    
                    self.results['configuration'] = {
                        'exists': True,
                        'complete': True,
                        'missing_sections': [],
                        'realtime_enabled': dashboard_config.get('enable_realtime', False),
                        'status': 'OK'
                    }
                    return True
                else:
                    logger.warning(f"⚠️ Secciones faltantes: {missing_sections}")
                    self.results['configuration'] = {
                        'exists': True,
                        'complete': False,
                        'missing_sections': missing_sections,
                        'status': 'WARNING'
                    }
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error leyendo configuración: {e}")
                self.results['configuration'] = {
                    'exists': True,
                    'error': str(e),
                    'status': 'ERROR'
                }
                return False
        else:
            logger.warning("⚠️ Archivo config.json no encontrado")
            self.results['configuration'] = {
                'exists': False,
                'status': 'WARNING'
            }
            return False
    
    async def run_full_diagnostic(self):
        """Ejecuta diagnóstico completo"""
        logger.info("🔍 Iniciando diagnóstico completo del sistema...")
        logger.info("=" * 60)
        
        checks = [
            ("Estructura de Archivos", self.check_file_structure),
            ("Dependencias Python", self.check_dependencies),
            ("Importaciones del Sistema", self.check_system_imports),
            ("Instalación Streamlit", self.check_streamlit_installation),
            ("Configuración", self.check_configuration),
            ("Inicialización del Sistema", self.test_system_initialization)
        ]
        
        passed = 0
        total = len(checks)
        
        for name, check_func in checks:
            logger.info(f"\n🔎 {name}...")
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                if result:
                    passed += 1
                    
            except Exception as e:
                logger.error(f"❌ Error en {name}: {e}")
        
        # Resumen
        logger.info("=" * 60)
        logger.info("📊 RESUMEN DEL DIAGNÓSTICO")
        logger.info("=" * 60)
        
        success_rate = (passed / total) * 100
        
        if success_rate >= 90:
            logger.info(f"✅ SISTEMA LISTO ({success_rate:.0f}% - {passed}/{total})")
            overall_status = 'READY'
        elif success_rate >= 70:
            logger.info(f"⚠️ SISTEMA FUNCIONAL CON ADVERTENCIAS ({success_rate:.0f}% - {passed}/{total})")
            overall_status = 'FUNCTIONAL'
        else:
            logger.info(f"❌ SISTEMA REQUIERE ATENCIÓN ({success_rate:.0f}% - {passed}/{total})")
            overall_status = 'NEEDS_ATTENTION'
        
        # Recomendaciones
        logger.info("\n💡 RECOMENDACIONES:")
        logger.info("-" * 30)
        
        if overall_status == 'READY':
            logger.info("🚀 Sistema listo para ejecutar:")
            logger.info("   1. python run_system_complete.py")
            logger.info("   2. Abrir http://localhost:8501")
        elif overall_status == 'FUNCTIONAL':
            logger.info("⚙️ Revisar advertencias e intentar:")
            logger.info("   1. Instalar dependencias faltantes")
            logger.info("   2. Verificar configuración")
            logger.info("   3. Ejecutar sistema en modo básico")
        else:
            logger.info("🔧 Acciones requeridas:")
            logger.info("   1. Instalar dependencias: pip install -r requirements.txt")
            logger.info("   2. Verificar estructura de archivos")
            logger.info("   3. Revisar configuración")
        
        self.results['summary'] = {
            'passed_checks': passed,
            'total_checks': total,
            'success_rate': success_rate,
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results
    
    def save_diagnostic_report(self):
        """Guarda reporte de diagnóstico"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / f"diagnostic_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"📄 Reporte guardado: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"❌ Error guardando reporte: {e}")
            return None

async def main():
    """Función principal de diagnóstico"""
    diagnostic = SystemDiagnostic()
    
    # Ejecutar diagnóstico
    results = await diagnostic.run_full_diagnostic()
    
    # Guardar reporte
    diagnostic.save_diagnostic_report()
    
    return results

if __name__ == "__main__":
    print("🔍 EUR/USD AI Trading System - Diagnóstico")
    print("=" * 60)
    
    results = asyncio.run(main())
    
    print(f"\n🏁 Diagnóstico completado")
    print(f"📊 Estado general: {results['summary']['overall_status']}")
    print(f"✅ Checks pasados: {results['summary']['passed_checks']}/{results['summary']['total_checks']}")
