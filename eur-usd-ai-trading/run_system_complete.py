"""
Script Principal para Ejecutar Sistema Completo
Sistema Principal + Dashboard en Tiempo Real
"""

import asyncio
import threading
import subprocess
import sys
import time
import logging
from pathlib import Path
import signal
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystemLauncher:
    """Lanzador del sistema completo de trading"""
    
    def __init__(self):
        self.main_system_process = None
        self.dashboard_process = None
        self.is_running = False
    
    def check_dependencies(self):
        """Verifica dependencias del sistema"""
        logger.info("🔍 Verificando dependencias...")
        
        required_modules = [
            'streamlit', 'pandas', 'numpy', 'plotly', 
            'yfinance', 'tensorflow', 'sklearn'
        ]
        
        missing = []
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"✅ {module}")
            except ImportError:
                missing.append(module)
                logger.error(f"❌ {module} - FALTANTE")
        
        if missing:
            logger.error(f"Instalar dependencias faltantes: pip install {' '.join(missing)}")
            return False
        
        logger.info("✅ Todas las dependencias están disponibles")
        return True
    
    def start_main_system(self):
        """Inicia el sistema principal en background"""
        logger.info("🚀 Iniciando sistema principal...")
        
        try:
            # Importar y ejecutar sistema principal
            from src.main import TradingSystemManager
            
            async def run_main_system():
                system = TradingSystemManager()
                if system.is_initialized:
                    logger.info("✅ Sistema principal inicializado")
                    await system.start_prediction_loop()
                else:
                    logger.error("❌ Error inicializando sistema principal")
            
            # Ejecutar en thread separado
            def main_system_thread():
                asyncio.run(run_main_system())
            
            self.main_system_thread = threading.Thread(
                target=main_system_thread,
                daemon=True
            )
            self.main_system_thread.start()
            
            logger.info("✅ Sistema principal ejecutándose en background")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error iniciando sistema principal: {e}")
            return False
    
    def start_dashboard(self):
        """Inicia el dashboard de Streamlit"""
        logger.info("🖥️ Iniciando dashboard...")
        
        try:
            # Ruta al dashboard
            dashboard_script = Path(__file__).parent / "src" / "dashboard" / "realtime_app.py"
            
            if not dashboard_script.exists():
                # Crear el script del dashboard si no existe
                self.create_dashboard_script(dashboard_script)
            
            # Ejecutar Streamlit
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(dashboard_script),
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ]
            
            self.dashboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("✅ Dashboard iniciado en http://localhost:8501")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error iniciando dashboard: {e}")
            return False
    
    def create_dashboard_script(self, script_path):
        """Crea el script del dashboard si no existe"""
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        dashboard_code = '''
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
'''
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_code)
        
        logger.info(f"✅ Dashboard script creado: {script_path}")
    
    def stop_system(self):
        """Detiene todo el sistema"""
        logger.info("🛑 Deteniendo sistema...")
        
        self.is_running = False
        
        # Detener dashboard
        if self.dashboard_process:
            self.dashboard_process.terminate()
            logger.info("✅ Dashboard detenido")
        
        # El sistema principal se detendrá automáticamente
        logger.info("✅ Sistema detenido completamente")
    
    def monitor_system(self):
        """Monitorea el estado del sistema"""
        while self.is_running:
            try:
                # Verificar dashboard
                if self.dashboard_process and self.dashboard_process.poll() is not None:
                    logger.warning("⚠️ Dashboard detenido inesperadamente")
                    self.start_dashboard()
                
                time.sleep(5)  # Verificar cada 5 segundos
                
            except KeyboardInterrupt:
                self.stop_system()
                break
            except Exception as e:
                logger.error(f"Error en monitoreo: {e}")
    
    def run_complete_system(self):
        """Ejecuta el sistema completo"""
        print("🤖 EUR/USD AI Trading System - Launcher")
        print("=" * 60)
        
        # Verificar dependencias
        if not self.check_dependencies():
            print("❌ Dependencias faltantes. Instalar e intentar nuevamente.")
            return False
        
        # Configurar manejo de señales
        signal.signal(signal.SIGINT, lambda s, f: self.stop_system())
        signal.signal(signal.SIGTERM, lambda s, f: self.stop_system())
        
        # Iniciar componentes
        print("\n🚀 Iniciando componentes del sistema...")
        
        # 1. Sistema principal
        if not self.start_main_system():
            print("❌ Error iniciando sistema principal")
            return False
        
        # Esperar un momento para que se inicialice
        time.sleep(3)
        
        # 2. Dashboard
        if not self.start_dashboard():
            print("❌ Error iniciando dashboard")
            return False
        
        self.is_running = True
        
        print("\n✅ Sistema completo iniciado!")
        print("=" * 60)
        print("🤖 Sistema Principal: Ejecutándose en background")
        print("🖥️ Dashboard: http://localhost:8501")
        print("📊 Predicciones IA: Actualizándose automáticamente")
        print("=" * 60)
        print("\n💡 Presiona Ctrl+C para detener el sistema")
        print("📱 Abre http://localhost:8501 en tu navegador")
        
        # Monitorear sistema
        try:
            self.monitor_system()
        except KeyboardInterrupt:
            print("\n👋 Deteniendo sistema...")
            self.stop_system()
        
        print("✅ Sistema detenido correctamente")
        return True

def main():
    """Función principal"""
    launcher = TradingSystemLauncher()
    launcher.run_complete_system()

if __name__ == "__main__":
    main()
