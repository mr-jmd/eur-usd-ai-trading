"""
Sistema Principal de Trading EUR/USD con IA
Integración completa de todos los componentes
"""

import asyncio
import logging
import os
import sys
import json
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingSystemConfig:
    """Configuración del sistema de trading"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Carga la configuración desde archivo"""
        default_config = {
            "data_sources": {
                "price_symbol": "EURUSD=X",
                "news_api_key": None,
                "twitter_api_key": None,
                "update_interval_minutes": 60
            },
            "model_settings": {
                "ensemble_weights": {
                    "lstm": 0.4,
                    "gru": 0.4,
                    "rf": 0.2
                },
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
                "email_smtp": "smtp.gmail.com",
                "email_port": 587,
                "email_user": None,
                "email_password": None,
                "alert_recipients": []
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                self.config = {**default_config, **user_config}
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Guarda la configuración actual"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        """Obtiene un valor de configuración"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value

class AlertSystem:
    """Sistema de alertas y notificaciones"""
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.alerts_sent = []
    
    def send_alert(self, level: str, title: str, message: str):
        """Envía una alerta"""
        alert = {
            'timestamp': datetime.now(),
            'level': level,
            'title': title,
            'message': message
        }
        
        logger.info(f"ALERT [{level}]: {title} - {message}")
        self.alerts_sent.append(alert)
        
        # Enviar por email si está configurado
        if self.config.get('alerts.email_enabled', False):
            self._send_email_alert(alert)
        
        # Mantener solo las últimas 100 alertas
        if len(self.alerts_sent) > 100:
            self.alerts_sent = self.alerts_sent[-100:]
    
    def _send_email_alert(self, alert: Dict):
        """Envía alerta por email"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            smtp_server = self.config.get('alerts.email_smtp')
            smtp_port = self.config.get('alerts.email_port')
            email_user = self.config.get('alerts.email_user')
            email_password = self.config.get('alerts.email_password')
            recipients = self.config.get('alerts.alert_recipients', [])
            
            if not all([smtp_server, email_user, email_password, recipients]):
                return
            
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Trading Alert [{alert['level']}]: {alert['title']}"
            
            body = f"""
            Alerta del Sistema de Trading EUR/USD
            
            Nivel: {alert['level']}
            Título: {alert['title']}
            Mensaje: {alert['message']}
            Tiempo: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            server.sendmail(email_user, recipients, msg.as_string())
            server.quit()
            
            logger.info("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")

class TradingSystemManager:
    """Gestor principal del sistema de trading"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config = TradingSystemConfig(config_file)
        self.alert_system = AlertSystem(self.config)
        
        # Importar componentes del sistema
        try:
            # Estos imports asumen que los otros archivos están en el mismo directorio
            from data_collection.trading_architecture import DataPipeline, DatabaseManager
            from models.ml_models import EnsembleModel
            from sentiment.sentiment_backtesting import AdvancedSentimentAnalyzer, BacktestEngine, AITradingStrategy
            
            self.data_pipeline = DataPipeline(self.config.get('data_sources.news_api_key'))
            self.db_manager = DatabaseManager(self.config.get('database.path'))
            self.sentiment_analyzer = AdvancedSentimentAnalyzer()
            self.ensemble_model = EnsembleModel()
            self.backtest_engine = BacktestEngine()
            
            self.is_initialized = True
            logger.info("Sistema inicializado correctamente")
            
        except ImportError as e:
            logger.error(f"Error importing system components: {e}")
            self.is_initialized = False
        
        self.is_running = False
        self.last_prediction = None
        self.current_signal = "HOLD"
        self.model_trained = False
        
    async def initialize_system(self):
        """Inicializa completamente el sistema"""
        if not self.is_initialized:
            logger.error("System components not properly imported")
            return False
        
        try:
            # Verificar y crear directorios necesarios
            Path("models").mkdir(exist_ok=True)
            Path("data").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            
            # Cargar o entrenar modelos
            if self._models_exist():
                logger.info("Loading existing models...")
                self.ensemble_model.load_models()
                self.model_trained = True
            else:
                logger.info("Training models for the first time...")
                await self._initial_model_training()
            
            # Recolección inicial de datos
            logger.info("Initial data collection...")
            await self.data_pipeline.run_data_collection()
            
            self.alert_system.send_alert("INFO", "System Initialized", "Trading system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            self.alert_system.send_alert("ERROR", "Initialization Failed", str(e))
            return False
    
    def _models_exist(self) -> bool:
        """Verifica si los modelos existen"""
        model_files = [
            "models/lstm_model.h5",
            "models/gru_model.h5",
            "models/rf_model.pkl",
            "models/feature_engineer.pkl"
        ]
        return all(os.path.exists(f) for f in model_files)
    
    async def _initial_model_training(self):
        """Entrenamiento inicial de modelos"""
        try:
            # Obtener datos históricos para entrenamiento
            price_data = await self.data_pipeline.price_collector.collect_data(period="2y", interval="1h")
            
            if price_data.empty:
                raise ValueError("No historical data available for training")
            
            # Entrenar ensemble
            logger.info("Training ensemble models...")
            self.ensemble_model.train(price_data)
            
            # Guardar modelos
            self.ensemble_model.save_models()
            self.model_trained = True
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise
    
    async def update_data(self):
        """Actualiza los datos del sistema"""
        try:
            logger.info("Updating market data...")
            
            # Recolectar nuevos datos
            price_data, news_data = await self.data_pipeline.run_data_collection()
            
            # Analizar sentimiento de noticias recientes
            if not news_data.empty:
                recent_news = news_data.tail(10)
                sentiment_results = []
                
                for _, news in recent_news.iterrows():
                    text = f"{news['title']} {news['description']}"
                    result = self.sentiment_analyzer.analyze_text(text)
                    sentiment_results.append({
                        'timestamp': news['timestamp'],
                        'sentiment_score': result.confidence if result.sentiment == 'POSITIVE' 
                                          else -result.confidence if result.sentiment == 'NEGATIVE' else 0,
                        'impact_score': result.impact_score
                    })
                
                # Calcular sentimiento promedio ponderado por impacto
                if sentiment_results:
                    total_impact = sum(r['impact_score'] for r in sentiment_results)
                    if total_impact > 0:
                        weighted_sentiment = sum(
                            r['sentiment_score'] * r['impact_score'] for r in sentiment_results
                        ) / total_impact
                    else:
                        weighted_sentiment = 0
                else:
                    weighted_sentiment = 0
            else:
                weighted_sentiment = 0
            
            return price_data, weighted_sentiment
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            self.alert_system.send_alert("ERROR", "Data Update Failed", str(e))
            return pd.DataFrame(), 0
    
    async def generate_prediction(self):
        """Genera predicción y señal de trading"""
        if not self.model_trained:
            logger.warning("Models not trained, cannot generate prediction")
            return None
        
        try:
            # Obtener datos recientes
            price_data, sentiment_score = await self.update_data()
            
            if price_data.empty or len(price_data) < 60:
                logger.warning("Insufficient data for prediction")
                return None
            
            # Generar predicción
            prediction = self.ensemble_model.predict(price_data.tail(100), sentiment_score)
            
            # Determinar señal de trading
            min_confidence = self.config.get('model_settings.min_confidence_threshold', 0.65)
            
            if prediction.confidence >= min_confidence:
                if prediction.direction_prediction == 'UP':
                    signal = "BUY"
                elif prediction.direction_prediction == 'DOWN':
                    signal = "SELL"
                else:
                    signal = "HOLD"
            else:
                signal = "HOLD"
            
            # Actualizar estado
            self.last_prediction = prediction
            self.current_signal = signal
            
            # Log de la predicción
            current_price = price_data['Close'].iloc[-1] if 'Close' in price_data.columns else price_data['close'].iloc[-1]
            logger.info(f"Prediction - Price: {current_price:.5f} -> {prediction.price_prediction:.5f}, "
                       f"Direction: {prediction.direction_prediction}, Signal: {signal}, "
                       f"Confidence: {prediction.confidence:.2f}")
            
            # Alertas para señales de alta confianza
            if prediction.confidence >= 0.8 and signal != "HOLD":
                self.alert_system.send_alert(
                    "HIGH", 
                    f"Strong {signal} Signal",
                    f"Confidence: {prediction.confidence:.1%}, Direction: {prediction.direction_prediction}"
                )
            
            return {
                'prediction': prediction,
                'signal': signal,
                'current_price': current_price,
                'sentiment_score': sentiment_score,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            self.alert_system.send_alert("ERROR", "Prediction Failed", str(e))
            return None
    
    async def run_backtest(self, start_date: str = None, end_date: str = None):
        """Ejecuta backtesting del sistema"""
        try:
            logger.info("Running backtest...")
            
            # Obtener datos históricos
            if start_date:
                period = None
            else:
                period = "1y"
            
            price_data = await self.data_pipeline.price_collector.collect_data(period=period, interval="1h")
            
            # Simular datos de sentimiento (en producción vendría de la base de datos)
            sentiment_data = pd.DataFrame({
                'timestamp': price_data.index[::4],  # Cada 4 horas
                'sentiment_score': np.random.randn(len(price_data) // 4) * 0.3
            })
            
            # Crear estrategia AI
            strategy = AITradingStrategy(
                self.ensemble_model,
                sentiment_weight=self.config.get('sentiment_settings.sentiment_weight', 0.3)
            )
            
            # Ejecutar backtest
            results = self.backtest_engine.run_backtest(
                price_data=price_data,
                sentiment_data=sentiment_data,
                strategy=strategy
            )
            
            logger.info(f"Backtest completed - Return: {results.total_return:.2f}%, "
                       f"Win Rate: {results.win_rate:.1f}%, Trades: {results.total_trades}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    def start_scheduler(self):
        """Inicia el programador de tareas"""
        update_interval = self.config.get('data_sources.update_interval_minutes', 60)
        retrain_interval = self.config.get('model_settings.retrain_interval_days', 7)
        backup_interval = self.config.get('database.backup_interval_hours', 6)
        
        # Programar tareas
        schedule.every(update_interval).minutes.do(self._scheduled_update)
        schedule.every(retrain_interval).days.do(self._scheduled_retrain)
        schedule.every(backup_interval).hours.do(self._scheduled_backup)
        
        # Ejecutar programador en hilo separado
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Verificar cada minuto
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Scheduler started")
    
    def _scheduled_update(self):
        """Tarea programada de actualización"""
        if not self.is_running:
            return
        
        try:
            # Ejecutar actualización en loop de eventos
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.generate_prediction())
            loop.close()
        except Exception as e:
            logger.error(f"Error in scheduled update: {e}")
    
    def _scheduled_retrain(self):
        """Tarea programada de reentrenamiento"""
        if not self.is_running:
            return
        
        try:
            logger.info("Starting scheduled model retraining...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._initial_model_training())
            loop.close()
            
            self.alert_system.send_alert("INFO", "Model Retrained", "Scheduled model retraining completed")
        except Exception as e:
            logger.error(f"Error in scheduled retraining: {e}")
            self.alert_system.send_alert("ERROR", "Retraining Failed", str(e))
    
    def _scheduled_backup(self):
        """Tarea programada de backup"""
        try:
            backup_path = f"backups/trading_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            os.makedirs("backups", exist_ok=True)
            
            # Copiar base de datos
            import shutil
            shutil.copy2(self.config.get('database.path'), backup_path)
            
            logger.info(f"Database backup created: {backup_path}")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    async def start(self):
        """Inicia el sistema de trading"""
        if not await self.initialize_system():
            logger.error("Failed to initialize system")
            return False
        
        self.is_running = True
        self.start_scheduler()
        
        logger.info("Trading system started successfully")
        
        # Generar predicción inicial
        await self.generate_prediction()
        
        return True
    
    def stop(self):
        """Detiene el sistema de trading"""
        self.is_running = False
        logger.info("Trading system stopped")
        self.alert_system.send_alert("INFO", "System Stopped", "Trading system has been stopped")
    
    def get_status(self) -> Dict:
        """Obtiene el estado actual del sistema"""
        return {
            'is_running': self.is_running,
            'model_trained': self.model_trained,
            'current_signal': self.current_signal,
            'last_prediction': self.last_prediction.__dict__ if self.last_prediction else None,
            'last_update': datetime.now(),
            'config': self.config.config
        }

# Script principal
async def main():
    """Función principal del sistema"""
    print("🤖 EUR/USD AI Trading System")
    print("=" * 50)
    
    # Crear instancia del sistema
    trading_system = TradingSystemManager()
    
    try:
        # Iniciar sistema
        if await trading_system.start():
            print("✅ Sistema iniciado correctamente")
            
            # Mantener el sistema corriendo
            print("Sistema ejecutándose... Presiona Ctrl+C para detener")
            
            while True:
                await asyncio.sleep(60)  # Dormir 1 minuto
                
                # Mostrar estado cada hora
                status = trading_system.get_status()
                if datetime.now().minute == 0:  # Cada hora en punto
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Signal: {status['current_signal']}, "
                          f"Running: {status['is_running']}")
        
        else:
            print("❌ Error al iniciar el sistema")
            
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo sistema...")
        trading_system.stop()
        print("✅ Sistema detenido")
    
    except Exception as e:
        print(f"❌ Error en el sistema: {e}")
        trading_system.stop()

if __name__ == "__main__":
    asyncio.run(main())
