"""
Sistema Principal de Trading EUR/USD con IA - Versión Mejorada
Exponiendo predicciones para integración con dashboard
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
                "update_interval_minutes": 60,
                "use_real_data": True
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
            "dashboard": {
                "enable_integration": True,
                "shared_predictions": True,
                "update_interval_seconds": 60
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

class PredictionCache:
    """Cache de predicciones para compartir con dashboard"""
    
    def __init__(self):
        self.cache = {}
        self.last_update = datetime.now()
    
    def store_prediction(self, prediction_data: Dict):
        """Almacena una predicción en el cache"""
        self.cache['latest_prediction'] = prediction_data
        self.last_update = datetime.now()
        logger.info("Predicción almacenada en cache")
    
    def get_latest_prediction(self) -> Optional[Dict]:
        """Obtiene la última predicción del cache"""
        return self.cache.get('latest_prediction')
    
    def is_fresh(self, max_age_minutes: int = 5) -> bool:
        """Verifica si la predicción es reciente"""
        age = datetime.now() - self.last_update
        return age.total_seconds() < (max_age_minutes * 60)

class TradingSystemManager:
    """Gestor principal del sistema de trading - Versión mejorada"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config = TradingSystemConfig(config_file)
        self.prediction_cache = PredictionCache()
        
        # Estado del sistema
        self.is_running = False
        self.is_initialized = False
        self.model_trained = False
        self.last_prediction = None
        self.current_signal = "HOLD"
        
        # Componentes del sistema
        self.data_pipeline = None
        self.ensemble_model = None
        self.sentiment_analyzer = None
        
        # Intentar cargar componentes
        self._load_system_components()
    
    def _load_system_components(self):
        """Carga los componentes del sistema"""
        try:
            # Importar componentes si están disponibles
            try:
                sys.path.append('src/data_collection')
                from trading_architecture import DataPipeline
                self.data_pipeline = DataPipeline(self.config.get('data_sources.news_api_key'))
                logger.info("Data pipeline cargado")
            except ImportError:
                logger.warning("Data pipeline no disponible")
            
            try:
                sys.path.append('src/models')
                from ml_models import EnsembleModel
                self.ensemble_model = EnsembleModel()
                
                # Verificar si hay modelos entrenados
                if self._models_exist():
                    self.ensemble_model.load_models()
                    self.model_trained = True
                    logger.info("Modelos IA cargados")
                else:
                    logger.info("Modelos no encontrados - se entrenarán si es necesario")
                    
            except ImportError:
                logger.warning("Modelos IA no disponibles")
            
            try:
                sys.path.append('src/sentiment')
                from sentiment_backtesting import AdvancedSentimentAnalyzer
                self.sentiment_analyzer = AdvancedSentimentAnalyzer()
                logger.info("Analizador de sentimiento cargado")
            except ImportError:
                logger.warning("Analizador de sentimiento no disponible")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error cargando componentes: {e}")
            self.is_initialized = False
    
    def _models_exist(self) -> bool:
        """Verifica si los modelos existen"""
        model_files = [
            "models/lstm_model.h5",
            "models/gru_model.h5", 
            "models/rf_model.pkl",
            "models/feature_engineer.pkl"
        ]
        return all(os.path.exists(f) for f in model_files)
    
    async def get_market_data(self):
        """Obtiene datos de mercado actualizados"""
        try:
            if self.data_pipeline:
                # Usar data pipeline si está disponible
                price_data, news_data = await self.data_pipeline.run_data_collection()
                return price_data, news_data
            else:
                # Fallback a yfinance directo
                import yfinance as yf
                ticker = yf.Ticker("EURUSD=X")
                price_data = ticker.history(period="6mo", interval="1h")
                
                # Agregar indicadores técnicos básicos
                if not price_data.empty:
                    price_data['SMA_20'] = price_data['Close'].rolling(20).mean()
                    price_data['SMA_50'] = price_data['Close'].rolling(50).mean()
                    price_data['EMA_12'] = price_data['Close'].ewm(span=12).mean()
                    price_data['EMA_26'] = price_data['Close'].ewm(span=26).mean()
                    price_data['MACD'] = price_data['EMA_12'] - price_data['EMA_26']
                    price_data['MACD_Signal'] = price_data['MACD'].ewm(span=9).mean()
                    
                    # RSI
                    delta = price_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    price_data['RSI'] = 100 - (100 / (1 + rs))
                
                return price_data, pd.DataFrame()  # Sin noticias en fallback
                
        except Exception as e:
            logger.error(f"Error obteniendo datos de mercado: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    async def generate_prediction(self):
        """Genera predicción y la almacena en cache"""
        try:
            # Obtener datos de mercado
            price_data, news_data = await self.get_market_data()
            
            if price_data.empty:
                logger.warning("No hay datos de precios disponibles")
                return None
            
            # Calcular sentimiento
            sentiment_score = 0.0
            if not news_data.empty and self.sentiment_analyzer:
                try:
                    recent_news = news_data.tail(5)
                    sentiment_results = []
                    for _, news in recent_news.iterrows():
                        text = f"{news.get('title', '')} {news.get('description', '')}"
                        result = self.sentiment_analyzer.analyze_text(text)
                        score = result.confidence if result.sentiment == 'POSITIVE' else \
                               -result.confidence if result.sentiment == 'NEGATIVE' else 0
                        sentiment_results.append(score)
                    
                    sentiment_score = np.mean(sentiment_results) if sentiment_results else 0.0
                except Exception as e:
                    logger.warning(f"Error calculando sentimiento: {e}")
                    sentiment_score = 0.0
            
            # Generar predicción
            prediction = None
            
            if self.model_trained and self.ensemble_model and len(price_data) >= 60:
                try:
                    # Usar modelo IA
                    prediction = self.ensemble_model.predict(price_data.tail(100), sentiment_score)
                    logger.info("Predicción generada con modelo IA")
                except Exception as e:
                    logger.error(f"Error con modelo IA: {e}")
                    prediction = None
            
            if not prediction:
                # Fallback a análisis técnico
                prediction = self._generate_technical_prediction(price_data, sentiment_score)
                logger.info("Predicción generada con análisis técnico")
            
            # Preparar datos de predicción
            current_price = price_data['Close'].iloc[-1]
            
            prediction_data = {
                'prediction': prediction,
                'signal': self._determine_signal(prediction),
                'current_price': current_price,
                'sentiment_score': sentiment_score,
                'timestamp': datetime.now(),
                'data_points': len(price_data),
                'model_type': 'AI_ENSEMBLE' if self.model_trained else 'TECHNICAL_ANALYSIS'
            }
            
            # Almacenar en cache
            self.prediction_cache.store_prediction(prediction_data)
            
            # Actualizar estado
            self.last_prediction = prediction
            self.current_signal = prediction_data['signal']
            
            logger.info(f"Predicción: {prediction_data['signal']} - "
                       f"Confianza: {prediction.confidence:.1%} - "
                       f"Precio: {current_price:.5f} -> {prediction.price_prediction:.5f}")
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error generando predicción: {e}")
            return None
    
    def _generate_technical_prediction(self, price_data, sentiment_score=0.0):
        """Genera predicción usando análisis técnico"""
        try:
            current_price = price_data['Close'].iloc[-1]
            
            # Recopilar señales técnicas
            signals = []
            
            # SMA
            if 'SMA_20' in price_data.columns and 'SMA_50' in price_data.columns:
                sma_20 = price_data['SMA_20'].iloc[-1]
                sma_50 = price_data['SMA_50'].iloc[-1]
                signals.append(1 if sma_20 > sma_50 else -1)
            
            # RSI
            if 'RSI' in price_data.columns:
                rsi = price_data['RSI'].iloc[-1]
                if rsi < 30:
                    signals.append(1)  # Sobrevendido
                elif rsi > 70:
                    signals.append(-1)  # Sobrecomprado
                else:
                    signals.append(0)
            
            # MACD
            if 'MACD' in price_data.columns and 'MACD_Signal' in price_data.columns:
                macd = price_data['MACD'].iloc[-1]
                macd_signal = price_data['MACD_Signal'].iloc[-1]
                signals.append(1 if macd > macd_signal else -1)
            
            # Calcular predicción
            signal_sum = sum(signals) if signals else 0
            signal_strength = abs(signal_sum) / len(signals) if signals else 0
            
            if signal_sum > 0:
                direction = "UP"
                price_change = np.random.uniform(0.0005, 0.002)
            elif signal_sum < 0:
                direction = "DOWN"
                price_change = -np.random.uniform(0.0005, 0.002)
            else:
                direction = "STABLE"
                price_change = np.random.uniform(-0.0005, 0.0005)
            
            # Incluir influencia del sentimiento
            sentiment_influence = abs(sentiment_score) * 0.3
            if sentiment_score > 0.1 and direction == "UP":
                price_change *= 1.2
            elif sentiment_score < -0.1 and direction == "DOWN":
                price_change *= 1.2
            
            predicted_price = current_price + price_change
            confidence = min(0.9, max(0.5, signal_strength + sentiment_influence + 0.2))
            
            # Crear objeto de predicción compatible
            class TechnicalPrediction:
                def __init__(self, price_pred, direction_pred, conf, tech_inf, sent_inf):
                    self.price_prediction = price_pred
                    self.direction_prediction = direction_pred
                    self.confidence = conf
                    self.technical_influence = tech_inf
                    self.sentiment_influence = sent_inf
            
            return TechnicalPrediction(
                predicted_price,
                direction,
                confidence,
                confidence - sentiment_influence,
                sentiment_influence
            )
            
        except Exception as e:
            logger.error(f"Error en predicción técnica: {e}")
            # Predicción básica de emergencia
            class BasicPrediction:
                def __init__(self, price):
                    self.price_prediction = price
                    self.direction_prediction = "STABLE"
                    self.confidence = 0.5
                    self.technical_influence = 0.5
                    self.sentiment_influence = 0.0
            
            return BasicPrediction(current_price)
    
    def _determine_signal(self, prediction):
        """Determina la señal de trading basada en la predicción"""
        min_confidence = self.config.get('model_settings.min_confidence_threshold', 0.65)
        
        if prediction.confidence >= min_confidence:
            if prediction.direction_prediction == 'UP':
                return "BUY"
            elif prediction.direction_prediction == 'DOWN':
                return "SELL"
        
        return "HOLD"
    
    def get_cached_prediction(self) -> Optional[Dict]:
        """Obtiene la predicción del cache (para dashboard)"""
        if self.prediction_cache.is_fresh():
            return self.prediction_cache.get_latest_prediction()
        return None
    
    def get_system_status(self) -> Dict:
        """Obtiene el estado del sistema"""
        return {
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'model_trained': self.model_trained,
            'current_signal': self.current_signal,
            'last_prediction_time': self.prediction_cache.last_update,
            'has_fresh_prediction': self.prediction_cache.is_fresh(),
            'components': {
                'data_pipeline': self.data_pipeline is not None,
                'ensemble_model': self.ensemble_model is not None,
                'sentiment_analyzer': self.sentiment_analyzer is not None
            }
        }
    
    async def start_prediction_loop(self):
        """Inicia el loop de predicciones automáticas"""
        self.is_running = True
        update_interval = self.config.get('dashboard.update_interval_seconds', 60)
        
        logger.info(f"Iniciando loop de predicciones (intervalo: {update_interval}s)")
        
        while self.is_running:
            try:
                await self.generate_prediction()
                await asyncio.sleep(update_interval)
            except Exception as e:
                logger.error(f"Error en loop de predicciones: {e}")
                await asyncio.sleep(30)  # Pausa corta en caso de error
    
    def stop(self):
        """Detiene el sistema"""
        self.is_running = False
        logger.info("Sistema detenido")
    
    # Métodos adicionales para compatibilidad
    async def initialize_system(self):
        """Inicializa el sistema (compatibilidad)"""
        return self.is_initialized
    
    async def update_data(self):
        """Actualiza datos (compatibilidad)"""
        return await self.get_market_data()

# Función para ejecutar el sistema principal
async def run_trading_system():
    """Ejecuta el sistema principal de trading"""
    logger.info("Iniciando Sistema Principal de Trading EUR/USD")
    
    # Crear sistema
    trading_system = TradingSystemManager()
    
    if not trading_system.is_initialized:
        logger.error("Error inicializando el sistema")
        return
    
    try:
        # Generar predicción inicial
        await trading_system.generate_prediction()
        
        # Mostrar estado inicial
        status = trading_system.get_system_status()
        logger.info(f"Sistema iniciado - Señal: {status['current_signal']}")
        
        # Iniciar loop de predicciones
        await trading_system.start_prediction_loop()
        
    except KeyboardInterrupt:
        logger.info("Sistema detenido por el usuario")
        trading_system.stop()
    except Exception as e:
        logger.error(f"Error en sistema principal: {e}")
        trading_system.stop()

# Script principal
async def main():
    """Función principal"""
    print("EUR/USD AI Trading System - Sistema Principal")
    print("=" * 60)
    
    await run_trading_system()

if __name__ == "__main__":
    asyncio.run(main())