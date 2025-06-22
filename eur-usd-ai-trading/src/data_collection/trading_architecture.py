"""
Sistema de Trading EUR/USD con IA - Arquitectura Principal
Basado en el proyecto del Instituto Tecnológico Metropolitano
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import sqlite3
from datetime import datetime, timedelta
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Estructura para señales de trading"""
    timestamp: datetime
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    price_prediction: float
    sentiment_score: float
    technical_indicators: Dict
    news_impact: float

@dataclass
class MarketData:
    """Estructura para datos de mercado"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
class DataCollector(ABC):
    """Clase abstracta para recolección de datos"""
    
    @abstractmethod
    async def collect_data(self) -> pd.DataFrame:
        pass

class PriceDataCollector(DataCollector):
    """Recolector de datos de precios EUR/USD"""
    
    def __init__(self, symbol: str = "EURUSD=X"):
        self.symbol = symbol
        
    async def collect_data(self, period: str = "1y", interval: str = "1h") -> pd.DataFrame:
        """Recolecta datos históricos de precios"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Agregar indicadores técnicos básicos
            data = self._add_technical_indicators(data)
            return data
            
        except Exception as e:
            logger.error(f"Error recolectando datos de precios: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega indicadores técnicos al DataFrame"""
        # SMA (Simple Moving Average)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # EMA (Exponential Moving Average)
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
        
        return df

class NewsDataCollector(DataCollector):
    """Recolector de noticias financieras"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        
    async def collect_data(self, query: str = "EUR USD forex", days_back: int = 7) -> pd.DataFrame:
        """Recolecta noticias financieras"""
        try:
            if not self.api_key:
                logger.warning("No API key provided for news collection")
                return self._get_mock_news_data()
            
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
            articles = data.get('articles', [])
            
            news_df = pd.DataFrame([
                {
                    'timestamp': pd.to_datetime(article['publishedAt']),
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'source': article['source']['name']
                }
                for article in articles if article['title'] and article['description']
            ])
            
            return news_df
            
        except Exception as e:
            logger.error(f"Error recolectando noticias: {e}")
            return self._get_mock_news_data()
    
    def _get_mock_news_data(self) -> pd.DataFrame:
        """Datos mock para desarrollo sin API key"""
        mock_data = [
            {
                'timestamp': datetime.now() - timedelta(hours=i),
                'title': f'EUR/USD Market Update {i}',
                'description': 'European Central Bank announces new monetary policy.',
                'content': 'The ECB has decided to maintain current interest rates...',
                'source': 'Financial Times'
            }
            for i in range(10)
        ]
        return pd.DataFrame(mock_data)

class SentimentAnalyzer:
    """Analizador de sentimientos para noticias financieras"""
    
    def __init__(self):
        self.financial_terms = {
            'bullish': 1.0, 'bearish': -1.0, 'positive': 0.8, 'negative': -0.8,
            'strong': 0.6, 'weak': -0.6, 'rise': 0.5, 'fall': -0.5,
            'increase': 0.4, 'decrease': -0.4, 'up': 0.3, 'down': -0.3
        }
    
    def analyze_sentiment(self, text: str) -> float:
        """Analiza el sentimiento de un texto"""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        sentiment_score = 0.0
        word_count = 0
        
        words = text_lower.split()
        
        for word in words:
            if word in self.financial_terms:
                sentiment_score += self.financial_terms[word]
                word_count += 1
        
        # Normalizar el score
        if word_count > 0:
            sentiment_score = sentiment_score / word_count
        
        # Limitar entre -1 y 1
        return max(-1.0, min(1.0, sentiment_score))
    
    def batch_analyze(self, texts: List[str]) -> List[float]:
        """Analiza múltiples textos"""
        return [self.analyze_sentiment(text) for text in texts]

class DatabaseManager:
    """Gestor de base de datos"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializa las tablas de la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de precios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                sma_20 REAL,
                sma_50 REAL,
                rsi REAL,
                macd REAL
            )
        ''')
        
        # Tabla de noticias
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                title TEXT,
                description TEXT,
                content TEXT,
                source TEXT,
                sentiment_score REAL
            )
        ''')
        
        # Tabla de señales
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                signal TEXT,
                confidence REAL,
                price_prediction REAL,
                actual_price REAL,
                profit_loss REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_price_data(self, df: pd.DataFrame):
        """Guarda datos de precios en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        df.to_sql('price_data', conn, if_exists='append', index_label='timestamp')
        conn.close()
    
    def save_news_data(self, df: pd.DataFrame):
        """Guarda datos de noticias en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        df.to_sql('news_data', conn, if_exists='append', index=False)
        conn.close()
    
    def get_latest_data(self, table: str, limit: int = 1000) -> pd.DataFrame:
        """Obtiene los datos más recientes de una tabla"""
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

class DataPipeline:
    """Pipeline principal de datos"""
    
    def __init__(self, news_api_key: Optional[str] = None):
        self.price_collector = PriceDataCollector()
        self.news_collector = NewsDataCollector(news_api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.db_manager = DatabaseManager()
    
    async def run_data_collection(self):
        """Ejecuta la recolección completa de datos"""
        logger.info("Iniciando recolección de datos...")
        
        # Recolectar datos de precios
        price_data = await self.price_collector.collect_data()
        if not price_data.empty:
            logger.info(f"Recolectados {len(price_data)} registros de precios")
            self.db_manager.save_price_data(price_data)
        
        # Recolectar noticias
        news_data = await self.news_collector.collect_data()
        if not news_data.empty:
            # Analizar sentimientos
            combined_text = news_data['title'] + ' ' + news_data['description']
            news_data['sentiment_score'] = self.sentiment_analyzer.batch_analyze(combined_text.tolist())
            
            logger.info(f"Recolectadas {len(news_data)} noticias con análisis de sentimiento")
            self.db_manager.save_news_data(news_data)
        
        logger.info("Recolección de datos completada")
        return price_data, news_data

# Ejemplo de uso
async def main():
    """Función principal de ejemplo"""
    pipeline = DataPipeline()
    
    # Ejecutar recolección de datos
    price_data, news_data = await pipeline.run_data_collection()
    
    print("Datos de precios:")
    print(price_data.tail() if not price_data.empty else "No hay datos de precios")
    
    print("\nDatos de noticias:")
    print(news_data[['timestamp', 'title', 'sentiment_score']].tail() if not news_data.empty else "No hay datos de noticias")

if __name__ == "__main__":
    asyncio.run(main())
