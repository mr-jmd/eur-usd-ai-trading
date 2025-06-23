"""
Sistema de Trading EUR/USD con IA - Arquitectura Principal
Basado en el proyecto del Instituto Tecnológico Metropolitano
VERSIÓN OPTIMIZADA PARA YAHOO FINANCE
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
    """Recolector optimizado de datos EUR/USD con Yahoo Finance"""
    
    def __init__(self, symbol: str = "EURUSD=X"):
        self.symbol = symbol
        self.cache = {}  # Cache simple para evitar llamadas repetidas
        
    async def collect_data(self, period: str = "1y", interval: str = "1h") -> pd.DataFrame:
        """Recolecta datos históricos optimizados de precios"""
        cache_key = f"{self.symbol}_{period}_{interval}"
        
        try:
            logger.info(f"Collecting EUR/USD data - Period: {period}, Interval: {interval}")
            
            # Verificar cache (datos de menos de 1 hora)
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if datetime.now() - cached_time < timedelta(hours=1):
                    logger.info(f"Using cached data ({len(cached_data)} points)")
                    return cached_data
            
            # Configurar parámetros optimizados para Yahoo Finance
            ticker = yf.Ticker(self.symbol)
            
            # Usar configuración robusta
            data = ticker.history(
                period=period,
                interval=interval,
                prepost=False,        # No incluir pre/post market
                auto_adjust=True,     # Ajustar por splits/dividendos
                back_adjust=False,    # No usar back-adjustment
                repair=True,          # Reparar datos malos automáticamente
                keepna=False,         # Remover NAs
                actions=True          # Incluir dividendos/splits
            )
            
            if data.empty:
                logger.warning(f"No data returned from Yahoo Finance for {self.symbol}")
                return pd.DataFrame()
            
            logger.info(f"Raw data from Yahoo Finance: {len(data)} records")
            
            # Validación y limpieza de datos
            data = self._validate_and_clean_data(data)
            
            # Agregar indicadores técnicos
            data = self._add_technical_indicators(data)
            
            # Cachear datos
            self.cache[cache_key] = (data.copy(), datetime.now())
            
            logger.info(f"✅ Successfully collected {len(data)} EUR/USD data points")
            logger.info(f"📅 Data range: {data.index[0]} to {data.index[-1]}")
            logger.info(f"💰 Price range: {data['Close'].min():.5f} - {data['Close'].max():.5f}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting EUR/USD data: {e}")
            return pd.DataFrame()
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida y limpia los datos de Yahoo Finance"""
        logger.info(f"🔍 Validating data: {len(df)} initial records")
        
        if df.empty:
            return df
        
        # Remover filas con datos faltantes en columnas críticas
        critical_columns = ['Open', 'High', 'Low', 'Close']
        initial_count = len(df)
        df = df.dropna(subset=critical_columns)
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} rows with missing critical data")
        
        # Validar que High >= Low, Open, Close
        invalid_high_low = df['High'] < df['Low']
        if invalid_high_low.any():
            logger.warning(f"Found {invalid_high_low.sum()} records with High < Low, fixing...")
            df.loc[invalid_high_low, 'High'] = df.loc[invalid_high_low, ['Open', 'Close']].max(axis=1)
            df.loc[invalid_high_low, 'Low'] = df.loc[invalid_high_low, ['Open', 'Close']].min(axis=1)
        
        # Validar que Close esté entre Low y High
        close_above_high = df['Close'] > df['High']
        close_below_low = df['Close'] < df['Low']
        
        if close_above_high.any():
            logger.warning(f"Fixed {close_above_high.sum()} Close prices above High")
            df.loc[close_above_high, 'Close'] = df.loc[close_above_high, 'High']
            
        if close_below_low.any():
            logger.warning(f"Fixed {close_below_low.sum()} Close prices below Low")
            df.loc[close_below_low, 'Close'] = df.loc[close_below_low, 'Low']
        
        # Detectar y suavizar outliers extremos (cambios > 5%)
        if len(df) > 1:
            price_changes = df['Close'].pct_change().abs()
            outliers = price_changes > 0.05  # 5% cambio
            if outliers.any():
                outlier_count = outliers.sum()
                logger.warning(f"Found {outlier_count} potential outliers (>5% price change)")
                
                # Suavizar outliers en lugar de eliminarlos
                for idx in df[outliers].index:
                    idx_pos = df.index.get_loc(idx)
                    if idx_pos > 0:
                        prev_close = df['Close'].iloc[idx_pos - 1]
                        current_close = df.loc[idx, 'Close']
                        # Limitar el cambio al 2%
                        max_change = 0.02
                        if current_close > prev_close:
                            df.loc[idx, 'Close'] = prev_close * (1 + max_change)
                        else:
                            df.loc[idx, 'Close'] = prev_close * (1 - max_change)
                        
                        # Ajustar OHLC para consistencia
                        df.loc[idx, 'High'] = max(df.loc[idx, ['Open', 'Close']].max(), df.loc[idx, 'High'])
                        df.loc[idx, 'Low'] = min(df.loc[idx, ['Open', 'Close']].min(), df.loc[idx, 'Low'])
        
        # Validar rango de precios EUR/USD razonable (0.8 - 1.5)
        valid_range = (df['Close'] >= 0.8) & (df['Close'] <= 1.5)
        invalid_count = (~valid_range).sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} records with unrealistic EUR/USD prices")
            df = df[valid_range]
        
        # Manejar columna de volumen
        if 'Volume' in df.columns:
            # Yahoo Finance a veces tiene volumen 0 para forex
            df.loc[df['Volume'] < 0, 'Volume'] = 0
            # Si todo el volumen es 0, generar volumen sintético
            if df['Volume'].sum() == 0:
                df['Volume'] = np.random.randint(1000, 5000, len(df))
                logger.info("Generated synthetic volume data (Yahoo Finance has no forex volume)")
        else:
            # Agregar volumen sintético si no existe
            df['Volume'] = np.random.randint(1000, 5000, len(df))
            logger.info("Added synthetic volume column")
        
        # Asegurar que Dividends y Stock Splits existen
        if 'Dividends' not in df.columns:
            df['Dividends'] = 0.0
        if 'Stock Splits' not in df.columns:
            df['Stock Splits'] = 0.0
        
        logger.info(f"✅ Data validation complete: {len(df)} valid records")
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega indicadores técnicos optimizados"""
        try:
            if df.empty:
                logger.warning("Empty DataFrame received for technical indicators")
                return df
            
            logger.info(f"📊 Adding technical indicators to {len(df)} data points")
            
            # Crear una copia para trabajar
            df_copy = df.copy()
            
            # Para datasets muy pequeños, usar valores por defecto
            if len(df_copy) < 20:
                logger.warning(f"Very limited data for technical indicators: {len(df_copy)} points")
                df_copy['SMA_20'] = df_copy['Close']
                df_copy['SMA_50'] = df_copy['Close']
                df_copy['EMA_12'] = df_copy['Close']
                df_copy['EMA_26'] = df_copy['Close']
                df_copy['MACD'] = 0
                df_copy['MACD_Signal'] = 0
                df_copy['RSI'] = 50
                df_copy['BB_Upper'] = df_copy['Close'] * 1.02
                df_copy['BB_Lower'] = df_copy['Close'] * 0.98
                return df_copy
            
            # SMA (Simple Moving Average) - usar ventanas adaptables
            sma_20_window = min(20, max(2, len(df_copy) // 3))
            sma_50_window = min(50, max(2, len(df_copy) // 2))
            
            df_copy['SMA_20'] = df_copy['Close'].rolling(window=sma_20_window, min_periods=1).mean()
            df_copy['SMA_50'] = df_copy['Close'].rolling(window=sma_50_window, min_periods=1).mean()
            
            # Agregar SMA_200 si hay suficientes datos
            if len(df_copy) >= 100:
                sma_200_window = min(200, len(df_copy) // 2)
                df_copy['SMA_200'] = df_copy['Close'].rolling(window=sma_200_window, min_periods=50).mean()
            
            # EMA (Exponential Moving Average)
            ema_12_span = min(12, max(2, len(df_copy) // 4))
            ema_26_span = min(26, max(2, len(df_copy) // 3))
            
            df_copy['EMA_12'] = df_copy['Close'].ewm(span=ema_12_span, min_periods=1).mean()
            df_copy['EMA_26'] = df_copy['Close'].ewm(span=ema_26_span, min_periods=1).mean()
            
            # MACD
            df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
            macd_signal_span = min(9, max(2, len(df_copy) // 5))
            df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=macd_signal_span, min_periods=1).mean()
            df_copy['MACD_Histogram'] = df_copy['MACD'] - df_copy['MACD_Signal']
            
            # RSI (Relative Strength Index)
            rsi_window = min(14, max(2, len(df_copy) // 4))
            if len(df_copy) >= rsi_window:
                delta = df_copy['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
                
                # Evitar división por cero
                loss = loss.replace(0, 1e-10)
                rs = gain / loss
                df_copy['RSI'] = 100 - (100 / (1 + rs))
            else:
                # RSI simplificado para pocos datos
                df_copy['RSI'] = 50  # Valor neutral
            
            # Bollinger Bands
            bb_window = min(20, max(2, len(df_copy) // 3))
            rolling_mean = df_copy['Close'].rolling(window=bb_window, min_periods=1).mean()
            rolling_std = df_copy['Close'].rolling(window=bb_window, min_periods=1).std()
            
            # Manejar casos donde std es 0 o NaN
            rolling_std = rolling_std.fillna(df_copy['Close'].std())
            rolling_std = rolling_std.replace(0, df_copy['Close'].std() if df_copy['Close'].std() > 0 else 0.001)
            
            df_copy['BB_Upper'] = rolling_mean + (rolling_std * 2)
            df_copy['BB_Lower'] = rolling_mean - (rolling_std * 2)
            df_copy['BB_Width'] = df_copy['BB_Upper'] - df_copy['BB_Lower']
            
            # Posición dentro de las Bollinger Bands
            bb_range = df_copy['BB_Upper'] - df_copy['BB_Lower']
            bb_range = bb_range.replace(0, 0.001)  # Evitar división por cero
            df_copy['BB_Position'] = (df_copy['Close'] - df_copy['BB_Lower']) / bb_range
            
            # Stochastic Oscillator (si hay suficientes datos)
            stoch_window = min(14, max(2, len(df_copy) // 4))
            if len(df_copy) >= stoch_window:
                low_min = df_copy['Low'].rolling(window=stoch_window, min_periods=1).min()
                high_max = df_copy['High'].rolling(window=stoch_window, min_periods=1).max()
                
                stoch_range = high_max - low_min
                stoch_range = stoch_range.replace(0, 0.001)
                df_copy['Stoch_K'] = 100 * ((df_copy['Close'] - low_min) / stoch_range)
                df_copy['Stoch_D'] = df_copy['Stoch_K'].rolling(window=3, min_periods=1).mean()
            
            # Average True Range (ATR)
            atr_window = min(14, max(2, len(df_copy) // 4))
            if len(df_copy) >= 2:
                high_low = df_copy['High'] - df_copy['Low']
                high_close = np.abs(df_copy['High'] - df_copy['Close'].shift())
                low_close = np.abs(df_copy['Low'] - df_copy['Close'].shift())
                
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                df_copy['ATR'] = true_range.rolling(window=atr_window, min_periods=1).mean()
            
            # Verificar y limpiar valores infinitos o NaN
            numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                # Reemplazar infinitos con NaN
                df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan)
            
            # Rellenar valores NaN
            df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
            
            # Si aún hay NaN, usar la media de la columna
            for col in numeric_columns:
                if df_copy[col].isna().any():
                    col_mean = df_copy[col].mean()
                    if pd.isna(col_mean):
                        df_copy[col] = df_copy[col].fillna(0)
                    else:
                        df_copy[col] = df_copy[col].fillna(col_mean)
            
            # Verificación final
            nan_count = df_copy.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"Still {nan_count} NaN values after processing, filling with 0")
                df_copy = df_copy.fillna(0)
            
            logger.info("✅ Technical indicators added successfully")
            return df_copy
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            logger.error(f"DataFrame shape: {df.shape}")
            logger.error(f"DataFrame columns: {df.columns.tolist()}")
            
            # En caso de error, devolver DataFrame original con indicadores básicos
            try:
                df['SMA_20'] = df['Close']
                df['SMA_50'] = df['Close'] 
                df['EMA_12'] = df['Close']
                df['EMA_26'] = df['Close']
                df['MACD'] = 0
                df['MACD_Signal'] = 0
                df['RSI'] = 50
                df['BB_Upper'] = df['Close'] * 1.02
                df['BB_Lower'] = df['Close'] * 0.98
            except:
                pass
                
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
            logger.error(f"Error collecting news: {e}")
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
    """Gestor de base de datos mejorado"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializa las tablas de la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de precios con todas las columnas posibles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                dividends REAL DEFAULT 0,
                stock_splits REAL DEFAULT 0,
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
        logger.info("✅ Database initialized successfully")
    
    def save_price_data(self, df: pd.DataFrame):
        """Guarda datos de precios en la base de datos"""
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided to save_price_data")
                return
            
            # Crear una copia del DataFrame para modificar
            df_to_save = df.copy()
            
            # Normalizar nombres de columnas a minúsculas
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits',
                'SMA_20': 'sma_20',
                'SMA_50': 'sma_50', 
                'SMA_200': 'sma_200',
                'EMA_12': 'ema_12',
                'EMA_26': 'ema_26',
                'RSI': 'rsi',
                'MACD': 'macd',
                'MACD_Signal': 'macd_signal',
                'MACD_Histogram': 'macd_histogram',
                'BB_Upper': 'bb_upper',
                'BB_Lower': 'bb_lower',
                'BB_Width': 'bb_width',
                'BB_Position': 'bb_position',
                'Stoch_K': 'stoch_k',
                'Stoch_D': 'stoch_d',
                'ATR': 'atr'
            }
            
            # Aplicar mapeo de columnas
            for old_name, new_name in column_mapping.items():
                if old_name in df_to_save.columns:
                    if new_name not in df_to_save.columns:
                        df_to_save[new_name] = df_to_save[old_name]
                    if old_name != new_name:
                        df_to_save.drop(columns=[old_name], inplace=True)
            
            # Asegurar columnas básicas
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df_to_save.columns:
                    logger.warning(f"Missing required column {col}, using close price as fallback")
                    df_to_save[col] = df_to_save.get('close', 1.0800)
            
            # Agregar columnas faltantes con valores por defecto
            default_columns = {
                'dividends': 0.0,
                'stock_splits': 0.0,
                'sma_20': None, 'sma_50': None, 'sma_200': None,
                'ema_12': None, 'ema_26': None,
                'rsi': 50.0,
                'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
                'bb_upper': None, 'bb_lower': None, 'bb_width': None, 'bb_position': 0.5,
                'stoch_k': 50.0, 'stoch_d': 50.0,
                'atr': None
            }
            
            for col_name, default_value in default_columns.items():
                if col_name not in df_to_save.columns:
                    df_to_save[col_name] = default_value
            
            # Llenar None con valores calculados o 0
            for col in df_to_save.columns:
                if df_to_save[col].isna().all():
                    if col in ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower']:
                        df_to_save[col] = df_to_save.get('close', 1.0800)
                    elif col in ['bb_width', 'atr']:
                        df_to_save[col] = 0.001  # Valor pequeño para anchura/volatilidad
                    else:
                        df_to_save[col] = 0.0
            
            # Guardar en la base de datos
            conn = sqlite3.connect(self.db_path)
            df_to_save.to_sql('price_data', conn, if_exists='append', index_label='timestamp')
            conn.close()
            
            logger.info(f"✅ Saved {len(df_to_save)} price records to database")
            
        except Exception as e:
            logger.error(f"Error saving price data to database: {e}")
            logger.error(f"DataFrame columns: {df.columns.tolist()}")
            logger.error(f"DataFrame shape: {df.shape}")
    
    def save_news_data(self, df: pd.DataFrame):
        """Guarda datos de noticias en la base de datos"""
        try:
            if df.empty:
                logger.warning("Empty news DataFrame provided")
                return
                
            conn = sqlite3.connect(self.db_path)
            df.to_sql('news_data', conn, if_exists='append', index=False)
            conn.close()
            logger.info(f"✅ Saved {len(df)} news records to database")
        except Exception as e:
            logger.error(f"Error saving news data to database: {e}")
    
    def get_latest_data(self, table: str, limit: int = 1000) -> pd.DataFrame:
        """Obtiene los datos más recientes de una tabla"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error getting data from {table}: {e}")
            return pd.DataFrame()

class DataPipeline:
    """Pipeline principal de datos optimizado"""
    
    def __init__(self, news_api_key: Optional[str] = None):
        self.price_collector = PriceDataCollector()
        self.news_collector = NewsDataCollector(news_api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.db_manager = DatabaseManager()
    
    async def run_data_collection(self):
        """Ejecuta la recolección completa de datos"""
        logger.info("🚀 Starting data collection pipeline...")
        
        try:
            # Recolectar datos de precios
            logger.info("📊 Collecting price data...")
            price_data = await self.price_collector.collect_data()
            
            if not price_data.empty:
                logger.info(f"✅ Collected {len(price_data)} price records")
                self.db_manager.save_price_data(price_data)
            else:
                logger.error("❌ No price data collected")
            
            # Recolectar noticias
            logger.info("📰 Collecting news data...")
            try:
                news_data = await self.news_collector.collect_data()
                
                if not news_data.empty:
                    # Analizar sentimientos
                    logger.info("🧠 Analyzing sentiment...")
                    combined_text = news_data['title'] + ' ' + news_data['description']
                    news_data['sentiment_score'] = self.sentiment_analyzer.batch_analyze(combined_text.tolist())
                    
                    logger.info(f"✅ Collected {len(news_data)} news records with sentiment analysis")
                    self.db_manager.save_news_data(news_data)
                else:
                    logger.warning("⚠️ No news data collected, using mock data")
                    news_data = self.news_collector._get_mock_news_data()
                    if not news_data.empty:
                        combined_text = news_data['title'] + ' ' + news_data['description']
                        news_data['sentiment_score'] = self.sentiment_analyzer.batch_analyze(combined_text.tolist())
                        
            except Exception as e:
                logger.error(f"Error collecting news data: {e}")
                news_data = self.news_collector._get_mock_news_data()
                if not news_data.empty:
                    combined_text = news_data['title'] + ' ' + news_data['description']
                    news_data['sentiment_score'] = self.sentiment_analyzer.batch_analyze(combined_text.tolist())
        
            logger.info("✅ Data collection pipeline completed")
            return price_data, news_data
            
        except Exception as e:
            logger.error(f"❌ Error in data collection pipeline: {e}")
            return pd.DataFrame(), pd.DataFrame()

# Ejemplo de uso optimizado
async def main():
    """Función principal de ejemplo"""
    logger.info("🚀 Starting optimized EUR/USD data collection...")
    
    # Crear pipeline
    pipeline = DataPipeline()
    
    # Ejecutar recolección
    price_data, news_data = await pipeline.run_data_collection()
    
    # Mostrar resultados
    if not price_data.empty:
        print("\n📊 Price Data Summary:")
        print(f"Records: {len(price_data)}")
        print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
        print(f"Price range: {price_data['Close'].min():.5f} - {price_data['Close'].max():.5f}")
        print(f"Columns: {list(price_data.columns)}")
        
        print("\n📈 Latest prices:")
        print(price_data[['Open', 'High', 'Low', 'Close', 'RSI', 'MACD']].tail())
    
    if not news_data.empty:
        print(f"\n📰 News Data: {len(news_data)} articles")
        print("Recent headlines with sentiment:")
        for _, row in news_data.head(3).iterrows():
            sentiment = "📈 Positive" if row['sentiment_score'] > 0.1 else "📉 Negative" if row['sentiment_score'] < -0.1 else "📊 Neutral"
            print(f"  {sentiment}: {row['title'][:60]}...")

if __name__ == "__main__":
    asyncio.run(main())