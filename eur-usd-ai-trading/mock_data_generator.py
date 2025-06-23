"""
Generador de datos mock para desarrollo y testing
Crea datos sintéticos realistas para EUR/USD
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class MockDataGenerator:
    """Generador de datos mock para desarrollo"""
    
    def __init__(self, base_price: float = 1.0800, volatility: float = 0.001):
        self.base_price = base_price
        self.volatility = volatility
        np.random.seed(42)  # Para resultados reproducibles
    
    def generate_price_data(self, 
                          start_date: str = None, 
                          end_date: str = None,
                          periods: int = 2000,
                          freq: str = '1H') -> pd.DataFrame:
        """Genera datos de precios históricos sintéticos"""
        
        if start_date and end_date:
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        else:
            # Por defecto, generar últimos N períodos
            end_time = datetime.now()
            if freq == '1H':
                start_time = end_time - timedelta(hours=periods)
            elif freq == '1D':
                start_time = end_time - timedelta(days=periods)
            else:
                start_time = end_time - timedelta(hours=periods)
            
            dates = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        n_points = len(dates)
        
        # Generar serie de precios con random walk y tendencia
        returns = np.random.normal(0, self.volatility, n_points)
        
        # Agregar algunas tendencias y ciclos
        trend = np.sin(np.arange(n_points) * 2 * np.pi / 100) * 0.0005
        seasonal = np.sin(np.arange(n_points) * 2 * np.pi / 24) * 0.0002  # Ciclo diario
        
        returns += trend + seasonal
        
        # Calcular precios
        prices = self.base_price + np.cumsum(returns)
        
        # Generar OHLC
        close_prices = prices
        
        # Open = precio anterior + pequeño gap
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = self.base_price
        open_prices += np.random.normal(0, self.volatility * 0.1, n_points)
        
        # High y Low basados en volatilidad intraday
        intraday_volatility = np.random.uniform(0.0001, self.volatility * 2, n_points)
        high_prices = np.maximum(open_prices, close_prices) + intraday_volatility
        low_prices = np.minimum(open_prices, close_prices) - intraday_volatility
        
        # Volumen sintético
        base_volume = 2500
        volume = np.random.poisson(base_volume, n_points)
        
        # Crear DataFrame
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=dates)
        
        # Agregar indicadores técnicos
        df = self._add_technical_indicators(df)
        
        logger.info(f"Generated {len(df)} price data points from {df.index[0]} to {df.index[-1]}")
        return df
    
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
    
    def generate_news_data(self, 
                          start_date: str = None,
                          num_articles: int = 100) -> pd.DataFrame:
        """Genera datos de noticias sintéticas"""
        
        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = datetime.now() - timedelta(days=30)
        
        # Generar timestamps aleatorios
        timestamps = []
        for i in range(num_articles):
            random_hours = np.random.randint(0, 30 * 24)  # Últimos 30 días
            timestamp = start + timedelta(hours=random_hours)
            timestamps.append(timestamp)
        
        timestamps.sort()
        
        # Plantillas de noticias
        positive_templates = [
            "ECB signals dovish monetary policy supporting EUR strength",
            "European economic data shows strong GDP growth",
            "EUR/USD breaks resistance as investors show confidence",
            "Positive inflation data boosts Euro against Dollar",
            "Strong employment figures support European currency",
            "Trade surplus increases Euro appeal among investors",
            "ECB President optimistic about European economic recovery",
            "Rising commodity prices favor Euro denomination",
        ]
        
        negative_templates = [
            "Federal Reserve hints at aggressive rate hikes pressuring EUR",
            "European economic uncertainty weighs on EUR/USD",
            "Dollar strength continues as EUR faces headwinds",
            "ECB concerns about inflation target impact Euro",
            "Geopolitical tensions affect European currency stability",
            "Energy crisis impacts European economic outlook",
            "Trade deficit concerns pressure Euro valuation",
            "Political uncertainty in Europe affects currency markets",
        ]
        
        neutral_templates = [
            "EUR/USD consolidates in tight range amid mixed signals",
            "Currency markets await central bank policy decisions",
            "Technical analysis shows EUR/USD in sideways trend",
            "Mixed economic data keeps EUR/USD range-bound",
            "Market participants assess economic indicators",
            "Currency pair shows limited movement in Asian session",
            "Traders monitor upcoming economic releases",
            "EUR/USD maintains stability ahead of key events",
        ]
        
        articles = []
        for timestamp in timestamps:
            # Decidir el tipo de noticia (con sesgo hacia neutral)
            sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], 
                                            p=[0.3, 0.3, 0.4])
            
            if sentiment_type == 'positive':
                title = np.random.choice(positive_templates)
                sentiment_score = np.random.uniform(0.3, 0.9)
            elif sentiment_type == 'negative':
                title = np.random.choice(negative_templates)
                sentiment_score = np.random.uniform(-0.9, -0.3)
            else:
                title = np.random.choice(neutral_templates)
                sentiment_score = np.random.uniform(-0.2, 0.2)
            
            articles.append({
                'timestamp': timestamp,
                'title': title,
                'description': f"Market analysis of {title.lower()}",
                'content': f"Detailed analysis: {title} according to financial experts.",
                'source': np.random.choice(['Reuters', 'Bloomberg', 'Financial Times', 'MarketWatch']),
                'sentiment_score': sentiment_score
            })
        
        df = pd.DataFrame(articles)
        logger.info(f"Generated {len(df)} news articles")
        return df
    
    def generate_sample_dataset(self, save_to_csv: bool = True) -> tuple:
        """Genera un dataset completo de muestra"""
        
        # Generar datos de precios (últimos 2 años)
        price_data = self.generate_price_data(
            periods=17520,  # 2 años * 365 días * 24 horas
            freq='1H'
        )
        
        # Generar noticias
        news_data = self.generate_news_data(
            start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
            num_articles=200
        )
        
        if save_to_csv:
            # Guardar en archivos CSV
            price_data.to_csv('data/mock_price_data.csv')
            news_data.to_csv('data/mock_news_data.csv', index=False)
            logger.info("Mock data saved to CSV files")
        
        return price_data, news_data

# Función auxiliar para usar en el sistema principal
def get_mock_data_if_needed():
    """Obtiene datos mock si los datos reales no están disponibles"""
    try:
        generator = MockDataGenerator()
        price_data, news_data = generator.generate_sample_dataset(save_to_csv=False)
        return price_data, news_data
    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        return pd.DataFrame(), pd.DataFrame()

if __name__ == "__main__":
    # Ejemplo de uso
    generator = MockDataGenerator()
    
    # Generar datos
    price_data, news_data = generator.generate_sample_dataset()
    
    print("Price Data Sample:")
    print(price_data.tail())
    print(f"\nPrice Data Shape: {price_data.shape}")
    
    print("\nNews Data Sample:")
    print(news_data.tail())
    print(f"\nNews Data Shape: {news_data.shape}")
    
    # Mostrar estadísticas
    print(f"\nPrice Range: {price_data['Close'].min():.5f} - {price_data['Close'].max():.5f}")
    print(f"Sentiment Range: {news_data['sentiment_score'].min():.2f} - {news_data['sentiment_score'].max():.2f}")
