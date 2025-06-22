"""
Análisis de Sentimiento Avanzado con BERT y Sistema de Backtesting
para Trading EUR/USD
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Análisis de Sentimiento Avanzado
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logger.warning("Transformers library not available. Using simple sentiment analysis.")

@dataclass
class SentimentResult:
    """Resultado del análisis de sentimiento"""
    text: str
    sentiment: str  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
    confidence: float
    impact_score: float
    keywords: List[str] = field(default_factory=list)

class AdvancedSentimentAnalyzer:
    """Analizador de sentimiento avanzado con BERT"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.financial_keywords = {
            'bullish': {'score': 1.0, 'weight': 1.5},
            'bearish': {'score': -1.0, 'weight': 1.5},
            'rally': {'score': 0.8, 'weight': 1.2},
            'crash': {'score': -0.9, 'weight': 1.3},
            'surge': {'score': 0.7, 'weight': 1.1},
            'plunge': {'score': -0.8, 'weight': 1.2},
            'strong': {'score': 0.6, 'weight': 1.0},
            'weak': {'score': -0.6, 'weight': 1.0},
            'growth': {'score': 0.5, 'weight': 1.0},
            'decline': {'score': -0.5, 'weight': 1.0},
            'positive': {'score': 0.4, 'weight': 0.8},
            'negative': {'score': -0.4, 'weight': 0.8},
            'optimistic': {'score': 0.5, 'weight': 0.9},
            'pessimistic': {'score': -0.5, 'weight': 0.9},
            'recovery': {'score': 0.6, 'weight': 1.1},
            'recession': {'score': -0.7, 'weight': 1.2}
        }
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo BERT para análisis financiero"""
        if BERT_AVAILABLE:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name
                )
                logger.info(f"Modelo BERT {self.model_name} cargado exitosamente")
            except Exception as e:
                logger.warning(f"Error cargando modelo BERT: {e}. Usando análisis simple.")
                self.sentiment_pipeline = None
        else:
            self.sentiment_pipeline = None
    
    def analyze_text(self, text: str) -> SentimentResult:
        """Analiza el sentimiento de un texto"""
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                sentiment='NEUTRAL',
                confidence=0.0,
                impact_score=0.0
            )
        
        # Análisis con BERT si está disponible
        if self.sentiment_pipeline:
            try:
                bert_result = self.sentiment_pipeline(text[:512])  # Limitar longitud
                bert_sentiment = bert_result[0]['label'].upper()
                bert_confidence = bert_result[0]['score']
                
                # Mapear etiquetas de FinBERT
                if bert_sentiment in ['POSITIVE', 'BULLISH']:
                    sentiment = 'POSITIVE'
                elif bert_sentiment in ['NEGATIVE', 'BEARISH']:
                    sentiment = 'NEGATIVE'
                else:
                    sentiment = 'NEUTRAL'
                
            except Exception as e:
                logger.warning(f"Error en análisis BERT: {e}")
                sentiment, bert_confidence = self._simple_sentiment_analysis(text)
        else:
            sentiment, bert_confidence = self._simple_sentiment_analysis(text)
        
        # Análisis de palabras clave financieras
        keywords_found = []
        keyword_score = 0.0
        text_lower = text.lower()
        
        for keyword, data in self.financial_keywords.items():
            if keyword in text_lower:
                keywords_found.append(keyword)
                keyword_score += data['score'] * data['weight']
        
        # Combinar scores
        if keywords_found:
            # Promedio ponderado entre BERT y keywords
            final_score = (bert_confidence * 0.7 + abs(keyword_score) * 0.3)
            
            # Ajustar sentimiento basado en keywords si hay conflicto
            if keyword_score > 0.5 and sentiment == 'NEGATIVE':
                sentiment = 'POSITIVE'
            elif keyword_score < -0.5 and sentiment == 'POSITIVE':
                sentiment = 'NEGATIVE'
        else:
            final_score = bert_confidence
        
        # Calcular impacto (qué tan relevante es para trading)
        impact_score = self._calculate_impact_score(text, keywords_found, final_score)
        
        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=final_score,
            impact_score=impact_score,
            keywords=keywords_found
        )
    
    def _simple_sentiment_analysis(self, text: str) -> Tuple[str, float]:
        """Análisis de sentimiento simple como fallback"""
        text_lower = text.lower()
        positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'profit']
        negative_words = ['bad', 'terrible', 'negative', 'down', 'fall', 'loss', 'decline']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'POSITIVE', 0.6
        elif neg_count > pos_count:
            return 'NEGATIVE', 0.6
        else:
            return 'NEUTRAL', 0.5
    
    def _calculate_impact_score(self, text: str, keywords: List[str], confidence: float) -> float:
        """Calcula el score de impacto para trading"""
        base_score = confidence
        
        # Factores que aumentan el impacto
        impact_factors = {
            'central bank': 1.3,
            'ecb': 1.3,
            'federal reserve': 1.3,
            'fed': 1.3,
            'interest rate': 1.2,
            'monetary policy': 1.2,
            'inflation': 1.1,
            'gdp': 1.1,
            'unemployment': 1.1,
            'trade war': 1.2,
            'brexit': 1.2,
            'euro': 1.1,
            'dollar': 1.1
        }
        
        text_lower = text.lower()
        impact_multiplier = 1.0
        
        for factor, multiplier in impact_factors.items():
            if factor in text_lower:
                impact_multiplier = max(impact_multiplier, multiplier)
        
        # Ajuste por número de keywords encontradas
        keyword_bonus = min(0.3, len(keywords) * 0.1)
        
        final_impact = min(1.0, base_score * impact_multiplier + keyword_bonus)
        return final_impact
    
    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """Analiza múltiples textos"""
        return [self.analyze_text(text) for text in texts]

# Sistema de Backtesting
@dataclass
class Trade:
    """Representa una operación de trading"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    position: str = 'LONG'  # 'LONG' o 'SHORT'
    quantity: float = 1.0
    profit_loss: float = 0.0
    is_open: bool = True
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_reason: str = ""

@dataclass
class BacktestResult:
    """Resultado del backtesting"""
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

class TradingStrategy(ABC):
    """Estrategia de trading abstracta"""
    
    @abstractmethod
    def generate_signal(self, data: Dict) -> str:
        """Genera señal de trading: 'BUY', 'SELL', 'HOLD'"""
        pass

class AITradingStrategy(TradingStrategy):
    """Estrategia basada en el modelo de IA"""
    
    def __init__(self, model, sentiment_weight: float = 0.3):
        self.model = model
        self.sentiment_weight = sentiment_weight
        self.min_confidence = 0.6
    
    def generate_signal(self, data: Dict) -> str:
        """Genera señal basada en predicción del modelo"""
        try:
            price_data = data.get('price_data')
            sentiment_score = data.get('sentiment_score', 0.0)
            
            if price_data is None or len(price_data) < 60:
                return 'HOLD'
            
            # Obtener predicción del modelo
            prediction = self.model.predict(price_data, sentiment_score)
            
            # Ajustar confianza con sentimiento
            adjusted_confidence = prediction.confidence
            if sentiment_score > 0.1 and prediction.direction_prediction == 'UP':
                adjusted_confidence = min(1.0, adjusted_confidence + self.sentiment_weight * sentiment_score)
            elif sentiment_score < -0.1 and prediction.direction_prediction == 'DOWN':
                adjusted_confidence = min(1.0, adjusted_confidence + self.sentiment_weight * abs(sentiment_score))
            
            # Generar señal
            if adjusted_confidence >= self.min_confidence:
                if prediction.direction_prediction == 'UP':
                    return 'BUY'
                elif prediction.direction_prediction == 'DOWN':
                    return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generando señal: {e}")
            return 'HOLD'

class BacktestEngine:
    """Motor de backtesting"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.0001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.current_capital = initial_capital
        self.current_position = None
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, 
                     price_data: pd.DataFrame,
                     sentiment_data: pd.DataFrame,
                     strategy: TradingStrategy,
                     risk_management: Dict = None) -> BacktestResult:
        """Ejecuta el backtesting"""
        
        if risk_management is None:
            risk_management = {
                'stop_loss_pct': 0.02,  # 2%
                'take_profit_pct': 0.04,  # 4%
                'max_position_size': 0.1  # 10% del capital
            }
        
        self._reset_backtest()
        
        # Combinar datos de precios y sentimiento
        combined_data = self._combine_data(price_data, sentiment_data)
        
        for i in range(60, len(combined_data)):  # Necesitamos al menos 60 puntos para el modelo
            current_data = {
                'price_data': combined_data.iloc[max(0, i-100):i],  # Últimos 100 puntos
                'sentiment_score': combined_data.iloc[i]['sentiment_score'],
                'current_price': combined_data.iloc[i]['Close']
            }
            
            signal = strategy.generate_signal(current_data)
            current_price = combined_data.iloc[i]['Close']
            current_time = combined_data.index[i]
            
            # Gestión de posición existente
            if self.current_position and self.current_position.is_open:
                self._check_exit_conditions(current_price, current_time, risk_management)
            
            # Nueva entrada
            if signal in ['BUY', 'SELL'] and (not self.current_position or not self.current_position.is_open):
                self._enter_position(signal, current_price, current_time, risk_management)
            
            # Actualizar equity curve
            current_value = self._calculate_portfolio_value(current_price)
            self.equity_curve.append(current_value)
        
        # Cerrar posición abierta al final
        if self.current_position and self.current_position.is_open:
            final_price = combined_data.iloc[-1]['Close']
            final_time = combined_data.index[-1]
            self._exit_position(final_price, final_time, "End of backtest")
        
        return self._calculate_results()
    
    def _reset_backtest(self):
        """Resetea el estado del backtest"""
        self.current_capital = self.initial_capital
        self.current_position = None
        self.trades = []
        self.equity_curve = []
    
    def _combine_data(self, price_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Combina datos de precios y sentimiento"""
        # Resample sentiment data to match price data frequency
        sentiment_resampled = sentiment_data.set_index('timestamp').resample('1H').mean()
        sentiment_resampled['sentiment_score'] = sentiment_resampled['sentiment_score'].fillna(0)
        
        # Merge with price data
        combined = price_data.join(sentiment_resampled[['sentiment_score']], how='left')
        combined['sentiment_score'] = combined['sentiment_score'].fillna(0)
        
        return combined
    
    def _enter_position(self, signal: str, price: float, time: datetime, risk_management: Dict):
        """Entra en una nueva posición"""
        position_size = min(
            self.current_capital * risk_management['max_position_size'],
            self.current_capital * 0.95  # Mantener algo de efectivo
        )
        
        quantity = position_size / price
        commission_cost = position_size * self.commission
        
        position_type = 'LONG' if signal == 'BUY' else 'SHORT'
        
        # Calcular stop loss y take profit
        if position_type == 'LONG':
            stop_loss = price * (1 - risk_management['stop_loss_pct'])
            take_profit = price * (1 + risk_management['take_profit_pct'])
        else:
            stop_loss = price * (1 + risk_management['stop_loss_pct'])
            take_profit = price * (1 - risk_management['take_profit_pct'])
        
        self.current_position = Trade(
            entry_time=time,
            entry_price=price,
            position=position_type,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_reason=f"{signal} signal"
        )
        
        self.current_capital -= commission_cost
        logger.info(f"Entered {position_type} position at {price} on {time}")
    
    def _check_exit_conditions(self, current_price: float, current_time: datetime, risk_management: Dict):
        """Verifica condiciones de salida"""
        if not self.current_position or not self.current_position.is_open:
            return
        
        exit_reason = None
        
        # Check stop loss
        if self.current_position.position == 'LONG' and current_price <= self.current_position.stop_loss:
            exit_reason = "Stop Loss"
        elif self.current_position.position == 'SHORT' and current_price >= self.current_position.stop_loss:
            exit_reason = "Stop Loss"
        
        # Check take profit
        elif self.current_position.position == 'LONG' and current_price >= self.current_position.take_profit:
            exit_reason = "Take Profit"
        elif self.current_position.position == 'SHORT' and current_price <= self.current_position.take_profit:
            exit_reason = "Take Profit"
        
        if exit_reason:
            self._exit_position(current_price, current_time, exit_reason)
    
    def _exit_position(self, price: float, time: datetime, reason: str):
        """Sale de la posición actual"""
        if not self.current_position or not self.current_position.is_open:
            return
        
        self.current_position.exit_time = time
        self.current_position.exit_price = price
        self.current_position.is_open = False
        
        # Calcular profit/loss
        if self.current_position.position == 'LONG':
            pnl = (price - self.current_position.entry_price) * self.current_position.quantity
        else:  # SHORT
            pnl = (self.current_position.entry_price - price) * self.current_position.quantity
        
        # Descontar comisiones
        commission_cost = price * self.current_position.quantity * self.commission
        pnl -= commission_cost
        
        self.current_position.profit_loss = pnl
        self.current_capital += pnl
        
        self.trades.append(self.current_position)
        logger.info(f"Exited position at {price} on {time} ({reason}). P&L: {pnl:.2f}")
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calcula el valor actual del portafolio"""
        if not self.current_position or not self.current_position.is_open:
            return self.current_capital
        
        # Valor de la posición abierta
        if self.current_position.position == 'LONG':
            position_value = current_price * self.current_position.quantity
            unrealized_pnl = (current_price - self.current_position.entry_price) * self.current_position.quantity
        else:  # SHORT
            position_value = self.current_position.entry_price * self.current_position.quantity
            unrealized_pnl = (self.current_position.entry_price - current_price) * self.current_position.quantity
        
        return self.current_capital + unrealized_pnl
    
    def _calculate_results(self) -> BacktestResult:
        """Calcula los resultados del backtesting"""
        if not self.trades:
            return BacktestResult(
                total_return=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_profit=0.0,
                avg_loss=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                trades=self.trades,
                equity_curve=self.equity_curve
            )
        
        # Métricas básicas
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        total_trades = len(self.trades)
        
        profits = [trade.profit_loss for trade in self.trades if trade.profit_loss > 0]
        losses = [trade.profit_loss for trade in self.trades if trade.profit_loss < 0]
        
        winning_trades = len(profits)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Máximo drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Sharpe ratio (simplificado)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1] if len(self.equity_curve) > 1 else [0]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades,
            equity_curve=self.equity_curve
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calcula el máximo drawdown"""
        if not self.equity_curve:
            return 0.0
        
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        
        return abs(np.min(drawdown))

def plot_backtest_results(result: BacktestResult, save_path: Optional[str] = None):
    """Visualiza los resultados del backtesting"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Equity curve
    ax1.plot(result.equity_curve)
    ax1.set_title('Equity Curve')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Portfolio Value')
    ax1.grid(True)
    
    # Distribución de P&L
    profits_losses = [trade.profit_loss for trade in result.trades]
    if profits_losses:
        ax2.hist(profits_losses, bins=20, alpha=0.7)
        ax2.set_title('P&L Distribution')
        ax2.set_xlabel('Profit/Loss')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.grid(True)
    
    # Métricas principales
    metrics_text = f"""
    Total Return: {result.total_return:.2f}%
    Total Trades: {result.total_trades}
    Win Rate: {result.win_rate:.2f}%
    Avg Profit: {result.avg_profit:.2f}
    Avg Loss: {result.avg_loss:.2f}
    Max Drawdown: {result.max_drawdown:.2f}%
    Sharpe Ratio: {result.sharpe_ratio:.2f}
    """
    ax3.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Performance Metrics')
    
    # Trades por mes
    if result.trades:
        trade_dates = [trade.entry_time for trade in result.trades]
        monthly_trades = pd.Series(trade_dates).dt.to_period('M').value_counts().sort_index()
        ax4.bar(range(len(monthly_trades)), monthly_trades.values)
        ax4.set_title('Trades per Month')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Number of Trades')
        ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Ejemplo de uso completo
def run_complete_backtest_example():
    """Ejemplo completo de backtesting con IA y análisis de sentimiento"""
    
    # Simular datos de ejemplo (en la práctica vendrían de la base de datos)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    n_points = len(dates)
    
    # Datos de precios simulados
    np.random.seed(42)
    price_data = pd.DataFrame({
        'Open': 1.0800 + np.random.randn(n_points) * 0.001,
        'High': 1.0810 + np.random.randn(n_points) * 0.001,
        'Low': 1.0790 + np.random.randn(n_points) * 0.001,
        'Close': 1.0800 + np.cumsum(np.random.randn(n_points) * 0.0001),
        'Volume': np.random.randint(1000, 5000, n_points)
    }, index=dates)
    
    # Datos de sentimiento simulados
    sentiment_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='4H')
    sentiment_data = pd.DataFrame({
        'timestamp': sentiment_dates,
        'sentiment_score': np.random.randn(len(sentiment_dates)) * 0.3
    })
    
    # Crear estrategia mock (en la práctica usaríamos el modelo entrenado)
    class MockStrategy(TradingStrategy):
        def generate_signal(self, data):
            # Estrategia simple: comprar si el sentimiento es positivo y precio está subiendo
            current_price = data['current_price']
            price_data = data['price_data']
            sentiment = data['sentiment_score']
            
            if len(price_data) < 2:
                return 'HOLD'
            
            price_momentum = (current_price - price_data['Close'].iloc[-10].mean()) / price_data['Close'].iloc[-10].mean()
            
            if sentiment > 0.1 and price_momentum > 0.001:
                return 'BUY'
            elif sentiment < -0.1 and price_momentum < -0.001:
                return 'SELL'
            else:
                return 'HOLD'
    
    # Ejecutar backtesting
    strategy = MockStrategy()
    backtest_engine = BacktestEngine(initial_capital=10000)
    
    result = backtest_engine.run_backtest(
        price_data=price_data,
        sentiment_data=sentiment_data,
        strategy=strategy
    )
    
    # Mostrar resultados
    print("=== RESULTADOS DEL BACKTESTING ===")
    print(f"Retorno Total: {result.total_return:.2f}%")
    print(f"Total de Trades: {result.total_trades}")
    print(f"Trades Ganadores: {result.winning_trades}")
    print(f"Trades Perdedores: {result.losing_trades}")
    print(f"Tasa de Acierto: {result.win_rate:.2f}%")
    print(f"Ganancia Promedio: {result.avg_profit:.2f}")
    print(f"Pérdida Promedio: {result.avg_loss:.2f}")
    print(f"Máximo Drawdown: {result.max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    
    # Visualizar resultados
    plot_backtest_results(result)
    
    return result

if __name__ == "__main__":
    # Ejemplo de análisis de sentimiento
    analyzer = AdvancedSentimentAnalyzer()
    
    sample_texts = [
        "The EUR/USD is showing strong bullish momentum after ECB rate decision",
        "Economic uncertainty continues to weigh on European markets",
        "Federal Reserve maintains dovish stance, supporting risk assets"
    ]
    
    print("=== ANÁLISIS DE SENTIMIENTO ===")
    for text in sample_texts:
        result = analyzer.analyze_text(text)
        print(f"Texto: {text}")
        print(f"Sentimiento: {result.sentiment} (Confianza: {result.confidence:.2f})")
        print(f"Impacto: {result.impact_score:.2f}")
        print(f"Keywords: {result.keywords}")
        print("-" * 50)
    
    # Ejecutar ejemplo de backtesting
    run_complete_backtest_example()
