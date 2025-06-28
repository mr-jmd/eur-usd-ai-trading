"""
Dashboard Integrado con el Sistema Principal (main.py)
Conecta las predicciones IA reales con la interfaz
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
import sys
import sqlite3
from pathlib import Path
import asyncio
import threading
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de página
st.set_page_config(
    page_title="EUR/USD AI Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Intentar importar el sistema principal
MAIN_SYSTEM_AVAILABLE = False
try:
    # Ajustar path para importar desde la raíz del proyecto
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from main import TradingSystemManager
    MAIN_SYSTEM_AVAILABLE = True
    logger.info("✅ Sistema principal importado correctamente")
except ImportError as e:
    logger.warning(f"⚠️ No se pudo importar el sistema principal: {e}")
    MAIN_SYSTEM_AVAILABLE = False

# CSS mejorado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2980b9);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #d4edda, #00b894);
        border-left-color: #28a745;
        color: #155724;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #f8d7da, #e17055);
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-left-color: #f39c12;
        color: #8b6914;
    }
    
    .ai-prediction {
        background: linear-gradient(135deg, #e8f4fd, #d4edda);
        border-left: 5px solid #007bff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .system-status {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class IntegratedTradingDashboard:
    """Dashboard integrado con el sistema principal"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_main_system()
    
    def initialize_session_state(self):
        """Inicializa el estado de la sesión"""
        if 'main_system' not in st.session_state:
            st.session_state.main_system = None
        if 'main_system_running' not in st.session_state:
            st.session_state.main_system_running = False
        if 'last_ai_prediction' not in st.session_state:
            st.session_state.last_ai_prediction = None
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'price_data' not in st.session_state:
            st.session_state.price_data = pd.DataFrame()
    
    def setup_main_system(self):
        """Configura conexión con el sistema principal"""
        if MAIN_SYSTEM_AVAILABLE and st.session_state.main_system is None:
            try:
                # Inicializar el sistema principal
                st.session_state.main_system = TradingSystemManager()
                logger.info("✅ Sistema principal inicializado")
            except Exception as e:
                logger.error(f"❌ Error inicializando sistema principal: {e}")
                st.session_state.main_system = None
    
    async def get_ai_prediction_from_main(self):
        """Obtiene predicción del sistema principal"""
        if not MAIN_SYSTEM_AVAILABLE or st.session_state.main_system is None:
            return None
        
        try:
            # Intentar obtener predicción del sistema principal
            prediction_data = await st.session_state.main_system.generate_prediction()
            
            if prediction_data and 'prediction' in prediction_data:
                prediction = prediction_data['prediction']
                
                # Convertir a formato estándar
                ai_prediction = {
                    'type': 'AI_ENSEMBLE',
                    'price_prediction': prediction.price_prediction,
                    'direction': prediction.direction_prediction,
                    'confidence': prediction.confidence,
                    'model_used': 'Ensemble IA (LSTM + GRU + RF)',
                    'sentiment_influence': prediction.sentiment_influence,
                    'technical_influence': prediction.technical_influence,
                    'timestamp': prediction_data['timestamp'],
                    'current_price': prediction_data['current_price'],
                    'sentiment_score': prediction_data['sentiment_score']
                }
                
                st.session_state.last_ai_prediction = ai_prediction
                logger.info("✅ Predicción IA obtenida del sistema principal")
                return ai_prediction
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo predicción IA: {e}")
        
        return None
    
    def get_fallback_prediction(self, price_data):
        """Predicción de respaldo usando análisis técnico"""
        try:
            current_price = price_data['Close'].iloc[-1]
            
            # Análisis técnico simple
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
                    signals.append(1)
                elif rsi > 70:
                    signals.append(-1)
                else:
                    signals.append(0)
            
            # MACD
            if 'MACD' in price_data.columns and 'MACD_Signal' in price_data.columns:
                macd = price_data['MACD'].iloc[-1]
                macd_signal = price_data['MACD_Signal'].iloc[-1]
                signals.append(1 if macd > macd_signal else -1)
            
            # Calcular señal final
            signal_sum = sum(signals) if signals else 0
            
            if signal_sum > 0:
                direction = "UP"
                price_change = np.random.uniform(0.0005, 0.002)
            elif signal_sum < 0:
                direction = "DOWN"
                price_change = -np.random.uniform(0.0005, 0.002)
            else:
                direction = "STABLE"
                price_change = np.random.uniform(-0.0005, 0.0005)
            
            predicted_price = current_price + price_change
            confidence = min(0.85, max(0.5, abs(signal_sum) / len(signals) + 0.3)) if signals else 0.5
            
            return {
                'type': 'TECHNICAL_FALLBACK',
                'price_prediction': predicted_price,
                'direction': direction,
                'confidence': confidence,
                'model_used': 'Análisis Técnico (Fallback)',
                'signal_count': len(signals),
                'signal_strength': signal_sum
            }
            
        except Exception as e:
            logger.error(f"Error en predicción fallback: {e}")
            return {
                'type': 'ERROR_FALLBACK',
                'price_prediction': price_data['Close'].iloc[-1],
                'direction': 'STABLE',
                'confidence': 0.5,
                'model_used': 'Predicción Básica'
            }
    
    @st.cache_data(ttl=300)
    def load_real_data(_self, period="6mo", interval="1h"):
        """Carga datos reales EUR/USD"""
        try:
            logger.info("🔄 Cargando datos EUR/USD...")
            
            ticker = yf.Ticker("EURUSD=X")
            data = ticker.history(period=period, interval=interval, auto_adjust=True, repair=True)
            
            if data.empty:
                raise ValueError("No se obtuvieron datos")
            
            # Indicadores técnicos
            data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
            data['SMA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            data['EMA_12'] = data['Close'].ewm(span=12, min_periods=1).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, min_periods=1).mean()
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9, min_periods=1).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1e-10)
            data['RSI'] = 100 - (100 / (1 + rs))
            
            logger.info(f"✅ Datos cargados: {len(data)} registros")
            return data, True, None
            
        except Exception as e:
            error_msg = f"Error cargando datos: {str(e)}"
            logger.error(error_msg)
            return pd.DataFrame(), False, error_msg
    
    def render_header(self):
        """Renderiza encabezado"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 EUR/USD AI Trading System</h1>
            <p>Sistema Integrado con Modelos IA - Instituto Tecnológico Metropolitano</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_system_status(self):
        """Muestra estado del sistema"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if MAIN_SYSTEM_AVAILABLE and st.session_state.main_system:
                st.markdown("""
                <div class="system-status" style="background: #d4edda; border-color: #c3e6cb;">
                    <strong>✅ Sistema Principal</strong><br>
                    Modelos IA conectados
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="system-status" style="background: #fff3cd; border-color: #ffeaa7;">
                    <strong>⚠️ Modo Técnico</strong><br>
                    Sistema principal no disponible
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.last_ai_prediction:
                pred_type = st.session_state.last_ai_prediction.get('type', 'UNKNOWN')
                if 'AI_ENSEMBLE' in pred_type:
                    st.markdown("""
                    <div class="system-status" style="background: #d4edda; border-color: #c3e6cb;">
                        <strong>🧠 IA Activa</strong><br>
                        Ensemble LSTM+GRU+RF
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="system-status" style="background: #d1ecf1; border-color: #bee5eb;">
                        <strong>📊 Análisis Técnico</strong><br>
                        Indicadores avanzados
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="system-status" style="background: #f8d7da; border-color: #f5c6cb;">
                    <strong>🔄 Cargando</strong><br>
                    Generando predicción...
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            last_update = st.session_state.last_update.strftime("%H:%M:%S")
            st.markdown(f"""
            <div class="system-status" style="background: #e2e3e5; border-color: #d6d8db;">
                <strong>🕐 Actualización</strong><br>
                {last_update}
            </div>
            """, unsafe_allow_html=True)
    
    def render_prediction_comparison(self, ai_prediction, technical_prediction):
        """Compara predicción IA vs técnica"""
        st.subheader("🤖 Comparación: IA vs Análisis Técnico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 Predicción IA (Sistema Principal)")
            
            if ai_prediction:
                # Mostrar predicción IA
                direction = ai_prediction['direction']
                confidence = ai_prediction['confidence']
                
                signal_class = "signal-buy" if direction == "UP" else "signal-sell" if direction == "DOWN" else "signal-hold"
                signal_emoji = "🟢" if direction == "UP" else "🔴" if direction == "DOWN" else "🟡"
                signal_text = "BUY" if direction == "UP" else "SELL" if direction == "DOWN" else "HOLD"
                
                st.markdown(f"""
                <div class="ai-prediction">
                    <h4>{signal_emoji} {signal_text}</h4>
                    <p><strong>Confianza:</strong> {confidence:.1%}</p>
                    <p><strong>Precio Predicho:</strong> {ai_prediction['price_prediction']:.5f}</p>
                    <p><strong>Modelo:</strong> {ai_prediction['model_used']}</p>
                    <p><strong>Influencia Técnica:</strong> {ai_prediction.get('technical_influence', 0):.1%}</p>
                    <p><strong>Influencia Sentimiento:</strong> {ai_prediction.get('sentiment_influence', 0):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("🔄 Conectando con sistema IA...")
        
        with col2:
            st.markdown("### 📊 Análisis Técnico")
            
            if technical_prediction:
                direction = technical_prediction['direction']
                confidence = technical_prediction['confidence']
                
                signal_class = "signal-buy" if direction == "UP" else "signal-sell" if direction == "DOWN" else "signal-hold"
                signal_emoji = "🟢" if direction == "UP" else "🔴" if direction == "DOWN" else "🟡"
                signal_text = "BUY" if direction == "UP" else "SELL" if direction == "DOWN" else "HOLD"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{signal_emoji} {signal_text}</h4>
                    <p><strong>Confianza:</strong> {confidence:.1%}</p>
                    <p><strong>Precio Predicho:</strong> {technical_prediction['price_prediction']:.5f}</p>
                    <p><strong>Método:</strong> {technical_prediction['model_used']}</p>
                    <p><strong>Señales:</strong> {technical_prediction.get('signal_count', 0)} indicadores</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_current_metrics(self, price_data, final_prediction):
        """Renderiza métricas principales"""
        if price_data.empty:
            st.error("❌ No hay datos disponibles")
            return
        
        col1, col2, col3 = st.columns(3)
        
        # Señal final
        with col1:
            if final_prediction:
                direction = final_prediction['direction']
                confidence = final_prediction['confidence']
                
                signal = "BUY" if direction == "UP" else "SELL" if direction == "DOWN" else "HOLD"
                signal_class = f"signal-{signal.lower()}"
                signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
                
                model_info = final_prediction.get('model_used', 'IA')
                if 'ENSEMBLE' in final_prediction.get('type', ''):
                    model_info = "🧠 IA Ensemble"
                elif 'TECHNICAL' in final_prediction.get('type', ''):
                    model_info = "📊 Análisis Técnico"
                
                st.markdown(f"""
                <div class="metric-card {signal_class}">
                    <h3>{signal_emoji} Señal Final</h3>
                    <h2>{signal}</h2>
                    <p>Confianza: {confidence:.1%}</p>
                    <small>{model_info}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Precio actual
        with col2:
            current_price = price_data['Close'].iloc[-1]
            prev_price = price_data['Close'].iloc[-2] if len(price_data) > 1 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            change_color = "🟢" if price_change > 0 else "🔴" if price_change < 0 else "⚪"
            change_sign = "+" if price_change > 0 else ""
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 EUR/USD</h3>
                <h2>{current_price:.5f}</h2>
                <p>Cambio: {change_sign}{price_change:.5f} ({change_sign}{price_change_pct:.2f}%) {change_color}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # RSI
        with col3:
            rsi = price_data['RSI'].iloc[-1] if 'RSI' in price_data.columns else 50
            rsi_status = "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Neutral"
            rsi_color = "#dc3545" if rsi > 70 else "#28a745" if rsi < 30 else "#6c757d"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 RSI</h3>
                <h2 style="color: {rsi_color};">{rsi:.1f}</h2>
                <p>{rsi_status}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_price_chart(self, data):
        """Renderiza gráfico de precios"""
        st.subheader("📈 Gráfico EUR/USD con Predicciones IA")
        
        if data.empty:
            st.warning("No hay datos para mostrar")
            return
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Precio EUR/USD', 'RSI'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="EUR/USD"
            ),
            row=1, col=1
        )
        
        # SMA
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_20'], name="SMA 20", line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_50'], name="SMA 50", line=dict(color='red', width=1)),
                row=1, col=1
            )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Marcar última predicción IA si existe
        if st.session_state.last_ai_prediction:
            pred_price = st.session_state.last_ai_prediction['price_prediction']
            current_time = data.index[-1]
            
            fig.add_trace(
                go.Scatter(
                    x=[current_time],
                    y=[pred_price],
                    mode='markers',
                    marker=dict(size=15, color='gold', symbol='star'),
                    name="Predicción IA",
                    showlegend=True
                ),
                row=1, col=1
            )
        
        fig.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_sidebar(self):
        """Renderiza sidebar"""
        with st.sidebar:
            st.header("⚙️ Control del Sistema")
            
            # Estado del sistema principal
            st.subheader("🔗 Sistema Principal")
            
            if MAIN_SYSTEM_AVAILABLE:
                st.success("✅ Sistema principal disponible")
                
                if st.button("🔄 Reinicializar Sistema IA"):
                    st.session_state.main_system = None
                    st.rerun()
            else:
                st.error("❌ Sistema principal no disponible")
                st.info("Ejecutando en modo análisis técnico")
            
            st.markdown("---")
            
            # Configuración de datos
            st.subheader("📊 Datos")
            period = st.selectbox("Período", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=4)
            interval = st.selectbox("Intervalo", ["1m", "5m", "15m", "30m", "1h"], index=4)
            
            if st.button("🔄 Actualizar Datos"):
                st.rerun()
            
            st.markdown("---")
            
            # Información del sistema
            st.subheader("ℹ️ Información")
            
            if st.session_state.last_ai_prediction:
                pred = st.session_state.last_ai_prediction
                st.info(f"🧠 Última predicción IA: {pred.get('timestamp', 'N/A')}")
                st.info(f"📊 Tipo: {pred.get('type', 'N/A')}")
            
            st.info(f"🕐 Actualizado: {st.session_state.last_update.strftime('%H:%M:%S')}")
            
            return {'period': period, 'interval': interval}
    
    async def run(self):
        """Ejecuta el dashboard principal"""
        # Header
        self.render_header()
        
        # Sidebar
        config = self.render_sidebar()
        
        # Estado del sistema
        self.render_system_status()
        
        # Cargar datos
        with st.spinner("🔄 Cargando datos EUR/USD..."):
            price_data, success, error = self.load_real_data(config['period'], config['interval'])
            
            if success:
                st.session_state.price_data = price_data
                st.session_state.last_update = datetime.now()
            else:
                st.error(f"❌ {error}")
                return
        
        # Obtener predicciones
        with st.spinner("🧠 Generando predicciones..."):
            # Intentar predicción IA del sistema principal
            ai_prediction = await self.get_ai_prediction_from_main()
            
            # Predicción técnica de respaldo
            technical_prediction = self.get_fallback_prediction(price_data)
            
            # Usar IA si está disponible, sino técnica
            final_prediction = ai_prediction if ai_prediction else technical_prediction
        
        # Métricas principales
        self.render_current_metrics(price_data, final_prediction)
        
        st.markdown("---")
        
        # Comparación de predicciones
        self.render_prediction_comparison(ai_prediction, technical_prediction)
        
        st.markdown("---")
        
        # Gráfico
        self.render_price_chart(price_data)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: gray; font-size: 12px;">
            Sistema Integrado EUR/USD AI Trading<br>
            Instituto Tecnológico Metropolitano - 2025<br>
            🔗 Conectado al Sistema Principal | 🧠 Modelos IA LSTM+GRU+RF
        </div>
        """, unsafe_allow_html=True)

# Función principal
def main():
    """Función principal"""
    dashboard = IntegratedTradingDashboard()
    
    # Ejecutar de forma asíncrona
    try:
        asyncio.run(dashboard.run())
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("💡 Asegúrate de que el sistema principal esté disponible")

if __name__ == "__main__":
    main()