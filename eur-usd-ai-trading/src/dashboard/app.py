"""
Dashboard Interactivo para Sistema de Trading EUR/USD con IA
Interfaz principal del usuario para monitoreo y control
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import sqlite3
import json
import time
from typing import Dict, List, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="EUR/USD AI Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
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
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2980b9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .signal-buy {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    .signal-sell {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    .signal-hold {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    """Clase principal del dashboard"""
    
    def __init__(self):
        self.db_path = "trading_data.db"
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Inicializa el estado de la sesión"""
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'current_signal' not in st.session_state:
            st.session_state.current_signal = "HOLD"
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
    
    def get_sample_data(self) -> Dict:
        """Genera datos de muestra para demostración"""
        # En producción, estos datos vendrían de la base de datos real
        current_time = datetime.now()
        
        # Datos de precios (últimas 100 horas)
        dates = pd.date_range(end=current_time, periods=100, freq='H')
        base_price = 1.0800
        
        price_data = pd.DataFrame({
            'timestamp': dates,
            'open': base_price + np.random.randn(100) * 0.001,
            'high': base_price + 0.002 + np.random.randn(100) * 0.001,
            'low': base_price - 0.002 + np.random.randn(100) * 0.001,
            'close': base_price + np.cumsum(np.random.randn(100) * 0.0001),
            'volume': np.random.randint(1000, 5000, 100)
        })
        
        # Indicadores técnicos
        price_data['sma_20'] = price_data['close'].rolling(20).mean()
        price_data['sma_50'] = price_data['close'].rolling(50).mean()
        price_data['rsi'] = 50 + np.random.randn(100) * 15  # RSI simulado
        
        # Datos de noticias recientes
        news_data = [
            {
                'timestamp': current_time - timedelta(hours=i),
                'title': f'EUR/USD Market Update {i}',
                'sentiment': np.random.choice(['POSITIVE', 'NEGATIVE', 'NEUTRAL']),
                'confidence': np.random.uniform(0.6, 0.95),
                'impact': np.random.uniform(0.3, 0.9)
            }
            for i in range(10)
        ]
        
        # Métricas del modelo
        model_metrics = {
            'accuracy': 67.8,
            'precision': 71.2,
            'recall': 64.5,
            'f1_score': 67.6,
            'last_prediction': {
                'price': price_data['close'].iloc[-1] + np.random.uniform(-0.001, 0.001),
                'direction': np.random.choice(['UP', 'DOWN', 'STABLE']),
                'confidence': np.random.uniform(0.6, 0.9)
            }
        }
        
        # Resultados de backtesting
        backtest_results = {
            'total_return': 15.7,
            'win_rate': 68.3,
            'sharpe_ratio': 1.42,
            'max_drawdown': 8.5,
            'total_trades': 124
        }
        
        return {
            'price_data': price_data,
            'news_data': news_data,
            'model_metrics': model_metrics,
            'backtest_results': backtest_results
        }
    
    def render_header(self):
        """Renderiza el encabezado principal"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 EUR/USD AI Trading System</h1>
            <p>Sistema Inteligente de Trading con Análisis de Sentimiento</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Renderiza la barra lateral"""
        with st.sidebar:
            st.header("⚙️ Configuración")
            
            # Configuración del modelo
            st.subheader("Modelo")
            model_status = st.selectbox(
                "Estado del Modelo",
                ["Cargado", "Entrenando", "Error"]
            )
            
            if model_status == "Cargado":
                st.success("✅ Modelo cargado correctamente")
            elif model_status == "Entrenando":
                st.warning("🔄 Modelo en entrenamiento...")
            else:
                st.error("❌ Error en el modelo")
            
            # Configuración de trading
            st.subheader("Trading")
            auto_trading = st.checkbox("Trading Automático")
            risk_level = st.slider("Nivel de Riesgo", 1, 10, 5)
            min_confidence = st.slider("Confianza Mínima", 0.5, 1.0, 0.7, 0.05)
            
            # Configuración de datos
            st.subheader("Datos")
            auto_refresh = st.checkbox("Actualización Automática", value=False)
            refresh_interval = st.slider("Intervalo (segundos)", 10, 300, 60)
            
            if st.button("🔄 Actualizar Datos"):
                st.session_state.last_update = datetime.now()
                st.experimental_rerun()
            
            # Información del sistema
            st.subheader("Sistema")
            st.info(f"Última actualización: {st.session_state.last_update.strftime('%H:%M:%S')}")
            
            return {
                'auto_trading': auto_trading,
                'risk_level': risk_level,
                'min_confidence': min_confidence,
                'auto_refresh': auto_refresh,
                'refresh_interval': refresh_interval
            }
    
    def render_current_signal(self, data: Dict, config: Dict):
        """Renderiza la señal actual de trading"""
        prediction = data['model_metrics']['last_prediction']
        
        # Determinar señal basada en predicción y configuración
        if prediction['confidence'] >= config['min_confidence']:
            if prediction['direction'] == 'UP':
                signal = "BUY"
                signal_class = "signal-buy"
                signal_emoji = "🟢"
            elif prediction['direction'] == 'DOWN':
                signal = "SELL"
                signal_class = "signal-sell"
                signal_emoji = "🔴"
            else:
                signal = "HOLD"
                signal_class = "signal-hold"
                signal_emoji = "🟡"
        else:
            signal = "HOLD"
            signal_class = "signal-hold"
            signal_emoji = "🟡"
        
        st.session_state.current_signal = signal
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card {signal_class}">
                <h3>{signal_emoji} Señal Actual</h3>
                <h2>{signal}</h2>
                <p>Confianza: {prediction['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            current_price = data['price_data']['close'].iloc[-1]
            predicted_price = prediction['price']
            price_change = ((predicted_price - current_price) / current_price) * 100
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Precio Actual</h3>
                <h2>{current_price:.5f}</h2>
                <p>Predicción: {predicted_price:.5f} ({price_change:+.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            sentiment_score = np.mean([
                1 if news['sentiment'] == 'POSITIVE' else -1 if news['sentiment'] == 'NEGATIVE' else 0
                for news in data['news_data'][:5]
            ])
            
            sentiment_emoji = "😊" if sentiment_score > 0.2 else "😐" if sentiment_score > -0.2 else "😔"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{sentiment_emoji} Sentimiento</h3>
                <h2>{sentiment_score:.2f}</h2>
                <p>Análisis de noticias recientes</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_price_chart(self, data: Dict):
        """Renderiza el gráfico de precios"""
        st.subheader("📊 Gráfico de Precios EUR/USD")
        
        price_data = data['price_data']
        
        # Crear gráfico de candlestick
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Precio', 'Volumen', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_data['timestamp'],
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name="EUR/USD"
            ),
            row=1, col=1
        )
        
        # Medias móviles
        fig.add_trace(
            go.Scatter(
                x=price_data['timestamp'],
                y=price_data['sma_20'],
                name="SMA 20",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_data['timestamp'],
                y=price_data['sma_50'],
                name="SMA 50",
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
        
        # Volumen
        fig.add_trace(
            go.Bar(
                x=price_data['timestamp'],
                y=price_data['volume'],
                name="Volumen",
                marker_color='rgba(158,202,225,0.6)'
            ),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=price_data['timestamp'],
                y=price_data['rsi'],
                name="RSI",
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # Líneas de RSI (sobrecompra/sobreventa)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_news_sentiment(self, data: Dict):
        """Renderiza análisis de noticias y sentimiento"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📰 Noticias Recientes")
            
            for news in data['news_data'][:5]:
                sentiment_color = {
                    'POSITIVE': 'green',
                    'NEGATIVE': 'red',
                    'NEUTRAL': 'gray'
                }[news['sentiment']]
                
                st.markdown(f"""
                <div style="border-left: 4px solid {sentiment_color}; padding-left: 10px; margin-bottom: 10px;">
                    <strong>{news['title']}</strong><br>
                    <small style="color: gray;">{news['timestamp'].strftime('%Y-%m-%d %H:%M')}</small><br>
                    <span style="color: {sentiment_color};">
                        {news['sentiment']} ({news['confidence']:.1%})
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📈 Análisis de Sentimiento")
            
            # Gráfico de distribución de sentimientos
            sentiment_counts = pd.Series([news['sentiment'] for news in data['news_data']]).value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Distribución de Sentimientos",
                color_discrete_map={
                    'POSITIVE': '#28a745',
                    'NEGATIVE': '#dc3545',
                    'NEUTRAL': '#6c757d'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Evolución del sentimiento
            sentiment_over_time = pd.DataFrame(data['news_data'])
            sentiment_over_time['sentiment_numeric'] = sentiment_over_time['sentiment'].map({
                'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1
            })
            
            fig2 = px.line(
                sentiment_over_time,
                x='timestamp',
                y='sentiment_numeric',
                title="Evolución del Sentimiento",
                labels={'sentiment_numeric': 'Sentimiento', 'timestamp': 'Tiempo'}
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig2, use_container_width=True)
    
    def render_model_performance(self, data: Dict):
        """Renderiza métricas de rendimiento del modelo"""
        st.subheader("🤖 Rendimiento del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            metrics = data['model_metrics']
            
            # Métricas principales
            st.metric("Precisión", f"{metrics['accuracy']:.1f}%")
            st.metric("Precisión (Precision)", f"{metrics['precision']:.1f}%")
            st.metric("Recall", f"{metrics['recall']:.1f}%")
            st.metric("F1-Score", f"{metrics['f1_score']:.1f}%")
            
        with col2:
            # Gráfico de radar para métricas
            metrics_df = pd.DataFrame({
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Valor': [metrics['accuracy'], metrics['precision'], 
                         metrics['recall'], metrics['f1_score']]
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=metrics_df['Valor'],
                theta=metrics_df['Métrica'],
                fill='toself',
                name='Modelo Actual'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title="Métricas del Modelo"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_backtest_results(self, data: Dict):
        """Renderiza resultados de backtesting"""
        st.subheader("📊 Resultados de Backtesting")
        
        results = data['backtest_results']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Retorno Total", f"{results['total_return']:.1f}%")
        
        with col2:
            st.metric("Tasa de Acierto", f"{results['win_rate']:.1f}%")
        
        with col3:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
        
        with col5:
            st.metric("Total Trades", results['total_trades'])
        
        # Gráfico de equity curve simulado
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        equity_curve = 10000 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.01))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_curve,
            mode='lines',
            name='Equity Curve',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Curva de Capital (Backtesting)",
            xaxis_title="Fecha",
            yaxis_title="Capital ($)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_trading_controls(self, config: Dict):
        """Renderiza controles de trading"""
        st.subheader("🎛️ Controles de Trading")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🟢 Ejecutar BUY", use_container_width=True):
                st.success("Orden de COMPRA enviada")
                st.balloons()
        
        with col2:
            if st.button("🔴 Ejecutar SELL", use_container_width=True):
                st.error("Orden de VENTA enviada")
        
        with col3:
            if st.button("⏹️ Cerrar Posición", use_container_width=True):
                st.info("Posición cerrada")
        
        # Panel de órdenes activas
        st.subheader("📋 Órdenes Activas")
        
        # Datos de ejemplo de órdenes
        orders_data = pd.DataFrame({
            'ID': ['ORD001', 'ORD002', 'ORD003'],
            'Tipo': ['BUY', 'SELL', 'BUY'],
            'Precio': [1.08123, 1.08456, 1.08234],
            'Cantidad': [10000, 15000, 8000],
            'Estado': ['Pending', 'Filled', 'Pending'],
            'Tiempo': ['10:30:15', '10:28:45', '10:25:30']
        })
        
        st.dataframe(orders_data, use_container_width=True)
    
    def run(self):
        """Ejecuta el dashboard principal"""
        self.render_header()
        
        # Configuración de la barra lateral
        config = self.render_sidebar()
        
        # Obtener datos
        data = self.get_sample_data()
        
        # Auto-refresh
        if config['auto_refresh']:
            time.sleep(config['refresh_interval'])
            st.experimental_rerun()
        
        # Señal actual y métricas principales
        self.render_current_signal(data, config)
        
        # Separador
        st.markdown("---")
        
        # Gráfico de precios
        self.render_price_chart(data)
        
        # Separador
        st.markdown("---")
        
        # Análisis de noticias y sentimiento
        self.render_news_sentiment(data)
        
        # Separador
        st.markdown("---")
        
        # Rendimiento del modelo y backtesting
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_model_performance(data)
        
        with col2:
            self.render_backtest_results(data)
        
        # Separador
        st.markdown("---")
        
        # Controles de trading
        self.render_trading_controls(config)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: gray; font-size: 12px;">
            Sistema de Trading EUR/USD con Inteligencia Artificial<br>
            Instituto Tecnológico Metropolitano - 2025
        </div>
        """, unsafe_allow_html=True)

# Script principal para ejecutar el dashboard
def main():
    """Función principal"""
    dashboard = TradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()

# Para ejecutar este dashboard, usar el comando:
# streamlit run trading_dashboard.py

# Alternativamente, aquí está una versión simplificada usando Dash:

"""
Dashboard alternativo usando Dash (descomentar para usar):

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("EUR/USD AI Trading System", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H3("Señal Actual"),
            html.H2(id="current-signal", style={'color': 'green'})
        ], className='four columns'),
        
        html.Div([
            html.H3("Precio Actual"),
            html.H2(id="current-price")
        ], className='four columns'),
        
        html.Div([
            html.H3("Sentimiento"),
            html.H2(id="sentiment-score")
        ], className='four columns'),
    ], className='row'),
    
    dcc.Graph(id='price-chart'),
    
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Actualizar cada minuto
        n_intervals=0
    )
])

@app.callback(
    [Output('current-signal', 'children'),
     Output('current-price', 'children'),
     Output('sentiment-score', 'children'),
     Output('price-chart', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Generar datos de ejemplo
    price = 1.0800 + np.random.randn() * 0.001
    signal = np.random.choice(['BUY', 'SELL', 'HOLD'])
    sentiment = np.random.uniform(-1, 1)
    
    # Crear gráfico
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='H')
    prices = 1.0800 + np.cumsum(np.random.randn(100) * 0.0001)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='EUR/USD'))
    fig.update_layout(title='EUR/USD Price Chart')
    
    return signal, f"{price:.5f}", f"{sentiment:.2f}", fig

if __name__ == '__main__':
    app.run_server(debug=True)
"""
