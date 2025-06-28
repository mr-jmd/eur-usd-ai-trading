"""
Dashboard en Tiempo Real - EUR/USD AI Trading System
Conectado directamente con el sistema principal para mostrar predicciones IA en vivo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import threading
import time
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import logging
import sys
import os

# Configurar página
st.set_page_config(
    page_title="EUR/USD AI Trading - Tiempo Real",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para mejor visualización
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 2px solid #0ea5e9;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .signal-buy { 
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border-color: #22c55e;
        color: #166534;
    }
    
    .signal-sell { 
        background: linear-gradient(135deg, #fef2f2, #fecaca);
        border-color: #ef4444;
        color: #dc2626;
    }
    
    .signal-hold { 
        background: linear-gradient(135deg, #fefce8, #fef3c7);
        border-color: #f59e0b;
        color: #92400e;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #ef4444;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

class RealTimeDashboard:
    """Dashboard integrado con sistema principal"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_system_connection()
    
    def initialize_session_state(self):
        """Inicializa estado de sesión"""
        if 'trading_system' not in st.session_state:
            st.session_state.trading_system = None
        if 'last_prediction' not in st.session_state:
            st.session_state.last_prediction = None
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 30  # segundos
    
    def setup_system_connection(self):
        """Configura conexión con sistema principal"""
        try:
            # Importar sistema principal
            project_root = Path(__file__).parent.parent
            sys.path.insert(0, str(project_root))
            
            from src.main import TradingSystemManager
            
            if st.session_state.trading_system is None:
                st.session_state.trading_system = TradingSystemManager()
                
            return True
            
        except ImportError as e:
            st.error(f"❌ Error conectando con sistema principal: {e}")
            return False
    
    async def get_real_time_prediction(self):
        """Obtiene predicción en tiempo real del sistema"""
        try:
            if st.session_state.trading_system:
                # Generar predicción actual
                prediction_data = await st.session_state.trading_system.generate_prediction()
                
                if prediction_data:
                    st.session_state.last_prediction = prediction_data
                    
                    # Agregar a historial
                    st.session_state.prediction_history.append({
                        'timestamp': prediction_data['timestamp'],
                        'signal': prediction_data['signal'],
                        'confidence': prediction_data['prediction'].confidence,
                        'price': prediction_data['current_price'],
                        'predicted_price': prediction_data['prediction'].price_prediction
                    })
                    
                    # Mantener solo últimas 100 predicciones
                    if len(st.session_state.prediction_history) > 100:
                        st.session_state.prediction_history = st.session_state.prediction_history[-100:]
                    
                    return prediction_data
            
            return None
            
        except Exception as e:
            st.error(f"Error obteniendo predicción: {e}")
            return None
    
    def get_cached_prediction(self):
        """Obtiene predicción del cache si está disponible"""
        if st.session_state.trading_system:
            return st.session_state.trading_system.get_cached_prediction()
        return None
    
    def render_live_header(self):
        """Renderiza encabezado con indicador en vivo"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div class="main-header">
                <h1>🤖 EUR/USD AI Trading System</h1>
                <p><span class="live-indicator"></span> DATOS EN TIEMPO REAL</p>
                <small>Predicciones IA Actualizadas • Instituto Tecnológico Metropolitano</small>
            </div>
            """, unsafe_allow_html=True)
    
    def render_prediction_panel(self, prediction_data):
        """Panel principal de predicción IA"""
        st.subheader("🧠 Predicción IA en Tiempo Real")
        
        if prediction_data:
            prediction = prediction_data['prediction']
            signal = prediction_data['signal']
            current_price = prediction_data['current_price']
            timestamp = prediction_data['timestamp']
            
            # Determinar estilo de señal
            signal_class = f"signal-{signal.lower()}"
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[signal]
            
            # Panel de predicción principal
            st.markdown(f"""
            <div class="prediction-card {signal_class}">
                <h2>{signal_emoji} SEÑAL: {signal}</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div>
                        <h4>💰 Precio Actual</h4>
                        <h3>{current_price:.5f}</h3>
                    </div>
                    <div>
                        <h4>🎯 Precio Predicho</h4>
                        <h3>{prediction.price_prediction:.5f}</h3>
                    </div>
                    <div>
                        <h4>📊 Confianza IA</h4>
                        <h3>{prediction.confidence:.1%}</h3>
                    </div>
                    <div>
                        <h4>⏰ Actualizado</h4>
                        <h3>{timestamp.strftime('%H:%M:%S')}</h3>
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <strong>Modelo:</strong> {prediction_data.get('model_type', 'IA Ensemble')} |
                    <strong>Datos:</strong> {prediction_data.get('data_points', 'N/A')} puntos |
                    <strong>Sentimiento:</strong> {prediction_data.get('sentiment_score', 0):.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas detalladas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                price_change = prediction.price_prediction - current_price
                change_pct = (price_change / current_price) * 100
                st.metric(
                    "Cambio Esperado",
                    f"{price_change:+.5f}",
                    f"{change_pct:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Influencia Técnica",
                    f"{prediction.technical_influence:.1%}",
                    "Indicadores"
                )
            
            with col3:
                st.metric(
                    "Influencia Sentimiento",
                    f"{prediction.sentiment_influence:.1%}",
                    "Noticias/Social"
                )
            
            with col4:
                system_status = st.session_state.trading_system.get_system_status()
                st.metric(
                    "Estado Sistema",
                    "🟢 ACTIVO" if system_status['is_running'] else "🟡 STANDBY",
                    f"Modelos: {sum(system_status['components'].values())}/3"
                )
        
        else:
            st.warning("🔄 Conectando con sistema IA...")
            st.info("💡 El sistema está inicializando. Las predicciones aparecerán en breve.")
    
    def render_prediction_history(self):
        """Historial de predicciones"""
        st.subheader("📈 Historial de Predicciones")
        
        if st.session_state.prediction_history:
            # Convertir a DataFrame
            df_history = pd.DataFrame(st.session_state.prediction_history)
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            
            # Gráfico de señales
            fig = go.Figure()
            
            # Precios actuales
            fig.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=df_history['price'],
                mode='lines',
                name='Precio Real',
                line=dict(color='blue', width=2)
            ))
            
            # Precios predichos
            fig.add_trace(go.Scatter(
                x=df_history['timestamp'],
                y=df_history['predicted_price'],
                mode='lines',
                name='Precio Predicho',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            # Señales BUY
            buy_signals = df_history[df_history['signal'] == 'BUY']
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals['timestamp'],
                    y=buy_signals['price'],
                    mode='markers',
                    name='Señal BUY',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ))
            
            # Señales SELL
            sell_signals = df_history[df_history['signal'] == 'SELL']
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals['timestamp'],
                    y=sell_signals['price'],
                    mode='markers',
                    name='Señal SELL',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ))
            
            fig.update_layout(
                title="Evolución de Precios y Señales IA",
                xaxis_title="Tiempo",
                yaxis_title="Precio EUR/USD",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas del historial
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_signals = len(df_history)
                buy_count = len(df_history[df_history['signal'] == 'BUY'])
                sell_count = len(df_history[df_history['signal'] == 'SELL'])
                hold_count = len(df_history[df_history['signal'] == 'HOLD'])
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Distribución de Señales</h4>
                    <p>BUY: {buy_count} ({buy_count/total_signals:.1%})</p>
                    <p>SELL: {sell_count} ({sell_count/total_signals:.1%})</p>
                    <p>HOLD: {hold_count} ({hold_count/total_signals:.1%})</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_confidence = df_history['confidence'].mean()
                max_confidence = df_history['confidence'].max()
                min_confidence = df_history['confidence'].min()
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 Confianza del Modelo</h4>
                    <p>Promedio: {avg_confidence:.1%}</p>
                    <p>Máxima: {max_confidence:.1%}</p>
                    <p>Mínima: {min_confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                latest_timestamp = df_history['timestamp'].iloc[-1]
                time_since_last = datetime.now() - latest_timestamp.to_pydatetime()
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⏰ Información Temporal</h4>
                    <p>Última actualización: {latest_timestamp.strftime('%H:%M:%S')}</p>
                    <p>Hace: {time_since_last.seconds//60}m {time_since_last.seconds%60}s</p>
                    <p>Total predicciones: {total_signals}</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("📊 El historial se poblará conforme se generen predicciones")
    
    def render_system_controls(self):
        """Controles del sistema"""
        st.sidebar.header("⚙️ Controles del Sistema")
        
        # Estado del sistema
        if st.session_state.trading_system:
            status = st.session_state.trading_system.get_system_status()
            
            if status['is_initialized']:
                st.sidebar.success("✅ Sistema Conectado")
            else:
                st.sidebar.error("❌ Sistema No Disponible")
            
            st.sidebar.info(f"🧠 Modelos IA: {sum(status['components'].values())}/3")
            st.sidebar.info(f"📊 Predicción fresca: {'Sí' if status['has_fresh_prediction'] else 'No'}")
        
        st.sidebar.markdown("---")
        
        # Configuración de actualización
        st.sidebar.subheader("🔄 Actualización Automática")
        
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "Actualización automática",
            value=st.session_state.auto_refresh
        )
        
        st.session_state.refresh_interval = st.sidebar.slider(
            "Intervalo (segundos)",
            min_value=5,
            max_value=300,
            value=st.session_state.refresh_interval,
            step=5
        )
        
        # Botones de control
        if st.sidebar.button("🔄 Actualizar Ahora"):
            st.rerun()
        
        if st.sidebar.button("🧹 Limpiar Historial"):
            st.session_state.prediction_history = []
            st.success("Historial limpiado")
        
        st.sidebar.markdown("---")
        
        # Información del sistema
        st.sidebar.subheader("ℹ️ Sistema")
        st.sidebar.info(f"🕐 Actualizado: {datetime.now().strftime('%H:%M:%S')}")
        st.sidebar.info(f"📡 Modo: Tiempo Real")
        
        # Exportar datos
        if st.sidebar.button("📥 Exportar Historial"):
            if st.session_state.prediction_history:
                df_export = pd.DataFrame(st.session_state.prediction_history)
                csv = df_export.to_csv(index=False)
                st.sidebar.download_button(
                    "💾 Descargar CSV",
                    csv,
                    f"predicciones_eur_usd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
    
    async def run_dashboard(self):
        """Ejecuta el dashboard principal"""
        # Header
        self.render_live_header()
        
        # Controles laterales
        self.render_system_controls()
        
        # Obtener predicción actual
        if st.session_state.auto_refresh:
            with st.spinner("🤖 Obteniendo predicción IA..."):
                prediction_data = await self.get_real_time_prediction()
        else:
            prediction_data = self.get_cached_prediction()
        
        # Panel principal de predicción
        self.render_prediction_panel(prediction_data)
        
        st.markdown("---")
        
        # Historial de predicciones
        self.render_prediction_history()
        
        # Auto-refresh
        if st.session_state.auto_refresh and prediction_data:
            time.sleep(st.session_state.refresh_interval)
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: gray;">
            🤖 EUR/USD AI Trading System - Tiempo Real<br>
            Instituto Tecnológico Metropolitano • 2025<br>
            <small>Datos actualizados automáticamente cada {refresh_interval} segundos</small>
        </div>
        """.format(refresh_interval=st.session_state.refresh_interval), unsafe_allow_html=True)

# Función principal
def main():
    """Función principal del dashboard"""
    dashboard = RealTimeDashboard()
    
    try:
        # Ejecutar dashboard de forma asíncrona
        asyncio.run(dashboard.run_dashboard())
    except Exception as e:
        st.error(f"❌ Error en dashboard: {e}")
        st.info("💡 Asegúrate de que el sistema principal esté ejecutándose")
        
        # Botón para reintentar
        if st.button("🔄 Reintentar Conexión"):
            st.rerun()

if __name__ == "__main__":
    main()
