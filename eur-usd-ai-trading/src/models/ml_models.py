"""
Modelos de Machine Learning y Deep Learning para Trading EUR/USD
Incluye LSTM, GRU, y análisis de sentimiento con BERT
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import joblib
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Estructura para predicciones del modelo"""
    price_prediction: float
    direction_prediction: str  # 'UP', 'DOWN', 'STABLE'
    confidence: float
    sentiment_influence: float
    technical_influence: float

class FeatureEngineer:
    """Ingeniero de características para el modelo"""
    
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
    
    def create_sequences(self, data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Crea secuencias para modelos de series temporales"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def prepare_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepara características para el modelo"""
        
        # Características técnicas
        technical_features = [
            'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'RSI', 'BB_Upper', 'BB_Lower'
        ]
        
        # Limpiar datos faltantes
        df_clean = df[technical_features].fillna(method='ffill').fillna(method='bfill')
                
        # Características derivadas
        df_clean['Price_Change'] = df_clean['Close'].pct_change()
        df_clean['Volume_Change'] = df_clean['Volume'].pct_change()
        df_clean['Volatility'] = df_clean['Close'].rolling(window=20).std()
        df_clean['Price_Position'] = (df_clean['Close'] - df_clean['BB_Lower']) / (df_clean['BB_Upper'] - df_clean['BB_Lower'])
        
        # Remover filas con NaN después de los cálculos
        df_clean = df_clean.dropna()
        
        if not self.is_fitted:
            # Escalar precios (target)
            prices = df_clean['Close'].values.reshape(-1, 1)
            self.price_scaler.fit(prices)
            
            # Escalar características
            features = df_clean.drop(['Close'], axis=1)
            self.feature_scaler.fit(features)
            self.is_fitted = True
        
        # Aplicar escalado
        scaled_prices = self.price_scaler.transform(df_clean['Close'].values.reshape(-1, 1))
        features = df_clean.drop(['Close'], axis=1)
        scaled_features = self.feature_scaler.transform(features)
        
        return {
            'prices': scaled_prices.flatten(),
            'features': scaled_features,
            'feature_names': features.columns.tolist(),
            'original_prices': df_clean['Close'].values
        }
    
    def create_training_data(self, prepared_data: Dict, sequence_length: int = 60) -> Dict:
        """Crea datos de entrenamiento para los modelos"""
        
        prices = prepared_data['prices']
        features = prepared_data['features']
        
        # Crear secuencias de precios para LSTM/GRU
        X_price_seq, y_price = self.create_sequences(prices, sequence_length)
        
        # Crear secuencias de características
        X_features_seq = []
        for i in range(sequence_length, len(features)):
            X_features_seq.append(features[i-sequence_length:i])
        X_features_seq = np.array(X_features_seq)
        
        # Características actuales para modelos tradicionales
        X_current_features = features[sequence_length:]
        
        # Crear etiquetas direccionales (clasificación)
        y_direction = []
        original_prices = prepared_data['original_prices']
        for i in range(sequence_length, len(original_prices)-1):
            current_price = original_prices[i]
            next_price = original_prices[i+1]
            
            if next_price > current_price * 1.001:  # Umbral del 0.1%
                y_direction.append(2)  # UP
            elif next_price < current_price * 0.999:
                y_direction.append(0)  # DOWN
            else:
                y_direction.append(1)  # STABLE
        
        y_direction = np.array(y_direction)
        
        # Ajustar tamaños
        min_length = min(len(X_price_seq), len(y_direction))
        X_price_seq = X_price_seq[:min_length]
        X_features_seq = X_features_seq[:min_length]
        X_current_features = X_current_features[:min_length]
        y_price = y_price[:min_length]
        y_direction = y_direction[:min_length]
        
        return {
            'X_price_sequences': X_price_seq,
            'X_feature_sequences': X_features_seq,
            'X_current_features': X_current_features,
            'y_price': y_price,
            'y_direction': y_direction
        }

class LSTMModel:
    """Modelo LSTM para predicción de precios"""
    
    def __init__(self, sequence_length: int = 60, feature_dim: int = 10):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.is_trained = False
    
    def build_model(self) -> keras.Model:
        """Construye la arquitectura del modelo LSTM"""
        
        # Input para secuencias de precios
        price_input = keras.Input(shape=(self.sequence_length, 1), name='price_input')
        
        # Input para secuencias de características
        feature_input = keras.Input(shape=(self.sequence_length, self.feature_dim), name='feature_input')
        
        # LSTM para precios
        price_lstm = layers.LSTM(50, return_sequences=True)(price_input)
        price_lstm = layers.Dropout(0.2)(price_lstm)
        price_lstm = layers.LSTM(50, return_sequences=False)(price_lstm)
        price_lstm = layers.Dropout(0.2)(price_lstm)
        
        # LSTM para características
        feature_lstm = layers.LSTM(100, return_sequences=True)(feature_input)
        feature_lstm = layers.Dropout(0.2)(feature_lstm)
        feature_lstm = layers.LSTM(100, return_sequences=False)(feature_lstm)
        feature_lstm = layers.Dropout(0.2)(feature_lstm)
        
        # Combinar ambas ramas
        combined = layers.concatenate([price_lstm, feature_lstm])
        combined = layers.Dense(100, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        combined = layers.Dense(50, activation='relu')(combined)
        
        # Outputs
        price_output = layers.Dense(1, activation='linear', name='price_output')(combined)
        direction_output = layers.Dense(3, activation='softmax', name='direction_output')(combined)
        
        model = keras.Model(
            inputs=[price_input, feature_input],
            outputs=[price_output, direction_output]
        )
        
        model.compile(
            optimizer='adam',
            loss={
                'price_output': 'mse',
                'direction_output': 'sparse_categorical_crossentropy'
            },
            metrics={
                'price_output': ['mae'],
                'direction_output': ['accuracy']
            },
            loss_weights={'price_output': 0.7, 'direction_output': 0.3}
        )
        
        return model
    
    def train(self, training_data: Dict, validation_split: float = 0.2, epochs: int = 100):
        """Entrena el modelo LSTM"""
        
        self.model = self.build_model()
        
        # Preparar datos
        X_price = training_data['X_price_sequences']
        X_features = training_data['X_feature_sequences']
        y_price = training_data['y_price']
        y_direction = training_data['y_direction']
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.001
        )
        
        # Entrenamiento
        history = self.model.fit(
            [X_price, X_features],
            [y_price, y_direction],
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X_price: np.ndarray, X_features: np.ndarray) -> ModelPrediction:
        """Realiza predicción"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        price_pred, direction_pred = self.model.predict([X_price, X_features], verbose=0)
        
        # Procesar predicciones
        predicted_price = float(price_pred[0][0])
        direction_probs = direction_pred[0]
        direction_class = np.argmax(direction_probs)
        confidence = float(np.max(direction_probs))
        
        direction_map = {0: 'DOWN', 1: 'STABLE', 2: 'UP'}
        
        return ModelPrediction(
            price_prediction=predicted_price,
            direction_prediction=direction_map[direction_class],
            confidence=confidence,
            sentiment_influence=0.0,  # Se calculará externamente
            technical_influence=confidence
        )

class GRUModel:
    """Modelo GRU alternativo"""
    
    def __init__(self, sequence_length: int = 60, feature_dim: int = 10):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.is_trained = False
    
    def build_model(self) -> keras.Model:
        """Construye la arquitectura del modelo GRU"""
        
        price_input = keras.Input(shape=(self.sequence_length, 1), name='price_input')
        feature_input = keras.Input(shape=(self.sequence_length, self.feature_dim), name='feature_input')
        
        # GRU para precios
        price_gru = layers.GRU(50, return_sequences=True)(price_input)
        price_gru = layers.Dropout(0.2)(price_gru)
        price_gru = layers.GRU(50, return_sequences=False)(price_gru)
        price_gru = layers.Dropout(0.2)(price_gru)
        
        # GRU para características
        feature_gru = layers.GRU(100, return_sequences=True)(feature_input)
        feature_gru = layers.Dropout(0.2)(feature_gru)
        feature_gru = layers.GRU(100, return_sequences=False)(feature_gru)
        feature_gru = layers.Dropout(0.2)(feature_gru)
        
        # Combinar con mecanismo de atención
        combined = layers.concatenate([price_gru, feature_gru])
        attention = layers.Dense(150, activation='tanh')(combined)
        attention = layers.Dense(1, activation='sigmoid')(attention)
        combined_weighted = layers.multiply([combined, attention])
        
        combined_weighted = layers.Dense(100, activation='relu')(combined_weighted)
        combined_weighted = layers.Dropout(0.2)(combined_weighted)
        combined_weighted = layers.Dense(50, activation='relu')(combined_weighted)
        
        price_output = layers.Dense(1, activation='linear', name='price_output')(combined_weighted)
        direction_output = layers.Dense(3, activation='softmax', name='direction_output')(combined_weighted)
        
        model = keras.Model(
            inputs=[price_input, feature_input],
            outputs=[price_output, direction_output]
        )
        
        model.compile(
            optimizer='adam',
            loss={
                'price_output': 'mse',
                'direction_output': 'sparse_categorical_crossentropy'
            },
            metrics={
                'price_output': ['mae'],
                'direction_output': ['accuracy']
            },
            loss_weights={'price_output': 0.7, 'direction_output': 0.3}
        )
        
        return model
    
    def train(self, training_data: Dict, validation_split: float = 0.2, epochs: int = 100):
        """Entrena el modelo GRU"""
        self.model = self.build_model()
        
        X_price = training_data['X_price_sequences']
        X_features = training_data['X_feature_sequences']
        y_price = training_data['y_price']
        y_direction = training_data['y_direction']
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = self.model.fit(
            [X_price, X_features],
            [y_price, y_direction],
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X_price: np.ndarray, X_features: np.ndarray) -> ModelPrediction:
        """Realiza predicción con GRU"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        price_pred, direction_pred = self.model.predict([X_price, X_features], verbose=0)
        
        predicted_price = float(price_pred[0][0])
        direction_probs = direction_pred[0]
        direction_class = np.argmax(direction_probs)
        confidence = float(np.max(direction_probs))
        
        direction_map = {0: 'DOWN', 1: 'STABLE', 2: 'UP'}
        
        return ModelPrediction(
            price_prediction=predicted_price,
            direction_prediction=direction_map[direction_class],
            confidence=confidence,
            sentiment_influence=0.0,
            technical_influence=confidence
        )

class RandomForestModel:
    """Modelo Random Forest como baseline"""
    
    def __init__(self):
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.direction_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def train(self, training_data: Dict):
        """Entrena los modelos Random Forest"""
        
        X = training_data['X_current_features']
        y_price = training_data['y_price']
        y_direction = training_data['y_direction']
        
        # Entrenar modelo de precios
        self.price_model.fit(X, y_price)
        
        # Entrenar modelo de dirección
        self.direction_model.fit(X, y_direction)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Realiza predicción con Random Forest"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        price_pred = self.price_model.predict(X.reshape(1, -1))[0]
        direction_pred = self.direction_model.predict(X.reshape(1, -1))[0]
        
        # Convertir predicción de dirección a clase
        direction_class = int(round(direction_pred))
        direction_class = max(0, min(2, direction_class))  # Limitar entre 0 y 2
        
        direction_map = {0: 'DOWN', 1: 'STABLE', 2: 'UP'}
        
        return ModelPrediction(
            price_prediction=price_pred,
            direction_prediction=direction_map[direction_class],
            confidence=0.7,  # Valor fijo para RF
            sentiment_influence=0.0,
            technical_influence=0.7
        )

class EnsembleModel:
    """Modelo ensemble que combina LSTM, GRU y Random Forest"""
    
    def __init__(self):
        self.lstm_model = None
        self.gru_model = None
        self.rf_model = None
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        """Entrena todos los modelos del ensemble"""
        
        logger.info("Preparando características...")
        prepared_data = self.feature_engineer.prepare_features(df)
        
        logger.info("Creando datos de entrenamiento...")
        training_data = self.feature_engineer.create_training_data(prepared_data)
        
        feature_dim = training_data['X_feature_sequences'].shape[2]
        
        # Dividir datos
        split_idx = int(len(training_data['y_price']) * (1 - test_size))
        
        train_data = {
            'X_price_sequences': training_data['X_price_sequences'][:split_idx],
            'X_feature_sequences': training_data['X_feature_sequences'][:split_idx],
            'X_current_features': training_data['X_current_features'][:split_idx],
            'y_price': training_data['y_price'][:split_idx],
            'y_direction': training_data['y_direction'][:split_idx]
        }
        
        # Entrenar LSTM
        logger.info("Entrenando modelo LSTM...")
        self.lstm_model = LSTMModel(feature_dim=feature_dim)
        self.lstm_model.train(train_data, epochs=50)
        
        # Entrenar GRU
        logger.info("Entrenando modelo GRU...")
        self.gru_model = GRUModel(feature_dim=feature_dim)
        self.gru_model.train(train_data, epochs=50)
        
        # Entrenar Random Forest
        logger.info("Entrenando modelo Random Forest...")
        self.rf_model = RandomForestModel()
        self.rf_model.train(train_data)
        
        self.is_trained = True
        logger.info("Entrenamiento del ensemble completado")
    
    def predict(self, recent_data: pd.DataFrame, sentiment_score: float = 0.0) -> ModelPrediction:
        """Realiza predicción ensemble"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        # Preparar datos para predicción
        prepared_data = self.feature_engineer.prepare_features(recent_data)
        
        if len(prepared_data['prices']) < 60:
            raise ValueError("Insufficient data for prediction (need at least 60 points)")
        
        # Tomar las últimas 60 observaciones
        X_price = prepared_data['prices'][-60:].reshape(1, 60, 1)
        X_features = prepared_data['features'][-60:].reshape(1, 60, -1)
        X_current = prepared_data['features'][-1]
        
        # Predicciones de cada modelo
        lstm_pred = self.lstm_model.predict(X_price, X_features)
        gru_pred = self.gru_model.predict(X_price, X_features)
        rf_pred = self.rf_model.predict(X_current)
        
        # Combinar predicciones (weighted average)
        weights = {'lstm': 0.4, 'gru': 0.4, 'rf': 0.2}
        
        final_price = (
            lstm_pred.price_prediction * weights['lstm'] +
            gru_pred.price_prediction * weights['gru'] +
            rf_pred.price_prediction * weights['rf']
        )
        
        # Determinar dirección final (voto mayoritario)
        directions = [lstm_pred.direction_prediction, gru_pred.direction_prediction, rf_pred.direction_prediction]
        final_direction = max(set(directions), key=directions.count)
        
        # Calcular confianza promedio ponderada
        final_confidence = (
            lstm_pred.confidence * weights['lstm'] +
            gru_pred.confidence * weights['gru'] +
            rf_pred.confidence * weights['rf']
        )
        
        # Incorporar análisis de sentimiento
        sentiment_influence = abs(sentiment_score) * 0.3  # Factor de influencia del sentimiento
        
        # Ajustar confianza basada en sentimiento
        if sentiment_score > 0.1 and final_direction == 'UP':
            final_confidence = min(1.0, final_confidence + sentiment_influence)
        elif sentiment_score < -0.1 and final_direction == 'DOWN':
            final_confidence = min(1.0, final_confidence + sentiment_influence)
        else:
            final_confidence = max(0.1, final_confidence - sentiment_influence)
        
        return ModelPrediction(
            price_prediction=final_price,
            direction_prediction=final_direction,
            confidence=final_confidence,
            sentiment_influence=sentiment_influence,
            technical_influence=final_confidence - sentiment_influence
        )
    
    def save_models(self, path_prefix: str = "models/"):
        """Guarda todos los modelos"""
        if self.is_trained:
            self.lstm_model.model.save(f"{path_prefix}lstm_model.h5")
            self.gru_model.model.save(f"{path_prefix}gru_model.h5")
            joblib.dump(self.rf_model, f"{path_prefix}rf_model.pkl")
            joblib.dump(self.feature_engineer, f"{path_prefix}feature_engineer.pkl")
        
    def load_models(self, path_prefix: str = "models/"):
        """Carga todos los modelos"""
        try:
            self.lstm_model = LSTMModel()
            self.lstm_model.model = keras.models.load_model(f"{path_prefix}lstm_model.h5")
            self.lstm_model.is_trained = True
            
            self.gru_model = GRUModel()
            self.gru_model.model = keras.models.load_model(f"{path_prefix}gru_model.h5")
            self.gru_model.is_trained = True
            
            self.rf_model = joblib.load(f"{path_prefix}rf_model.pkl")
            self.feature_engineer = joblib.load(f"{path_prefix}feature_engineer.pkl")
            
            self.is_trained = True
            logger.info("Modelos cargados exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")

# Función de ejemplo de uso
def train_ensemble_example():
    """Ejemplo de entrenamiento del ensemble"""
    import yfinance as yf
    
    # Obtener datos de ejemplo
    ticker = yf.Ticker("EURUSD=X")
    df = ticker.history(period="2y", interval="1h")
    
    # Agregar indicadores técnicos básicos
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    
    # Crear y entrenar ensemble
    ensemble = EnsembleModel()
    ensemble.train(df)
    
    # Hacer predicción de ejemplo
    recent_data = df.tail(100)  # Últimos 100 puntos
    prediction = ensemble.predict(recent_data, sentiment_score=0.2)
    
    print(f"Predicción de precio: {prediction.price_prediction}")
    print(f"Dirección: {prediction.direction_prediction}")
    print(f"Confianza: {prediction.confidence:.2f}")
    
    return ensemble

if __name__ == "__main__":
    ensemble = train_ensemble_example()
