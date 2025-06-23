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
        
        logger.info(f"Input DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # Verificar que el DataFrame no esté vacío
        if df.empty:
            logger.error("Input DataFrame is empty")
            return {
                'prices': np.array([]),
                'features': np.array([]),
                'feature_names': [],
                'original_prices': np.array([])
            }
        
        # Normalizar nombres de columnas (Yahoo Finance usa mayúsculas)
        df_normalized = df.copy()
        
        # Mapear columnas a nombres estándar
        column_mapping = {
            'Close': 'close',
            'Open': 'open', 
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df_normalized.columns:
                df_normalized[new_name] = df_normalized[old_name]
        
        # Verificar columnas esenciales
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in df_normalized.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {df_normalized.columns.tolist()}")
            return {
                'prices': np.array([]),
                'features': np.array([]),
                'feature_names': [],
                'original_prices': np.array([])
            }
        
        # Características técnicas disponibles
        potential_features = [
            'close', 'volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'RSI', 'BB_Upper', 'BB_Lower'
        ]
        
        # Usar solo las características que existen
        available_features = [col for col in potential_features if col in df_normalized.columns]
        
        if not available_features:
            logger.error("No valid features found in DataFrame")
            return {
                'prices': np.array([]),
                'features': np.array([]),
                'feature_names': [],
                'original_prices': np.array([])
            }
        
        logger.info(f"Using features: {available_features}")
        
        # Seleccionar solo características disponibles
        df_features = df_normalized[available_features].copy()
        
        # Limpiar datos faltantes
        logger.info(f"NaN values before cleaning: {df_features.isna().sum().sum()}")
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')
        
        # Si aún hay NaN, llenar con la media
        if df_features.isna().sum().sum() > 0:
            df_features = df_features.fillna(df_features.mean())
        
        logger.info(f"NaN values after cleaning: {df_features.isna().sum().sum()}")
        
        # Verificar que tenemos datos después de la limpieza
        if df_features.empty:
            logger.error("DataFrame is empty after cleaning")
            return {
                'prices': np.array([]),
                'features': np.array([]),
                'feature_names': [],
                'original_prices': np.array([])
            }
        
        # Características derivadas (solo si tenemos suficientes datos)
        if len(df_features) > 1:
            if 'close' in df_features.columns:
                df_features['Price_Change'] = df_features['close'].pct_change().fillna(0)
            
            if 'volume' in df_features.columns and len(df_features) > 1:
                df_features['Volume_Change'] = df_features['volume'].pct_change().fillna(0)
            
            if len(df_features) >= 20 and 'close' in df_features.columns:
                df_features['Volatility'] = df_features['close'].rolling(window=20).std().fillna(0)
            else:
                df_features['Volatility'] = 0
            
            if 'BB_Upper' in df_features.columns and 'BB_Lower' in df_features.columns and 'close' in df_features.columns:
                bb_range = df_features['BB_Upper'] - df_features['BB_Lower']
                bb_range = bb_range.replace(0, 1)  # Evitar división por cero
                df_features['Price_Position'] = (df_features['close'] - df_features['BB_Lower']) / bb_range
                df_features['Price_Position'] = df_features['Price_Position'].fillna(0.5)
            else:
                df_features['Price_Position'] = 0.5
        
        # Remover filas con NaN después de los cálculos
        df_features = df_features.dropna()
        
        if df_features.empty:
            logger.error("DataFrame is empty after feature engineering")
            return {
                'prices': np.array([]),
                'features': np.array([]),
                'feature_names': [],
                'original_prices': np.array([])
            }
        
        logger.info(f"Features after engineering: {df_features.shape}")
        
        # Extraer precios y características
        if 'close' in df_features.columns:
            prices = df_features['close'].values
            original_prices = prices.copy()
            
            # Separar características (todo excepto close)
            feature_columns = [col for col in df_features.columns if col != 'close']
            if feature_columns:
                features = df_features[feature_columns].values
            else:
                # Si no hay otras características, usar solo el precio
                features = prices.reshape(-1, 1)
                feature_columns = ['close']
        else:
            logger.error("No 'close' column found after processing")
            return {
                'prices': np.array([]),
                'features': np.array([]),
                'feature_names': [],
                'original_prices': np.array([])
            }
        
        logger.info(f"Prices shape: {prices.shape}")
        logger.info(f"Features shape: {features.shape}")
        
        # Verificar que tenemos datos válidos
        if len(prices) == 0 or len(features) == 0:
            logger.error("Empty prices or features arrays")
            return {
                'prices': np.array([]),
                'features': np.array([]),
                'feature_names': [],
                'original_prices': np.array([])
            }
        
        # Escalar datos
        if not self.is_fitted:
            logger.info("Fitting scalers...")
            # Escalar precios (target)
            self.price_scaler.fit(prices.reshape(-1, 1))
            
            # Escalar características
            self.feature_scaler.fit(features)
            self.is_fitted = True
        
        # Aplicar escalado
        scaled_prices = self.price_scaler.transform(prices.reshape(-1, 1)).flatten()
        scaled_features = self.feature_scaler.transform(features)
        
        logger.info(f"Scaled prices shape: {scaled_prices.shape}")
        logger.info(f"Scaled features shape: {scaled_features.shape}")
        
        return {
            'prices': scaled_prices,
            'features': scaled_features,
            'feature_names': feature_columns,
            'original_prices': original_prices
        }
    
    def create_training_data(self, prepared_data: Dict, sequence_length: int = 60) -> Dict:
        """Crea datos de entrenamiento para los modelos"""
        
        prices = prepared_data['prices']
        features = prepared_data['features']
        
        logger.info(f"Creating training data with {len(prices)} price points and {features.shape[1]} features")
        
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
        
        # Ajustar tamaños para que coincidan
        min_length = min(len(X_price_seq), len(y_direction))
        X_price_seq = X_price_seq[:min_length]
        X_features_seq = X_features_seq[:min_length]
        X_current_features = X_current_features[:min_length]
        y_price = y_price[:min_length]
        y_direction = y_direction[:min_length]
        
        logger.info(f"Training data created:")
        logger.info(f"  - Price sequences: {X_price_seq.shape}")
        logger.info(f"  - Feature sequences: {X_features_seq.shape}")
        logger.info(f"  - Current features: {X_current_features.shape}")
        logger.info(f"  - Price targets: {y_price.shape}")
        logger.info(f"  - Direction targets: {y_direction.shape}")
        
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
        """Realiza predicción ensemble con manejo de modelos faltantes"""
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
        
        predictions = []
        weights = []
        
        # Predicción LSTM
        if self.lstm_model is not None and self.lstm_model.is_trained:
            try:
                lstm_pred = self.lstm_model.predict(X_price, X_features)
                predictions.append(lstm_pred)
                weights.append(0.4)
                logger.info("LSTM prediction successful")
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
        
        # Predicción GRU
        if self.gru_model is not None and self.gru_model.is_trained:
            try:
                gru_pred = self.gru_model.predict(X_price, X_features)
                predictions.append(gru_pred)
                weights.append(0.4)
                logger.info("GRU prediction successful")
            except Exception as e:
                logger.warning(f"GRU prediction failed: {e}")
        
        # Predicción Random Forest
        if self.rf_model is not None:
            try:
                rf_pred = self.rf_model.predict(X_current)
                predictions.append(rf_pred)
                weights.append(0.2)
                logger.info("Random Forest prediction successful")
            except Exception as e:
                logger.warning(f"Random Forest prediction failed: {e}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Normalizar pesos
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        # Combinar predicciones (weighted average)
        final_price = sum(pred.price_prediction * weight 
                        for pred, weight in zip(predictions, normalized_weights))
        
        # Determinar dirección final (voto mayoritario)
        directions = [pred.direction_prediction for pred in predictions]
        direction_counts = {direction: directions.count(direction) for direction in set(directions)}
        final_direction = max(direction_counts, key=direction_counts.get)
        
        # Calcular confianza promedio ponderada
        final_confidence = sum(pred.confidence * weight 
                            for pred, weight in zip(predictions, normalized_weights))
        
        # Incorporar análisis de sentimiento
        sentiment_influence = abs(sentiment_score) * 0.3
        
        # Ajustar confianza basada en sentimiento
        if sentiment_score > 0.1 and final_direction == 'UP':
            final_confidence = min(1.0, final_confidence + sentiment_influence)
        elif sentiment_score < -0.1 and final_direction == 'DOWN':
            final_confidence = min(1.0, final_confidence + sentiment_influence)
        else:
            final_confidence = max(0.1, final_confidence - sentiment_influence)
        
        logger.info(f"Ensemble prediction: {final_direction} ({final_confidence:.2f}) using {len(predictions)} models")
        
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
            # Cargar feature engineer primero
            self.feature_engineer = joblib.load(f"{path_prefix}feature_engineer.pkl")
            
            # Cargar modelos con configuración personalizada
            import tensorflow as tf
            
            # Configurar métricas personalizadas
            custom_objects = {
                'mse': tf.keras.metrics.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError(),
                'accuracy': tf.keras.metrics.Accuracy()
            }
            
            # Cargar LSTM
            self.lstm_model = LSTMModel()
            try:
                self.lstm_model.model = tf.keras.models.load_model(
                    f"{path_prefix}lstm_model.h5", 
                    custom_objects=custom_objects,
                    compile=False  # No compilar automáticamente
                )
                # Re-compilar con métricas conocidas
                self.lstm_model.model.compile(
                    optimizer='adam',
                    loss={'price_output': 'mse', 'direction_output': 'sparse_categorical_crossentropy'},
                    metrics={'price_output': ['mae'], 'direction_output': ['accuracy']}
                )
                self.lstm_model.is_trained = True
                logger.info("LSTM model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load LSTM model: {e}")
                self.lstm_model = None
            
            # Cargar GRU
            self.gru_model = GRUModel()
            try:
                self.gru_model.model = tf.keras.models.load_model(
                    f"{path_prefix}gru_model.h5",
                    custom_objects=custom_objects,
                    compile=False
                )
                # Re-compilar
                self.gru_model.model.compile(
                    optimizer='adam',
                    loss={'price_output': 'mse', 'direction_output': 'sparse_categorical_crossentropy'},
                    metrics={'price_output': ['mae'], 'direction_output': ['accuracy']}
                )
                self.gru_model.is_trained = True
                logger.info("GRU model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load GRU model: {e}")
                self.gru_model = None
            
            # Cargar Random Forest
            try:
                self.rf_model = joblib.load(f"{path_prefix}rf_model.pkl")
                logger.info("Random Forest model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Random Forest model: {e}")
                self.rf_model = None
            
            # Verificar que al menos un modelo se cargó
            loaded_models = sum([
                self.lstm_model is not None and self.lstm_model.is_trained,
                self.gru_model is not None and self.gru_model.is_trained,
                self.rf_model is not None
            ])
            
            if loaded_models > 0:
                self.is_trained = True
                logger.info(f"Ensemble loaded successfully with {loaded_models}/3 models")
            else:
                self.is_trained = False
                logger.error("No models could be loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_trained = False

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
