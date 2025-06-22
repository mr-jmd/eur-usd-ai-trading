# EUR-USD-AI-TRADING

Sistema de Trading EUR/USD con Inteligencia Artificial

## 📖 Descripción

Sistema inteligente para trading del par EUR/USD que utiliza:
- Machine Learning (LSTM, GRU, Random Forest)
- Análisis de sentimiento con BERT
- Backtesting automatizado
- Dashboard interactivo en tiempo real

**Desarrollado para:** Instituto Tecnológico Metropolitano  
**Curso:** Seminario de Investigación  
**Año:** 2025

## 🚀 Inicio Rápido

```bash
# 1. Clonar/descargar proyecto
git clone <repository-url>
cd eur-usd-ai-trading

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar APIs
cp .env.template .env
nano .env

# 4. Entrenar modelos
python scripts/setup/train_models.py

# 5. Iniciar sistema
python src/main.py

# 6. Abrir dashboard
streamlit run src/dashboard/app.py
```

## 📁 Estructura del Proyecto

```
eur-usd-ai-trading/
├── src/                    # Código fuente
│   ├── data_collection/    # Recolección de datos
│   ├── models/            # Modelos de ML
│   ├── sentiment/         # Análisis de sentimiento
│   ├── backtesting/       # Sistema de backtesting
│   └── dashboard/         # Interfaz de usuario
├── data/                  # Datos del proyecto
├── models/               # Modelos entrenados
├── config/               # Configuraciones
├── logs/                 # Logs del sistema
├── tests/                # Pruebas
├── docs/                 # Documentación
└── notebooks/            # Jupyter notebooks
```

## 📊 Características

- ✅ Predicción con >65% de precisión
- ✅ Análisis de sentimiento en tiempo real
- ✅ Gestión automática de riesgo
- ✅ Dashboard interactivo
- ✅ Backtesting completo
- ✅ Alertas por email
- ✅ Trading automático opcional

## 📚 Documentación

- [Guía de Instalación](docs/installation.md)
- [Configuración](docs/configuration.md)
- [Guía de Usuario](docs/user_guide.md)
- [API Reference](docs/api/)
- [Modelos de ML](docs/models/)

## ⚠️ Advertencias

- **Riesgo financiero:** El trading implica riesgo de pérdida
- **Uso educativo:** Sistema para fines de investigación
- **No garantías:** Resultados pasados no garantizan futuros

## 📞 Contacto

**Autores:**
- Juan Manuel Amaya Cadavid
- Julio Cesar Jiménez García

**Supervisor:**
- Laura Stella Vega Escobar

**Institución:**
Instituto Tecnológico Metropolitano
