# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir al **EUR/USD AI Trading System**! 🎉

Este documento te guiará a través del proceso de contribución para que puedas ayudar a mejorar el proyecto de manera efectiva.

## 📋 Tabla de Contenidos

- [🎯 Tipos de Contribuciones](#-tipos-de-contribuciones)
- [🚀 Inicio Rápido](#-inicio-rápido)
- [🔄 Proceso de Contribución](#-proceso-de-contribución)
- [📝 Estándares de Código](#-estándares-de-código)
- [🧪 Testing](#-testing)
- [📚 Documentación](#-documentación)
- [🐛 Reportar Bugs](#-reportar-bugs)
- [💡 Sugerir Features](#-sugerir-features)
- [📞 Comunicación](#-comunicación)

---

## 🎯 Tipos de Contribuciones

Valoramos todos los tipos de contribuciones:

### 🐛 **Bug Fixes**
- Corrección de errores en el código
- Mejoras en el manejo de excepciones
- Optimización de rendimiento

### ✨ **Nuevas Features**
- Nuevos modelos de ML
- Indicadores técnicos adicionales
- Mejoras en el dashboard
- Integraciones con nuevas APIs

### 📚 **Documentación**
- Mejorar README y guías
- Agregar ejemplos de código
- Traducir documentación
- Crear tutoriales

### 🧪 **Testing**
- Agregar tests unitarios
- Mejorar cobertura de tests
- Tests de integración
- Tests de performance

### 🎨 **UI/UX**
- Mejoras en el dashboard
- Nuevas visualizaciones
- Mejor experiencia de usuario
- Diseño responsivo

---

## 🚀 Inicio Rápido

### 1️⃣ **Fork y Clone**

```bash
# Fork el repositorio en GitHub, luego:
git clone https://github.com/tu-usuario/eur-usd-ai-trading.git
cd eur-usd-ai-trading
```

### 2️⃣ **Setup del Entorno**

```bash
# Crear entorno virtual
python -m venv venv-dev
source venv-dev/bin/activate  # Linux/Mac
# venv-dev\Scripts\activate   # Windows

# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Instalar pre-commit hooks
pre-commit install
```

### 3️⃣ **Verificar Setup**

```bash
# Ejecutar tests
pytest tests/ -v

# Verificar linting
flake8 src/
black --check src/
isort --check-only src/

# Verificar tipos
mypy src/
```

### 4️⃣ **Crear Branch**

```bash
# Para nueva feature
git checkout -b feature/mi-nueva-feature

# Para bug fix
git checkout -b fix/descripcion-del-bug

# Para documentación
git checkout -b docs/actualizar-readme
```

---

## 🔄 Proceso de Contribución

### 📝 **1. Planificación**

1. **Revisar Issues**: Busca issues existentes relacionados
2. **Crear Issue**: Si no existe, crea uno nuevo describiendo tu propuesta
3. **Discutir**: Comenta en el issue para coordinar con otros contributors
4. **Asignación**: Solicita ser asignado al issue

### 💻 **2. Desarrollo**

1. **Crear Branch**: Desde `develop` (no desde `main`)
2. **Desarrollar**: Implementa tu cambio siguiendo los estándares
3. **Tests**: Agregar tests para tu código
4. **Commits**: Usar commits descriptivos ([ver convenciones](#convenciones-de-commits))

### ✅ **3. Testing y Quality**

```bash
# Ejecutar tests completos
pytest tests/ -v --cov=src

# Verificar calidad de código
flake8 src/
black src/
isort src/
mypy src/

# Tests de integración
pytest tests/integration/ -v

# Performance tests (si aplica)
pytest tests/performance/ -v --benchmark-only
```

### 📤 **4. Pull Request**

1. **Push Branch**: `git push origin feature/mi-nueva-feature`
2. **Crear PR**: Desde tu fork hacia `develop` (no `main`)
3. **Completar Template**: Usar el template de PR
4. **Revisar CI**: Asegurar que pase todos los checks

### 🔍 **5. Code Review**

- Responde a comentarios constructivamente
- Realiza cambios solicitados
- Mantén la conversación profesional y amigable

---

## 📝 Estándares de Código

### 🐍 **Python Style Guide**

Seguimos **PEP 8** con algunas excepciones:

```python
# ✅ Correcto
def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calcula el Relative Strength Index.
    
    Args:
        prices: Lista de precios
        period: Período para el cálculo (default: 14)
        
    Returns:
        Valor RSI entre 0 y 100
        
    Raises:
        ValueError: Si el período es menor a 2
    """
    if period < 2:
        raise ValueError("El período debe ser mayor a 1")
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ❌ Incorrecto
def calc_rsi(p,per=14):
    d=np.diff(p)
    g=np.where(d>0,d,0)
    l=np.where(d<0,-d,0)
    ag=np.mean(g[-per:])
    al=np.mean(l[-per:])
    rs=ag/al
    return 100-(100/(1+rs))
```

### 📏 **Configuración de Herramientas**

```ini
# setup.cfg
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = venv/, .git/, __pycache__/

[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True

[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

### 🔧 **Pre-commit Configuration**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
        language_version: python3.8

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
