# Hotel Booking Demand - Predicción de Cancelaciones

## 📌 Descripción del Proyecto
Análisis Exploratorio de Datos (EDA) completo y modelo predictivo para el dataset "Hotel Booking Demand" que contiene 32 variables de 119,390 reservas hoteleras. El objetivo es identificar patrones predictivos de cancelaciones y optimizar la gestión de revenue.

## 🗃️ Dataset
- **Fuente**: Kaggle - Hotel Booking Demand
- **Registros**: 119,390 reservas hoteleras
- **Período**: 2015-2017
- **Variables**: 32 características originales
- **Target**: `is_canceled` (37% tasa de cancelación)

## 🧹 Limpieza de Datos
Se prepararon los datos verificando y tratando valores faltantes y duplicados:

### Tratamiento de Datos Faltantes
```python
# Estrategias de imputación aplicadas
- 'company': ELIMINADA (94.3% faltantes)
- 'agent': Imputación con mediana + flag
- 'country': Imputación con moda
- 'children': Imputación con mediana
```

### Resultado de Limpieza
- **Datos originales**: (119390, 32)
- **Datos limpios**: (119390, 31)
- **Valores nulos restantes**: 0%

## 🔍 Hallazgos Clave del EDA

### 1. Desbalance de Clases
- **Ratio**: 63% reservas cumplidas vs 37% canceladas
- **Problema**: Clasificación moderadamente desbalanceada
- **Solución aplicada**: Estrategias de métricas (F1-Score, Recall)

### 2. Variables Críticas Identificadas
**8 variables más predictivas:**
```python
['lead_time', 'adr', 'adults', 'previous_cancellations',
 'hotel', 'deposit_type', 'customer_type', 'market_segment']
```

### 3. Tratamiento de Outliers
**Variables con distribuciones sesgadas:**
- `lead_time`: 0-737 días (skewness: 1.35)
- `adr`: -6.4 a $5,400 (skewness: 10.53)
- `adults`: 0-55 personas (skewness: 18.32)

**Solución aplicada:** RobustScaler para todas las variables numéricas

### 4. Ingeniería de Features
**Técnicas aplicadas:**
- Codificación: One-Hot Encoding para categóricas
- Escalado: RobustScaler para numéricas
- Reducción: De 31 a 21 features finales

## 🛠️ Pipeline de Preprocesamiento

### Transformaciones Implementadas
```python
# Pipeline final
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
```

### Resultado del Pipeline
```
✅ Transformación exitosa: (119390, 31) → (119390, 21)
📈 Features generadas: 21 listas para modelado
```

## 📊 Modelado Predictivo

### Algoritmos Implementados
- Random Forest Classifier
- Gradient Boosting
- Logistic Regression
- XGBoost

### Métricas de Evaluación
- **F1-Score**: 0.85
- **Recall**: 0.82
- **Precision**: 0.88
- **ROC-AUC**: 0.91

## 🎯 Variables Más Importantes
**Top 5 variables predictivas:**
1. `lead_time` - Tiempo de anticipación
2. `deposit_type` - Tipo de depósito
3. `adr` - Tarifa diaria promedio
4. `previous_cancellations` - Historial de cancelaciones
5. `customer_type` - Tipo de cliente

## 🚀 Instalación y Uso

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar análisis completo
python run_analysis.py
```

## 📁 Estructura del Proyecto
```
hotel-booking-analysis/
├── data/
│   └── hotel_bookings.csv
├── notebooks/
│   ├── 01_eda.ipynb          # Análisis exploratorio
│   ├── 02_preprocessing.ipynb # Pipeline de preprocesamiento
│   └── 03_modeling.ipynb     # Modelado predictivo
├── src/
│   ├── preprocessing.py      # Clases de preprocesamiento
│   ├── modeling.py          # Entrenamiento de modelos
│   └── utils.py             # Funciones auxiliares
├── requirements.txt
└── README.md
```

## 📈 Resultados y Conclusiones
- **Modelo optimizado**: Random Forest con 85% de F1-Score
- **Patrones identificados**: 
  - Reservas con >100 días de anticipación tienen 3x más probabilidad de cancelar
  - Clientes sin depósito cancelan 5x más frecuentemente
  - Segmento "Transient" representa el 75% de cancelaciones

