# Proyecto de Detección de Sudoku con Lógica Difusa

Este proyecto implementa un sistema de detección y resolución de Sudoku utilizando lógica difusa para la detección de bordes.

## ¿Qué hace?

- **Detección de bordes difusa**: Usa lógica difusa para encontrar bordes en imágenes
- **Resolución de Sudoku**: Detecta automáticamente un tablero de Sudoku en una foto y lo resuelve
- **Reconocimiento de dígitos**: Identifica números usando machine learning

## Archivos principales

- `fuzzy_edge_detection.py` - Detección de bordes con lógica difusa
- `fuzzy_sudoku_solver.py` - Solucionador completo de Sudoku  
- `tinyfuzzy.py` - Motor de lógica difusa
- `peppers.png` y `sudoku.jpg` - Imágenes de ejemplo

## Instalación

### 1. Clona el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_DIRECTORIO>
```

### 2. Crea un entorno virtual
```bash
# Crear entorno virtual
python -m venv venv

# Activar en Windows:
venv\Scripts\activate

# Activar en macOS/Linux:
source venv/bin/activate
```

### 3. Instala las dependencias
```bash
pip install -r requirements.txt
```

## Uso básico

### Detección de bordes
```python
python fuzzy_edge_detection.py
```

### Resolver Sudoku
```python
python fuzzy_sudoku_solver.py
```

## Dependencias

- `opencv-python` - Procesamiento de imágenes
- `numpy` - Operaciones numéricas
- `matplotlib` - Visualización
- `scikit-learn` - Machine learning

## Problemas comunes

**Error: "No module named 'cv2'"**
```bash
pip install opencv-python
```

**Error: "No module named 'sklearn'"**
```bash
pip install scikit-learn
```

**¿El Sudoku no se detecta bien?**
- Asegúrate de que la imagen tenga buena iluminación
- El tablero debe estar bien visible y contrastado

## Para usar en otro equipo

1. Instala Python 3.8+ 
2. Clona este repositorio
3. Crea el entorno virtual y actívalo
4. Instala las dependencias con `pip install -r requirements.txt`
5. ¡Listo para usar!
