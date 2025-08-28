# Proyecto de Detección de Sudoku con Lógica Difusa

Este proyecto implementa un sistema de detección y resolución de Sudoku utilizando lógica difusa para la detección de bordes.

## ¿Qué hace?

- **Detección de bordes difusa**: Usa lógica difusa para encontrar bordes en imágenes
- **Resolución de Sudoku**: Detecta automáticamente un tablero de Sudoku en una foto y lo resuelve
- **Reconocimiento de dígitos**: Identifica números usando machine learning ( otras formas se estan revisando )

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

## Cómo usar los programas

### 1. Detección de bordes difusa
Detecta bordes en cualquier imagen usando lógica difusa. Ejemplo:

**Para ver los gráficos de forma interactiva:**
```bash
python fuzzy_edge_detection.py --image peppers.png
```

**Para guardar los resultados sin mostrarlos:**
```bash
python fuzzy_edge_detection.py --image peppers.png --save_dir ./test/ --samples 200
```

**Para ver y guardar los resultados:**
```bash
python fuzzy_edge_detection.py --image peppers.png --save_dir ./test/ --show
```

Parámetros útiles:
- `--image <archivo>`: imagen de entrada
- `--save_dir <carpeta>`: carpeta donde se guardan los resultados
- `--show`: mostrar gráficos de forma interactiva
- `--samples <n>`: cantidad de muestras para graficar funciones
- `--no_plots`: omitir la generación de gráficos

### 2. Resolución automática de Sudoku
Detecta y resuelve un Sudoku desde una foto. Ejemplo:
```bash
python fuzzy_sudoku_solver.py --image sudoku.jpg --save_dir ./test/
```
Parámetros útiles:
- `--image <archivo>`: imagen del Sudoku
- `--save_dir <carpeta>`: carpeta donde se guardan los resultados

Ambos scripts muestran imágenes de diagnóstico y guardan los resultados en la carpeta indicada.

## ¿Por qué se usa machine learning?
Se utiliza machine learning para reconocer los números escritos en el tablero de Sudoku. Por ahora, es una forma sencilla de intentar capturar los dígitos automáticamente a partir de la imagen, usando un clasificador básico. Esto permite que el sistema resuelva el Sudoku sin intervención manual.

## Dependencias

- `opencv-python` - Procesamiento de imágenes
- `numpy` - Operaciones numéricas
- `matplotlib` - Visualización

## Problemas comunes

**Error: "No module named 'cv2'"**
```bash
pip install opencv-python
```

## Para usar en otro equipo

1. Instala Python 3.8+ 
2. Clona este repositorio
3. Crea el entorno virtual y actívalo
4. Instala las dependencias con `pip install -r requirements.txt`
