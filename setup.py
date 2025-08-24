#!/usr/bin/env python3
"""
Script de configuración y verificación del entorno para el proyecto de Sudoku con Lógica Difusa.

Este script verifica que todas las dependencias estén instaladas correctamente y
proporciona una demostración básica de las funcionalidades principales.
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Verifica que la versión de Python sea compatible."""
    print("🐍 Verificando versión de Python...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior.")
        print(f"   Versión actual: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_module(module_name, package_name=None):
    """Verifica si un módulo está instalado."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError:
        print(f"❌ {package_name or module_name} no está instalado")
        return False

def check_dependencies():
    """Verifica todas las dependencias requeridas."""
    print("\n📦 Verificando dependencias...")
    
    dependencies = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("sklearn", "scikit-learn"),
    ]
    
    all_ok = True
    for module, package in dependencies:
        if not check_module(module, package):
            all_ok = False
    
    return all_ok

def check_project_files():
    """Verifica que los archivos del proyecto estén presentes."""
    print("\n📁 Verificando archivos del proyecto...")
    
    required_files = [
        "fuzzy_edge_detection.py",
        "fuzzy_sudoku_solver.py", 
        "tinyfuzzy.py",
        "requirements.txt"
    ]
    
    all_ok = True
    current_dir = Path(".")
    
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} no encontrado")
            all_ok = False
    
    return all_ok

def install_dependencies():
    """Instala las dependencias desde requirements.txt."""
    print("\n🔧 Instalando dependencias...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al instalar dependencias: {e}")
        return False

def run_basic_test():
    """Ejecuta una prueba básica de las funcionalidades."""
    print("\n🧪 Ejecutando prueba básica...")
    
    try:
        # Importar módulos locales
        import tinyfuzzy
        import numpy as np
        
        # Prueba básica del motor de lógica difusa
        x = np.linspace(0, 10, 100)
        triangular = tinyfuzzy.trimf(x, 2, 5, 8)
        
        if len(triangular) == 100:
            print("✅ Motor de lógica difusa funcionando")
        else:
            print("❌ Problema con el motor de lógica difusa")
            return False
        
        # Verificar OpenCV
        import cv2
        print(f"✅ OpenCV versión {cv2.__version__}")
        
        # Verificar que las imágenes de ejemplo existan
        if Path("peppers.png").exists():
            print("✅ Imagen de ejemplo peppers.png encontrada")
        else:
            print("⚠️  Imagen peppers.png no encontrada (opcional)")
        
        if Path("sudoku.jpg").exists():
            print("✅ Imagen de ejemplo sudoku.jpg encontrada")
        else:
            print("⚠️  Imagen sudoku.jpg no encontrada (opcional)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en la prueba básica: {e}")
        return False

def main():
    """Función principal del script de configuración."""
    print("🚀 Configuración del Proyecto de Sudoku con Lógica Difusa")
    print("=" * 60)
    
    # Verificar Python
    if not check_python_version():
        print("\n❌ Configuración fallida. Actualiza Python a versión 3.8+")
        return False
    
    # Verificar archivos del proyecto
    if not check_project_files():
        print("\n❌ Archivos del proyecto faltantes")
        return False
    
    # Verificar dependencias
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n🔧 Instalando dependencias faltantes...")
        if not install_dependencies():
            print("\n❌ No se pudieron instalar las dependencias")
            return False
        
        # Verificar nuevamente después de la instalación
        if not check_dependencies():
            print("\n❌ Algunas dependencias aún faltan después de la instalación")
            return False
    
    # Ejecutar prueba básica
    if not run_basic_test():
        print("\n❌ Las pruebas básicas fallaron")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ¡Configuración completada exitosamente!")
    print("\nEl proyecto está listo para usar. Puedes:")
    print("1. Ejecutar 'python fuzzy_edge_detection.py' para detección de bordes")
    print("2. Ejecutar 'python fuzzy_sudoku_solver.py' para resolver Sudoku")
    print("3. Importar los módulos en tus propios scripts")
    print("\nConsulta el README.md para más información.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
