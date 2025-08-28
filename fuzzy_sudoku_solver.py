"""
Sudoku solver using fuzzy logic edge detection.

This script builds upon the fuzzy edge detection implemented in
``fuzzy_edge_detection.py`` to locate a Sudoku grid in a noisy
photograph, extract and normalise the grid, recognise the digits in
each cell and solve the puzzle.  The pipeline is roughly as follows:

1. **Edge detection** – load the photograph, convert it to grayscale and
   use the fuzzy logic algorithm (Gaussian membership functions on
   gradient magnitude and a simple rule base) to find edges in the
   image.  The resulting edge map highlights strong line segments and
   suppresses background noise.

2. **Board detection** – threshold and dilate the edge map to obtain a
   binary image.  Contours are extracted and the largest roughly
   rectangular contour is assumed to correspond to a Sudoku grid.  The
   four corners of this contour are ordered consistently and a
   perspective transform warps the board to a square of fixed size.

3. **Digit recognition** – the warped grid is divided into a 9×9 array
   of cells.  For each cell, connected component analysis and simple
   heuristics determine whether the cell contains a digit.  When a
   digit is found, it is centred on a blank square, resized to 8×8 and
   classified using a k‑nearest neighbours classifier trained on the
   ``sklearn`` digits dataset (which contains 8×8 images of the digits
   0–9).

4. **Puzzle solving** – the recognised digits form a 9×9 matrix with
   zeros representing empty cells.  A backtracking solver fills in the
   missing numbers subject to Sudoku constraints.  If a solution is
   found, it is returned and optionally drawn on the rectified grid.

This script outputs several images for inspection: the detected board
overlayed on the input, the rectified board, the recognised digits and
the solved puzzle.  It can be invoked from the command line; see
``python fuzzy_sudoku_solver.py --help`` for details.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tinyfuzzy import gaussmf, trimf

# Import helpers from our previous module
from fuzzy_edge_detection import (
    load_image_grayscale,
    compute_gradients,
    fuzzy_edge_detection,
)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as top‑left, top‑right, bottom‑right, bottom‑left.

    Parameters
    ----------
    pts : ndarray of shape (4, 2)
        Four unordered corner points.

    Returns
    -------
    ordered : ndarray of shape (4, 2)
        Points sorted in clockwise order starting from top‑left.
    """
    # Compute sums and differences to locate corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def find_sudoku_contour(edge_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Locate a plausible Sudoku grid in a binary edge image.

    This function searches for quadrilateral contours whose aspect ratio is
    roughly square and whose area is significant relative to the image.
    By iterating through all contours ordered by area, it avoids
    mistakenly selecting the page boundary or other spurious rectangles.

    Parameters
    ----------
    edge_img : ndarray of shape (H, W)
        Binary image (0 or 255) obtained by thresholding the edge map.

    Returns
    -------
    approx : ndarray of shape (4, 2)
        Approximated contour with 4 vertices corresponding to the Sudoku grid.
    contour : ndarray
        The full contour points.

    Raises
    ------
    RuntimeError
        If no suitable contour is found.
    """
    # Find external contours
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = edge_img.shape[0] * edge_img.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        # Skip very small contours (<2% of image area)
        if area < 0.02 * img_area:
            continue
        # Approximate contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        # Compute bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        # Accept near‑square boxes (aspect ratio between 0.8 and 1.2)
        if 0.8 <= ratio <= 1.2:
            return approx.reshape(4, 2), c
    raise RuntimeError("No suitable Sudoku contour found.")


def find_sudoku_contour_morphological(gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detect a Sudoku grid using morphological line extraction.

    This function performs adaptive thresholding on the input grayscale image
    followed by morphological operations to isolate horizontal and
    vertical lines.  The combined grid lines are then passed to
    ``find_sudoku_contour`` to locate the board.

    Parameters
    ----------
    gray_img : ndarray of shape (H, W)
        Grayscale image scaled in 0–255 range (dtype uint8).

    Returns
    -------
    approx : ndarray of shape (4, 2)
        Detected corner points of the Sudoku grid.
    contour : ndarray
        Full contour points.

    Raises
    ------
    RuntimeError
        If no suitable contour can be found.
    """
    # Adaptive threshold to binary (invert so grid lines become white)
    thr = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    # Extract horizontal lines
    horizontal = thr.copy()
    h_kernel_len = max(10, gray_img.shape[1] // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_temp = cv2.erode(horizontal, h_kernel, iterations=1)
    h_lines = cv2.dilate(h_temp, h_kernel, iterations=1)
    # Extract vertical lines
    vertical = thr.copy()
    v_kernel_len = max(10, gray_img.shape[0] // 30)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_temp = cv2.erode(vertical, v_kernel, iterations=1)
    v_lines = cv2.dilate(v_temp, v_kernel, iterations=1)
    # Combine horizontal and vertical lines
    grid = cv2.bitwise_or(h_lines, v_lines)
    # Optionally close gaps
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return find_sudoku_contour(grid)


def warp_sudoku(
    gray_img: np.ndarray,
    corner_points: np.ndarray,
    size: int = 450,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp the Sudoku grid to a square image.

    Parameters
    ----------
    gray_img : ndarray
        Grayscale source image.
    corner_points : ndarray of shape (4, 2)
        Detected corner points of the Sudoku grid, ordered arbitrarily.
    size : int, optional
        Size (in pixels) of the output square; default is 450×450.

    Returns
    -------
    warped : ndarray of shape (size, size)
        Rectified grayscale image of the Sudoku board.
    M : ndarray of shape (3, 3)
        Homography matrix used for warping.
    """
    rect = order_points(corner_points)
    (tl, tr, br, bl) = rect
    # Compute width and height of the new image based on distances between points
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    # Destination points for the warped image
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1],
    ], dtype="float32")
    # Compute perspective transform and apply
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray_img, M, (size, size))
    return warped, M




def extract_cell_digit(cell: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Extract the digit from a Sudoku cell if present.

    Parameters
    ----------
    cell : ndarray of shape (h, w)
        Grayscale cell image cropped from the warped Sudoku board.

    Returns
    -------
    digit_img : ndarray of shape (8, 8)
        Resized 8×8 image of the digit suitable for the classifier.  If the
        cell is empty, returns a zero array.
    has_digit : bool
        True if a digit was detected in the cell, False otherwise.
    """
    # Reduce influence of grid lines by cropping an inner region of the cell.
    h, w = cell.shape
    margin = int(min(h, w) * 0.15)
    if margin > 0:
        inner = cell[margin:h - margin, margin:w - margin]
    else:
        inner = cell.copy()
    # Blur and threshold the inner region; invert so digits become white on black
    inner_blur = cv2.GaussianBlur(inner, (3, 3), 0)
    _, thresh = cv2.threshold(inner_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # If the number of white pixels is very small, treat as empty
    # If the amount of white pixels is small, the cell is likely empty.
    if cv2.countNonZero(thresh) < 0.07 * thresh.size:
        return np.zeros((8, 8), dtype=np.float32), False
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((8, 8), dtype=np.float32), False
    # Largest contour
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    # Ignore very small contours which are likely noise or grid remnants
    if area < 0.05 * thresh.shape[0] * thresh.shape[1]:
        return np.zeros((8, 8), dtype=np.float32), False
    x, y, w_box, h_box = cv2.boundingRect(c)
    digit = thresh[y:y + h_box, x:x + w_box]
    # Resize to 8×8 with aspect ratio preserved (pad if necessary)
    # Compute aspect ratio
    h_d, w_d = digit.shape
    # Create square canvas
    size = max(h_d, w_d)
    square = np.zeros((size, size), dtype=np.uint8)
    # Center digit on square canvas
    y_off = (size - h_d) // 2
    x_off = (size - w_d) // 2
    square[y_off:y_off + h_d, x_off:x_off + w_d] = digit
    digit_resized = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)
    digit_normalised = digit_resized.astype(np.float32) / 255.0
    return digit_normalised, True


def recognise_digits(board: np.ndarray) -> np.ndarray:
    grid = np.zeros((9, 9), dtype=int)
    cell_size = board.shape[0] // 9
    for i in range(9):
        for j in range(9):
            y = i * cell_size
            x = j * cell_size
            cell = board[y:y + cell_size, x:x + cell_size]
            digit_img, has_digit = extract_cell_digit(cell)
            if has_digit:
                grid[i, j] = fuzzy_digit_identifier(digit_img)
    return grid

def fuzzy_digit_identifier(digit_img: np.ndarray) -> int:
    import cv2
    digit_bin = (digit_img > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(digit_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    metrics = {
        "vertical": np.sum(digit_img[:, 3:5]),
        "horizontal": np.sum(digit_img[3:5, :]),
        "left_area": np.sum(digit_img[:, :2]),
        "right_area": np.sum(digit_img[:, 6:]),
        "top_area": np.sum(digit_img[:2, :]),
        "bottom_area": np.sum(digit_img[6:, :]),
        "center_area": np.sum(digit_img[2:6, 2:6]),
        "percent_white": np.sum(digit_img > 0.5) / digit_img.size,
        "num_components": len(contours),
        "vert_sym": np.sum(np.abs(digit_img - np.flip(digit_img, axis=1))),
        "hor_sym": np.sum(np.abs(digit_img - np.flip(digit_img, axis=0))),
    }

    # Imprimir métricas para depuración
    print(f"Métricas celda: {metrics}")

    # 1. Filtrar saturados
    if metrics["percent_white"] < 0.10 or metrics["percent_white"] > 0.95:
        print(f"Detectado: vacío | {metrics}")
        return 0

    # 2. Separar por componentes (más permisivo)
    if metrics["num_components"] >= 2 and metrics["center_area"] > 10 and metrics["vert_sym"] < 15 and metrics["hor_sym"] < 15:
        print(f"Detectado: 8 | {metrics}")
        return 8

    # 3. Reglas afinadas para dígitos 1, 3 y 9
    # --- 1 ---
    # Alta verticalidad, extremos vacíos, simetría vertical baja
    if metrics["vertical"] > 15 and metrics["left_area"] < 5 and metrics["right_area"] < 5 and metrics["vert_sym"] < 8:
        print(f"Detectado: 1 | {metrics}")
        return 1

    # --- 3 ---
    # Más blanco a la derecha, centro medio, simetría vertical moderada
    if metrics["right_area"] > metrics["left_area"] + 2 and metrics["center_area"] > 7 and metrics["vert_sym"] > 8 and metrics["top_area"] > 5 and metrics["bottom_area"] > 5:
        print(f"Detectado: 3 | {metrics}")
        return 3

    # --- 9 ---
    # Centro lleno, más blanco arriba, simetría vertical alta
    if metrics["center_area"] > 10 and metrics["top_area"] > metrics["bottom_area"] and metrics["bottom_area"] < 7 and metrics["vert_sym"] > 12:
        print(f"Detectado: 9 | {metrics}")
        return 9

    # --- 7 ---
    if metrics["top_area"] > 6 and metrics["center_area"] < 7 and metrics["bottom_area"] < 7:
        print(f"Detectado: 7 | {metrics}")
        return 7
    # --- 5 ---
    if metrics["horizontal"] > 6 and metrics["right_area"] > metrics["left_area"] and metrics["top_area"] < 7:
        print(f"Detectado: 5 | {metrics}")
        return 5

    # 4. Reglas para dígitos ambiguos (más permisivo)
    if digit_img[0,0] > 0.4 and digit_img[7,7] > 0.4 and metrics["vert_sym"] < 15 and metrics["hor_sym"] < 15 and metrics["center_area"] < 10:
        print(f"Detectado: 4 | {metrics}")
        return 4
    if metrics["horizontal"] > 6 and metrics["left_area"] > metrics["right_area"] and metrics["top_area"] < 7:
        print(f"Detectado: 2 | {metrics}")
        return 2
    if metrics["center_area"] > 8 and metrics["bottom_area"] > metrics["top_area"] and metrics["top_area"] < 7:
        print(f"Detectado: 6 | {metrics}")
        return 6

    # 0: simétrico, centro y extremos llenos (más permisivo)
    if metrics["vert_sym"] < 10 and metrics["hor_sym"] < 10 and metrics["center_area"] > 10 and metrics["left_area"] > 8 and metrics["right_area"] > 8:
        print(f"Detectado: 0 | {metrics}")
        return 0

    # 5. Fallback visual
    print(f"Detectado: no identificado | {metrics}")
    return 0


def is_valid(board: np.ndarray, row: int, col: int, num: int) -> bool:
    """Check if placing ``num`` at position (row, col) is valid for Sudoku."""
    # Check row and column
    if num in board[row, :]:
        return False
    if num in board[:, col]:
        return False
    # Check 3x3 box
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    if num in board[box_row:box_row + 3, box_col:box_col + 3]:
        return False
    return True


def solve_sudoku(board: np.ndarray, verbose: bool = False) -> bool:
    """Backtracking Sudoku solver that modifies the board in place.

    Returns True if a solution is found, False otherwise.
    """
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i, j] = num
                        if verbose:
                            print(f"Probando {num} en posición ({i+1}, {j+1})")
                        if solve_sudoku(board, verbose):
                            return True
                        if verbose:
                            print(f"Retrocediendo: removiendo {num} de ({i+1}, {j+1})")
                        board[i, j] = 0
                return False
    return True


def draw_solution(board_img: np.ndarray, solution: np.ndarray) -> np.ndarray:
    """Draw the solved Sudoku digits onto the rectified board image.

    Parameters
    ----------
    board_img : ndarray of shape (450, 450)
        Warped grayscale image of the Sudoku board.
    solution : ndarray of shape (9, 9)
        Solved Sudoku grid.

    Returns
    -------
    out_img : ndarray of shape (450, 450, 3)
        Colour image with the solution drawn.
    """
    out = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    cell_size = board_img.shape[0] // 9
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(9):
        for j in range(9):
            num = solution[i, j]
            # Skip drawing original clues (they will be drawn later)
            # We'll draw all numbers for clarity
            text = str(int(num))
            # Compute position centred in the cell
            x = j * cell_size + cell_size // 2
            y = i * cell_size + cell_size // 2
            # Put text centred
            # Determine text size
            (w, h), _ = cv2.getTextSize(text, font, 0.7, 2)
            cv2.putText(out, text, (x - w // 2, y + h // 2), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def print_sudoku_grid(grid: np.ndarray, title: str = ""):
    """Imprime una cuadrícula de Sudoku de forma visual."""
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    
    print("┌───────┬───────┬───────┐")
    for i in range(9):
        if i == 3 or i == 6:
            print("├───────┼───────┼───────┤")
        print("│", end="")
        for j in range(9):
            if j == 3 or j == 6:
                print("│", end="")
            if grid[i, j] == 0:
                print(" . ", end="")
            else:
                print(f" {grid[i, j]} ", end="")
        print("│")
    print("└───────┴───────┴───────┘")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and solve Sudoku using fuzzy logic edge detection.")
    parser.add_argument("--image", type=str, default="sudoku.jpg", help="Path to the Sudoku photograph")
    parser.add_argument("--sx", type=float, default=0.1, help="Sigma for Ix zero membership")
    parser.add_argument("--sy", type=float, default=0.1, help="Sigma for Iy zero membership")
    parser.add_argument("--thresh", type=float, default=0.3, help="Threshold for edge map binarisation (0–1)")
    parser.add_argument("--samples", type=int, default=31, help="Number of samples for fuzzy defuzzification")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save intermediate results")
    parser.add_argument("--verbose", action="store_true", help="Show detailed solving steps")
    args = parser.parse_args()

    print("Iniciando detección y resolución de Sudoku...")
    print(f"Cargando imagen: {args.image}")

    # Load and preprocess image
    gray = load_image_grayscale(args.image)
    print("Imagen cargada y convertida a escala de grises")
    
    print("Detectando bordes con lógica difusa...")
    Ix, Iy = compute_gradients(gray)
    edge_map = fuzzy_edge_detection(
        Ix, Iy,
        sx=args.sx,
        sy=args.sy,
        wa=0.1,
        wb=1.0,
        wc=1.0,
        ba=0.0,
        bb=0.0,
        bc=0.7,
        sample_points=args.samples,
        dtype=np.float32,
    )
    print("Detección de bordes completada")
    
    # Normalise edge map to 8‑bit and apply threshold.  Since edges appear dark
    # on a lighter background, we invert the threshold so that edge pixels
    # become white (255) and background becomes black (0).
    edge_norm = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_bin = cv2.threshold(edge_norm, int(args.thresh * 255), 255, cv2.THRESH_BINARY_INV)
    # Use morphological closing to connect broken grid lines and fill small gaps
    kernel = np.ones((7, 7), np.uint8)
    edge_bin = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, kernel)

    # Find Sudoku contour using fuzzy edge map.  If this fails, fall back to
    # adaptive threshold and finally to morphological line extraction.
    print("Buscando contorno del tablero de Sudoku...")
    found = False
    corners = contour = None
    try:
        corners, contour = find_sudoku_contour(edge_bin)
        found = True
        print("Tablero detectado usando mapa de bordes difusos")
    except RuntimeError:
        print("Detección con bordes difusos falló, probando umbralización adaptativa...")
    if not found:
        # Adaptive threshold fallback
        gray8 = (gray * 255).astype(np.uint8)
        thr = cv2.adaptiveThreshold(gray8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 5)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        try:
            corners, contour = find_sudoku_contour(thr)
            found = True
            print("Tablero detectado usando umbralización adaptativa")
        except RuntimeError:
            print("⚠️  Umbralización adaptativa falló, probando extracción morfológica...")
    if not found:
        # Morphological line extraction fallback
        gray8 = (gray * 255).astype(np.uint8)
        try:
            corners, contour = find_sudoku_contour_morphological(gray8)
            found = True
            print("Tablero detectado usando extracción morfológica de líneas")
        except RuntimeError as e:
            print(f"Error: No se pudo detectar el tablero de Sudoku - {e}")
            return
    
    # Warp the board
    print("Rectificando perspectiva del tablero...")
    warped_board, M = warp_sudoku((gray * 255).astype(np.uint8), corners)
    print("Perspectiva rectificada")
    
    # Aplicar detección de bordes sobre el tablero rectificado para mejorar la segmentación de dígitos
    print("Aplicando detección de bordes sobre el tablero rectificado...")
    Ix2, Iy2 = compute_gradients(warped_board / 255.0)
    edge_map_board = fuzzy_edge_detection(
        Ix2, Iy2,
        sx=0.07,  # Parámetros ajustados para celdas pequeñas
        sy=0.07,
        wa=0.1,
        wb=1.0,
        wc=1.0,
        ba=0.0,
        bb=0.0,
        bc=0.7,
        sample_points=21,
        dtype=np.float32,
    )
    edge_norm_board = cv2.normalize(edge_map_board, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_bin_board = cv2.threshold(edge_norm_board, int(0.25 * 255), 255, cv2.THRESH_BINARY_INV)
    kernel2 = np.ones((3, 3), np.uint8)
    edge_bin_board = cv2.morphologyEx(edge_bin_board, cv2.MORPH_CLOSE, kernel2)
    print("Bordes en tablero rectificado generados")

    # Usar el edge_bin_board para extraer los dígitos
    print("Reconociendo dígitos en el tablero (lógica difusa, usando bordes del tablero rectificado)...")
    digits_grid = recognise_digits(edge_bin_board)
    print("Reconocimiento de dígitos completado")

    # Guardar imágenes de las celdas para análisis visual
    save_cells_dir = args.save_dir or "test"
    os.makedirs(save_cells_dir, exist_ok=True)
    cell_size = edge_bin_board.shape[0] // 9
    for i in range(9):
        for j in range(9):
            y = i * cell_size
            x = j * cell_size
            cell_img = edge_bin_board[y:y + cell_size, x:x + cell_size]
            cv2.imwrite(os.path.join(save_cells_dir, f"cell_{i+1}_{j+1}.png"), cell_img)
    print(f"Imágenes de celdas guardadas en: {save_cells_dir}")
    # Intentar resolver el tablero
    print_sudoku_grid(digits_grid, "🔢 DÍGITOS DETECTADOS EN EL TABLERO")
    detected_count = np.count_nonzero(digits_grid)
    print(f"Se detectaron {detected_count} dígitos en el tablero")
    print_sudoku_grid(digits_grid, "CUADRÍCULA DETECTADA (sin resolver)")
    print("\nIntentando resolver el tablero...")
    board_to_solve = digits_grid.copy()
    solved = solve_sudoku(board_to_solve, verbose=args.verbose)
    if solved:
        print_sudoku_grid(board_to_solve, "SOLUCIÓN DEL SUDOKU")
        # Guardar imagen del tablero recortado
        cv2.imwrite(os.path.join(save_cells_dir, "warped_board.png"), warped_board)
        # Generar y guardar imagen con la solución sobrepuesta
        solution_img = draw_solution(warped_board, board_to_solve)
        cv2.imwrite(os.path.join(save_cells_dir, "solution.png"), solution_img)
        print(f"Imagen del tablero recortado guardada en: {save_cells_dir}/warped_board.png")
        print(f"Imagen de la solución sobrepuesta guardada en: {save_cells_dir}/solution.png")
    else:
        print("No se pudo encontrar una solución. El tablero es muy complejo o insuficiente información.")
    # Crear imagen mosaico con todas las celdas juntas en escala de grises
    create_cells_grid_image(edge_bin_board, cell_size, save_cells_dir)
def create_cells_grid_image(board_img, cell_size, output_dir):
    """
    Crea una imagen mosaico con todas las celdas del Sudoku en escala de grises y la guarda en output_dir.
    """
    import numpy as np
    import cv2
    # El tablero ya está segmentado en edge_bin_board, así que solo lo guardamos como imagen general
    cv2.imwrite(os.path.join(output_dir, "all_cells_grid.png"), board_img)
    print(f"Imagen mosaico de todas las celdas guardada en: {output_dir}/all_cells_grid.png")
    


if __name__ == "__main__":
    main()
