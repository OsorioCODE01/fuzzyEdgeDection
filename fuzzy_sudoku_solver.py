#Sudoku solver using fuzzy logic edge detection.
from __future__ import annotations

import argparse
import os
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_edge_detection import (
    load_image_grayscale,
    compute_gradients,
    fuzzy_edge_detection,
)


def order_points(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def find_sudoku_contour(edge_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = edge_img.shape[0] * edge_img.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.02 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        if 0.8 <= ratio <= 1.2:
            return approx.reshape(4, 2), c
    raise RuntimeError("No suitable Sudoku contour found.")


def find_sudoku_contour_morphological(gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #Detect a Sudoku grid using morphological line extraction.


    thr = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    horizontal = thr.copy()
    h_kernel_len = max(10, gray_img.shape[1] // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_temp = cv2.erode(horizontal, h_kernel, iterations=1)
    h_lines = cv2.dilate(h_temp, h_kernel, iterations=1)
    vertical = thr.copy()
    v_kernel_len = max(10, gray_img.shape[0] // 30)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_temp = cv2.erode(vertical, v_kernel, iterations=1)
    v_lines = cv2.dilate(v_temp, v_kernel, iterations=1)
    grid = cv2.bitwise_or(h_lines, v_lines)
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return find_sudoku_contour(grid)


def warp_sudoku(
    gray_img: np.ndarray,
    corner_points: np.ndarray,
    size: int = 450,
) -> Tuple[np.ndarray, np.ndarray]:

    rect = order_points(corner_points)
    (tl, tr, br, bl) = rect
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
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray_img, M, (size, size))
    return warped, M




def extract_cell_digit(cell: np.ndarray) -> Tuple[np.ndarray, bool]:
    h, w = cell.shape
    margin = int(min(h, w) * 0.15)
    if margin > 0:
        inner = cell[margin:h - margin, margin:w - margin]
    else:
        inner = cell.copy()
    inner_blur = cv2.GaussianBlur(inner, (3, 3), 0)
    _, thresh = cv2.threshold(inner_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(thresh) < 0.07 * thresh.size:
        return np.zeros((8, 8), dtype=np.float32), False
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((8, 8), dtype=np.float32), False
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.05 * thresh.shape[0] * thresh.shape[1]:
        return np.zeros((8, 8), dtype=np.float32), False
    x, y, w_box, h_box = cv2.boundingRect(c)
    digit = thresh[y:y + h_box, x:x + w_box]
    h_d, w_d = digit.shape
    size = max(h_d, w_d)
    square = np.zeros((size, size), dtype=np.uint8)
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

    print(f"Métricas celda: {metrics}")

    if metrics["percent_white"] < 0.10 or metrics["percent_white"] > 0.95:
        print(f"Detectado: vacío | {metrics}")
        return 0

    if metrics["num_components"] >= 2 and metrics["center_area"] > 10 and metrics["vert_sym"] < 15 and metrics["hor_sym"] < 15:
        print(f"Detectado: 8 | {metrics}")
        return 8

    if metrics["vertical"] > 15 and metrics["left_area"] < 5 and metrics["right_area"] < 5 and metrics["vert_sym"] < 8:
        print(f"Detectado: 1 | {metrics}")
        return 1

    if metrics["right_area"] > metrics["left_area"] + 2 and metrics["center_area"] > 7 and metrics["vert_sym"] > 8 and metrics["top_area"] > 5 and metrics["bottom_area"] > 5:
        print(f"Detectado: 3 | {metrics}")
        return 3
    
    if metrics["center_area"] > 10 and metrics["top_area"] > metrics["bottom_area"] and metrics["bottom_area"] < 7 and metrics["vert_sym"] > 12:
        print(f"Detectado: 9 | {metrics}")
        return 9
    
    if metrics["top_area"] > 6 and metrics["center_area"] < 7 and metrics["bottom_area"] < 7:
        print(f"Detectado: 7 | {metrics}")
        return 7

    if metrics["horizontal"] > 6 and metrics["right_area"] > metrics["left_area"] and metrics["top_area"] < 7:
        print(f"Detectado: 5 | {metrics}")
        return 5

    if digit_img[0,0] > 0.4 and digit_img[7,7] > 0.4 and metrics["vert_sym"] < 15 and metrics["hor_sym"] < 15 and metrics["center_area"] < 10:
        print(f"Detectado: 4 | {metrics}")
        return 4
    if metrics["horizontal"] > 6 and metrics["left_area"] > metrics["right_area"] and metrics["top_area"] < 7:
        print(f"Detectado: 2 | {metrics}")
        return 2
    if metrics["center_area"] > 8 and metrics["bottom_area"] > metrics["top_area"] and metrics["top_area"] < 7:
        print(f"Detectado: 6 | {metrics}")
        return 6

    if metrics["vert_sym"] < 10 and metrics["hor_sym"] < 10 and metrics["center_area"] > 10 and metrics["left_area"] > 8 and metrics["right_area"] > 8:
        print(f"Detectado: 0 | {metrics}")
        return 0

    print(f"Detectado: no identificado | {metrics}")
    return 0


def is_valid(board: np.ndarray, row: int, col: int, num: int) -> bool:

    if num in board[row, :]:
        return False
    if num in board[:, col]:
        return False
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    if num in board[box_row:box_row + 3, box_col:box_col + 3]:
        return False
    return True


def solve_sudoku(board: np.ndarray, verbose: bool = False) -> bool:

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
    out = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    cell_size = board_img.shape[0] // 9
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(9):
        for j in range(9):
            num = solution[i, j]
            text = str(int(num))
            x = j * cell_size + cell_size // 2
            y = i * cell_size + cell_size // 2
            (w, h), _ = cv2.getTextSize(text, font, 0.7, 2)
            cv2.putText(out, text, (x - w // 2, y + h // 2), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def print_sudoku_grid(grid: np.ndarray, title: str = ""):
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
    edge_norm = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_bin = cv2.threshold(edge_norm, int(args.thresh * 255), 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((7, 7), np.uint8)
    edge_bin = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, kernel)

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
        gray8 = (gray * 255).astype(np.uint8)
        try:
            corners, contour = find_sudoku_contour_morphological(gray8)
            found = True
            print("Tablero detectado usando extracción morfológica de líneas")
        except RuntimeError as e:
            print(f"Error: No se pudo detectar el tablero de Sudoku - {e}")
            return
    
    print("Rectificando perspectiva del tablero...")
    warped_board, M = warp_sudoku((gray * 255).astype(np.uint8), corners)
    print("Perspectiva rectificada")
    
    print("Aplicando detección de bordes sobre el tablero rectificado...")
    Ix2, Iy2 = compute_gradients(warped_board / 255.0)
    edge_map_board = fuzzy_edge_detection(
        Ix2, Iy2,
        sx=0.07,
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

    print("Reconociendo dígitos en el tablero (lógica difusa, usando bordes del tablero rectificado)...")
    digits_grid = recognise_digits(edge_bin_board)
    print("Reconocimiento de dígitos completado")

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
    print_sudoku_grid(digits_grid, "🔢 DÍGITOS DETECTADOS EN EL TABLERO")
    detected_count = np.count_nonzero(digits_grid)
    print(f"Se detectaron {detected_count} dígitos en el tablero")
    print_sudoku_grid(digits_grid, "CUADRÍCULA DETECTADA (sin resolver)")
    print("\nIntentando resolver el tablero...")
    board_to_solve = digits_grid.copy()
    solved = solve_sudoku(board_to_solve, verbose=args.verbose)
    if solved:
        print_sudoku_grid(board_to_solve, "SOLUCIÓN DEL SUDOKU")
        cv2.imwrite(os.path.join(save_cells_dir, "warped_board.png"), warped_board)
        solution_img = draw_solution(warped_board, board_to_solve)
        cv2.imwrite(os.path.join(save_cells_dir, "solution.png"), solution_img)
        print(f"Imagen del tablero recortado guardada en: {save_cells_dir}/warped_board.png")
        print(f"Imagen de la solución sobrepuesta guardada en: {save_cells_dir}/solution.png")
    else:
        print("No se pudo encontrar una solución. El tablero es muy complejo o insuficiente información.")
    create_cells_grid_image(edge_bin_board, cell_size, save_cells_dir)

def create_cells_grid_image(board_img, cell_size, output_dir):
    import numpy as np
    import cv2
    cv2.imwrite(os.path.join(output_dir, "all_cells_grid.png"), board_img)
    print(f"Imagen mosaico de todas las celdas guardada en: {output_dir}/all_cells_grid.png")
    


if __name__ == "__main__":
    main()
