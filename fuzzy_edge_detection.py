
from __future__ import annotations

import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tinyfuzzy import gaussmf, trimf

try:
    matplotlib.use('TkAgg')
except ImportError:
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        matplotlib.use('Agg')  # Non-interactive fallback


def load_image_grayscale(path: str) -> np.ndarray:

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file '{path}' does not exist.")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read image '{path}'.")
    if img.ndim == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float64) / 255.0
        I = 0.2989 * img_float[..., 0] + 0.5870 * img_float[..., 1] + 0.1140 * img_float[..., 2]
    else:
        img_float = img.astype(np.float64)
        max_val = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0
        I = img_float / max_val
    return I


def compute_gradients(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Gx = np.array([[-1, 1]], dtype=np.float64)
    Gy = Gx.T
    Ix = cv2.filter2D(I, ddepth=-1, kernel=Gx, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(I, ddepth=-1, kernel=Gy, borderType=cv2.BORDER_REPLICATE)
    Ix = np.clip(Ix, -1.0, 1.0)
    Iy = np.clip(Iy, -1.0, 1.0)
    return Ix, Iy


def fuzzy_edge_detection(
    Ix: np.ndarray,
    Iy: np.ndarray,
    *,
    sx: float = 0.1,
    sy: float = 0.1,
    wa: float = 0.1,
    wb: float = 1.0,
    wc: float = 1.0,
    ba: float = 0.0,
    bb: float = 0.0,
    bc: float = 0.7,
    sample_points: int = 51,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    H, W = Ix.shape
    mu_zero_x = gaussmf(Ix.astype(dtype), mean=0.0, sigma=sx).astype(dtype)
    mu_zero_y = gaussmf(Iy.astype(dtype), mean=0.0, sigma=sy).astype(dtype)
    w_white = np.minimum(mu_zero_x, mu_zero_y)
    w_black = np.maximum(1.0 - mu_zero_x, 1.0 - mu_zero_y)

    xs = np.linspace(0.0, 1.0, sample_points, dtype=dtype)
    white_mf = trimf(xs, wa, wb, wc).astype(dtype)
    black_mf = trimf(xs, ba, bb, bc).astype(dtype)

    w_white_flat = w_white.ravel().astype(dtype)
    w_black_flat = w_black.ravel().astype(dtype)

    n_samples = xs.size
    n_pixels = w_white_flat.size

    numerator = np.zeros(n_pixels, dtype=dtype)
    denominator = np.zeros(n_pixels, dtype=dtype)

    for i in range(n_samples):
        wmf = white_mf[i]
        bmf = black_mf[i]

        clipped_white = np.minimum(wmf, w_white_flat)
        clipped_black = np.minimum(bmf, w_black_flat)

        mu_out_i = np.maximum(clipped_white, clipped_black)

        numerator += mu_out_i * xs[i]
        denominator += mu_out_i

    denom_safe = np.where(denominator == 0, 1.0, denominator)
    result_flat = numerator / denom_safe

    return result_flat.reshape(H, W)


def plot_membership_functions(
    sx: float,
    sy: float,
    wa: float,
    wb: float,
    wc: float,
    ba: float,
    bb: float,
    bc: float,
    *,
    save_path: str | None = None,
) -> None:

    # Input domains
    x_in = np.linspace(-1.0, 1.0, 400)
    # Output domain
    x_out = np.linspace(0.0, 1.0, 400)
    # Membership functions
    mu_zero = gaussmf(x_in, mean=0.0, sigma=sx)
    # Identical for Ix and Iy
    mu_white = trimf(x_out, wa, wb, wc)
    mu_black = trimf(x_out, ba, bb, bc)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Ix membership
    axes[0, 0].plot(x_in, mu_zero, label="zero", color="C0")
    axes[0, 0].set_title("Función de membresía para Ix (zero)")
    axes[0, 0].set_xlabel("Gradiente Ix")
    axes[0, 0].set_ylabel("Grado de pertenencia")
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].grid(True)
    # Iy membership
    axes[0, 1].plot(x_in, mu_zero, label="zero", color="C1")
    axes[0, 1].set_title("Función de membresía para Iy (zero)")
    axes[0, 1].set_xlabel("Gradiente Iy")
    axes[0, 1].set_ylabel("Grado de pertenencia")
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True)
    # Output membership
    axes[1, 0].plot(x_out, mu_white, label="white", color="C2")
    axes[1, 0].plot(x_out, mu_black, label="black", color="C3")
    axes[1, 0].set_title("Funciones de membresía para Iout")
    axes[1, 0].set_xlabel("Intensidad Iout")
    axes[1, 0].set_ylabel("Grado de pertenencia")
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].grid(True)
    # Remove the unused subplot
    axes[1, 1].axis('off')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzzy logic edge detection example")
    parser.add_argument("--image", type=str, default="peppers.png",
                        help="Path to the input image (default: peppers.png)")
    parser.add_argument("--sx", type=float, default=0.1, help="Sigma for Ix zero membership")
    parser.add_argument("--sy", type=float, default=0.1, help="Sigma for Iy zero membership")
    parser.add_argument("--wa", type=float, default=0.1, help="White membership left foot")
    parser.add_argument("--wb", type=float, default=1.0, help="White membership peak")
    parser.add_argument("--wc", type=float, default=1.0, help="White membership right foot")
    parser.add_argument("--ba", type=float, default=0.0, help="Black membership left foot")
    parser.add_argument("--bb", type=float, default=0.0, help="Black membership peak")
    parser.add_argument("--bc", type=float, default=0.7, help="Black membership right foot")
    parser.add_argument("--samples", type=int, default=51, help="Number of samples for defuzzification")
    parser.add_argument("--no_plots", action="store_true", help="Skip plotting results")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save output images instead of displaying")
    parser.add_argument("--show", action="store_true", help="Show plots interactively (default behavior when no save_dir is specified)")
    args = parser.parse_args()

    I = load_image_grayscale(args.image)
    Ix, Iy = compute_gradients(I)
    Iout = fuzzy_edge_detection(
        Ix,
        Iy,
        sx=args.sx,
        sy=args.sy,
        wa=args.wa,
        wb=args.wb,
        wc=args.wc,
        ba=args.ba,
        bb=args.bb,
        bc=args.bc,
        sample_points=args.samples,
    )

    if not args.no_plots:
        save_dir = args.save_dir
        show_plots = args.show or (save_dir is None)
        
        if save_dir is not None and not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        mfp = None
        if save_dir:
            mfp = os.path.join(save_dir, "membership_functions.png")
        plot_membership_functions(args.sx, args.sy, args.wa, args.wb, args.wc, args.ba, args.bb, args.bc, save_path=mfp)
        
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
        # Original grayscale
        axes2[0, 0].imshow(I, cmap='gray', vmin=0, vmax=1)
        axes2[0, 0].set_title("Imagen de entrada en escala de grises")
        axes2[0, 0].axis('off')
        # Ix
        axes2[0, 1].imshow(Ix, cmap='gray', vmin=-1, vmax=1)
        axes2[0, 1].set_title("Gradiente horizontal Ix")
        axes2[0, 1].axis('off')
        # Iy
        axes2[1, 0].imshow(Iy, cmap='gray', vmin=-1, vmax=1)
        axes2[1, 0].set_title("Gradiente vertical Iy")
        axes2[1, 0].axis('off')
        # Edge detection result
        axes2[1, 1].imshow(Iout, cmap='gray', vmin=0, vmax=1)
        axes2[1, 1].set_title("Detección de bordes mediante lógica difusa")
        axes2[1, 1].axis('off')
        fig2.tight_layout()
        
        if save_dir:
            fig2.savefig(os.path.join(save_dir, "edge_detection_results.png"), bbox_inches="tight")
        
        if show_plots:
            plt.show()
        
        if save_dir and not show_plots:
            plt.close(fig2)


if __name__ == "__main__":
    main()
