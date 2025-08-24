"""
Fuzzy edge detection using a simple Mamdani‐type fuzzy inference system.

This module replicates the Matlab/Scikit‑Fuzzy edge detection example described in
the PDF provided by the teacher.  The goal of that example is to detect edges in
images by comparing the local horizontal and vertical gradients and using
fuzzy logic to decide whether each pixel belongs to a uniform region (white)
or an edge (black).  The fuzzy system is defined as follows:

* Inputs: horizontal gradient (Ix) and vertical gradient (Iy), both in the
  range ``[-1, 1]``.  Each input has a single Gaussian membership function
  labelled ``zero`` centred at 0 with user‑defined standard deviations
  ``sx`` and ``sy``.  The degree of membership of a gradient
  value indicates how close it is to zero (i.e. how flat the local image
  patch is).

* Outputs: detected edge intensity (Iout) in the range ``[0, 1]``.  Two
  triangular membership functions are defined: ``white`` which peaks at 1 and
  represents uniform regions, and ``black`` which peaks at 0 and represents
  edges.  The parameters of these triangles can be tuned via ``wa``, ``wb``,
  ``wc``, ``ba``, ``bb`` and ``bc``.

* Rules: a pixel is considered ``white`` (non‑edge) if **both** horizontal and
  vertical gradients are zero; otherwise it is ``black`` (edge). 
  Formally:

  1. ``If Ix is zero and Iy is zero then Iout is white``
  2. ``If Ix is not zero or Iy is not zero then Iout is black``

The module uses NumPy and OpenCV to load and preprocess images, computes
gradients via simple convolution, evaluates the fuzzy system on the entire
image in a fully vectorised manner and produces diagnostic plots with
Matplotlib.  It does **not** depend on SciPy or scikit‑fuzzy – instead
membership functions from the accompanying ``tinyfuzzy.py`` are used.

Example:

    python fuzzy_edge_detection.py --image peppers.png --show

This will load ``peppers.png``, convert it to grayscale, run the fuzzy edge
detection algorithm and display the original image, the gradients, the
membership functions and the detected edges.

Notes
-----
* The implementation follows the algorithm in the provided PDF closely:
  gradients are computed with the kernels ``[-1, 1]`` and its transpose; default membership parameters match those in the
  example; and the rule base matches the one described.
* Because evaluating a fuzzy system for every pixel individually is slow,
  this code vectorises the Mamdani inference.  For each pixel two rule
  strengths are computed: one for the ``white`` rule and one for the
  ``black`` rule.  These rule strengths are used to clip the output
  membership functions (sampled on a user defined grid), and the centroid
  defuzzification is performed in closed form across all pixels at once.  The
  number of samples on the output universe can be controlled via
  ``sample_points``; increasing it will provide a more accurate defuzzified
  output at the cost of memory and computation time.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tinyfuzzy import gaussmf, trimf


def load_image_grayscale(path: str) -> np.ndarray:
    """Load an image from disk and convert it to a 2‑D grayscale array.

    Parameters
    ----------
    path : str
        Path to the image file.  Any format readable by OpenCV is supported.

    Returns
    -------
    I : ndarray of shape (H, W) with dtype float64
        Grayscale image normalised to the range [0, 1].
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file '{path}' does not exist.")
    # OpenCV loads images in BGR order by default.  Use cv2.IMREAD_UNCHANGED
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read image '{path}'.")
    # If the image has multiple channels, convert to grayscale via standard luminance
    if img.ndim == 3:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalise to [0,1]
        img_float = img_rgb.astype(np.float64) / 255.0
        # Luminance conversion (ITU-R BT.601): Y = 0.299 R + 0.587 G + 0.114 B
        I = 0.2989 * img_float[..., 0] + 0.5870 * img_float[..., 1] + 0.1140 * img_float[..., 2]
    else:
        # Single channel: assume 8‑bit or 16‑bit, normalise accordingly
        img_float = img.astype(np.float64)
        # Determine maximum value based on dtype
        max_val = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0
        I = img_float / max_val
    return I


def compute_gradients(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute horizontal and vertical gradients of a grayscale image.

    The gradients are computed by convolving the image with the kernels
    ``[-1, 1]`` (for x) and ``[-1, 1]^T`` (for y), replicating the Matlab
    example described in the PDF.  Border values are
    handled by replication so that the output has the same shape as the input.

    Parameters
    ----------
    I : ndarray of shape (H, W)
        Grayscale image with values in [0, 1].

    Returns
    -------
    Ix : ndarray of shape (H, W)
        Horizontal gradient values in the range [-1, 1].
    Iy : ndarray of shape (H, W)
        Vertical gradient values in the range [-1, 1].
    """
    # Define the horizontal and vertical kernels
    Gx = np.array([[-1, 1]], dtype=np.float64)
    Gy = Gx.T
    # Convolve using cv2.filter2D; ddepth=-1 preserves input depth
    Ix = cv2.filter2D(I, ddepth=-1, kernel=Gx, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(I, ddepth=-1, kernel=Gy, borderType=cv2.BORDER_REPLICATE)
    # Clip the range to [-1, 1] as per the example description
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
    """Run the fuzzy edge detector on precomputed gradients.

    This function implements a Mamdani inference system with two rules and
    performs vectorised defuzzification.  Each pixel is treated as a crisp
    input (Ix[p], Iy[p]) and two rule strengths are computed:

    ``w_white  = min(mu_zero(Ix[p]), mu_zero(Iy[p]))``
    ``w_black  = max(1 - mu_zero(Ix[p]), 1 - mu_zero(Iy[p]))``

    where ``mu_zero`` is the Gaussian membership function centred at zero with
    standard deviation ``sx`` or ``sy``.  These weights are used
    to clip the output membership functions for ``white`` and ``black`` and
    centroid defuzzification produces the final edge intensity value.

    Parameters
    ----------
    Ix, Iy : ndarray of shape (H, W)
        Horizontal and vertical gradients.
    sx, sy : float, optional
        Standard deviations of the Gaussian membership functions for the
        ``zero`` sets on ``Ix`` and ``Iy``.  Larger values make the
        detector less sensitive.
    wa, wb, wc : float, optional
        Parameters of the triangular membership function for the ``white``
        output set.  ``wa`` is the left foot, ``wb`` the peak
        and ``wc`` the right foot.
    ba, bb, bc : float, optional
        Parameters of the triangular membership function for the ``black``
        output set.  ``ba`` is the left foot, ``bb`` the peak and ``bc`` the right foot.
    sample_points : int, optional
        Number of sample points to discretise the output universe ``[0, 1]``.
        A larger value yields a more accurate defuzzified output but increases
        memory use.
    dtype : numpy dtype, optional
        Data type for intermediate arrays.  Using ``np.float32`` saves memory at
        the cost of precision.

    Returns
    -------
    Iout : ndarray of shape (H, W)
        Defuzzified edge intensity values in the range ``[0, 1]``.
    """
    H, W = Ix.shape
    # Compute membership degrees for "zero" on Ix and Iy
    mu_zero_x = gaussmf(Ix.astype(dtype), mean=0.0, sigma=sx).astype(dtype)
    mu_zero_y = gaussmf(Iy.astype(dtype), mean=0.0, sigma=sy).astype(dtype)
    # Rule strengths
    w_white = np.minimum(mu_zero_x, mu_zero_y)  # rule 1: both near zero
    w_black = np.maximum(1.0 - mu_zero_x, 1.0 - mu_zero_y)  # rule 2: any non‑zero

    # Universe of discourse for output variable
    xs = np.linspace(0.0, 1.0, sample_points, dtype=dtype)
    # Output membership functions
    white_mf = trimf(xs, wa, wb, wc).astype(dtype)
    black_mf = trimf(xs, ba, bb, bc).astype(dtype)

    # Flatten the rule strengths for vectorised computation
    w_white_flat = w_white.ravel().astype(dtype)
    w_black_flat = w_black.ravel().astype(dtype)

    # Broadcast output membership functions across pixels: shape (n_samples, n_pixels)
    # For each pixel: mu_out(x) = max(min(white_mf(x), w_white[p]), min(black_mf(x), w_black[p]))
    # Pre‑allocate arrays for numerator and denominator to avoid keeping the full
    # 2D mu_out array in memory simultaneously.  We compute contributions
    # sample by sample.
    n_samples = xs.size
    n_pixels = w_white_flat.size
    # Initialise accumulators
    numerator = np.zeros(n_pixels, dtype=dtype)
    denominator = np.zeros(n_pixels, dtype=dtype)

    # Loop over each sample point; accumulate centroid integrals.
    # Because we avoid constructing a (n_samples x n_pixels) array, this loop
    # scales with the number of sample points but is more memory efficient.
    for i in range(n_samples):
        wmf = white_mf[i]
        bmf = black_mf[i]
        # Clip white and black membership by rule strengths
        # min(white_mf[i], w_white) and min(black_mf[i], w_black)
        clipped_white = np.minimum(wmf, w_white_flat)
        clipped_black = np.minimum(bmf, w_black_flat)
        # Aggregation: max of the two clipped sets
        mu_out_i = np.maximum(clipped_white, clipped_black)
        # Update integrals
        numerator += mu_out_i * xs[i]
        denominator += mu_out_i
    # Avoid division by zero
    denom_safe = np.where(denominator == 0, 1.0, denominator)
    result_flat = numerator / denom_safe
    # Reshape back to image shape
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
    """Plot membership functions for inputs Ix, Iy and output Iout.

    Parameters
    ----------
    sx, sy : float
        Standard deviations for the Gaussian ``zero`` membership functions.
    wa, wb, wc, ba, bb, bc : float
        Parameters for the ``white`` and ``black`` output membership functions.
    save_path : str, optional
        If provided, save the figure to this path instead of displaying it.
    """
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
    args = parser.parse_args()

    # Load and preprocess the image
    I = load_image_grayscale(args.image)
    Ix, Iy = compute_gradients(I)
    # Run fuzzy edge detection
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

    # If the user wants plots, generate them
    if not args.no_plots:
        save_dir = args.save_dir
        if save_dir is not None and not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        # Plot membership functions
        mfp = None
        if save_dir:
            mfp = os.path.join(save_dir, "membership_functions.png")
        plot_membership_functions(args.sx, args.sy, args.wa, args.wb, args.wc, args.ba, args.bb, args.bc, save_path=mfp)
        # Plot gradients and results
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
            plt.close(fig2)
        else:
            plt.show()


if __name__ == "__main__":
    main()
