# tinyfuzzy.py
# A tiny, pure-NumPy fuzzy logic helper for Pythonista/iOS (no SciPy / scikit-learn).
# Features:
# - Membership functions: triangular, trapezoidal, gaussian
# - Operators: AND (min), OR (max), NOT (1-x)
# - Mamdani inference: fuzzify -> rule eval -> aggregate -> centroid defuzz
# - Sugeno inference (zero- or first-order) with weighted average
# - Fuzzy c-means (FCM) clustering (pure NumPy implementation)
#
# MIT License
#
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union, Optional

# ----------------
# Membership funcs
# ----------------
def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
	x = np.asarray(x, dtype=float)
	y = np.zeros_like(x)
	# left slope
	idx = (a < x) & (x < b)
	y[idx] = (x[idx] - a) / (b - a + 1e-12)
	# right slope
	idx = (b <= x) & (x < c)
	y[idx] = (c - x[idx]) / (c - b + 1e-12)
	y[x == b] = 1.0
	return np.clip(y, 0.0, 1.0)
	
def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
	x = np.asarray(x, dtype=float)
	y = np.zeros_like(x)
	# rising edge
	idx = (a < x) & (x < b)
	y[idx] = (x[idx] - a) / (b - a + 1e-12)
	# plateau
	y[(b <= x) & (x <= c)] = 1.0
	# falling edge
	idx = (c < x) & (x < d)
	y[idx] = (d - x[idx]) / (c - d + 1e-12)  # note: denominator uses (d - c)
	# fix sign (typo corrected)
	y[idx] = (d - x[idx]) / (d - c + 1e-12)
	return np.clip(y, 0.0, 1.0)
	
def gaussmf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
	x = np.asarray(x, dtype=float)
	sigma = max(float(sigma), 1e-12)
	return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
	
# ----------
# Data types
# ----------
@dataclass
class FuzzySet:
	name: str
	mf: Callable[[np.ndarray], np.ndarray]
	
@dataclass
class FuzzyVar:
	name: str
	universe: Tuple[float, float]
	sets: Dict[str, FuzzySet]
	
# ----------
# Utilities
# ----------
def _singleton(x: np.ndarray, val: float) -> np.ndarray:
	"""Return MF that is '1' at the closest sample to val, else 0 (for testing)."""
	y = np.zeros_like(x)
	idx = int(np.argmin(np.abs(x - val)))
	y[idx] = 1.0
	return y
	
# ----------------------
# Mamdani Inference Core
# ----------------------
def fuzzify(crisp_inputs: Dict[str, float], variables: Dict[str, LinguisticVariable]) -> Dict[str, Dict[str, float]]:
	mu = {}
	for vname, val in crisp_inputs.items():
		lv = variables[vname]
		# pick a fine grid for interpolation
		xs = np.linspace(*lv.universe, 2001)
		mu[vname] = {}
		for sname, fz in lv.sets.items():
			y = fz.mf(xs)
			idx = int(np.argmin(np.abs(xs - val)))
			mu[vname][sname] = float(y[idx])
	return mu
	
def _eval_antecedent(mu_inputs: Dict[str, Dict[str, float]], antecedent: List[Tuple[str, str, str]]) -> float:
	if not antecedent:
		return 0.0
	v, s, op = antecedent[0]
	acc = mu_inputs[v][s]
	pending = op
	for (v, s, op) in antecedent[1:]:
		if pending == "AND":
			acc = min(acc, mu_inputs[v][s])
		elif pending == "OR":
			acc = max(acc, mu_inputs[v][s])
		pending = op
	return float(acc)
	
def mamdani(
    crisp_inputs : Dict[str, float],
    variables: Dict[str, LinguisticVariable],
    rules: List[Tuple[List[Tuple[str, str, str]], Tuple[str, str]]],
    output_var: str,
    implication: str = "min",
    aggregation: str = "max",
    samples: int = 2001
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	rules: list of (antecedent, consequent)
	antecedent: [(var, set, op_next), ..., (var, set, "")]
	consequent: (out_var, out_set)
	returns (xs, mu_out) where xs is the discretized universe for output_var
	"""
	out_lv = variables[output_var]
	xs = np.linspace(*out_lv.universe, samples)
	mu_out = np.zeros_like(xs)
	mu_inputs = fuzzify(crisp_inputs, variables)
	
	for antecedent, (ov, oset) in rules:
		w = _eval_antecedent(mu_inputs, antecedent)
		y = out_lv.sets[oset].mf(xs)
		if implication == "min":
			clip = np.minimum(y, w)
		elif implication == "prod":
			clip = y * w
		else:
			raise ValueError("implication must be 'min' or 'prod'")
		if aggregation == "max":
			mu_out = np.maximum(mu_out, clip)
		elif aggregation == "sum_clip":
			mu_out = np.clip(mu_out + clip, 0.0, 1.0)
		else:
			raise ValueError("aggregation must be 'max' or 'sum_clip'")
	return xs, mu_out
	
def centroid(xs: np.ndarray, mu: np.ndarray) -> float:
	num = np.trapz(mu * xs, xs)
	den = np.trapz(mu, xs)
	if den == 0:
		return float(xs[len(xs)//2])
	return float(num / den)
	
# -------------------
# Sugeno Inference
# -------------------
# Consequent can be:
#  - constant: float
#  - linear: (bias, {"x1": a1, "x2": a2, ...}) -> z = bias + sum(a_i * x_i)
Consequent = Union[float, Tuple[float, Dict[str, float]]]

def sugeno(
    crisp_inputs: Dict[str, float],
    variables: Dict[str, LinguisticVariable],
    rules: List[Tuple[List[Tuple[str, str, str]], Consequent]]
) -> float:
	mu_inputs = fuzzify(crisp_inputs, variables)
	weights = []
	zs = []
	for antecedent, cons in rules:
		w = _eval_antecedent(mu_inputs, antecedent)
		if isinstance(cons, (float, int)):
			z = float(cons)
		else:
			bias, coeffs = cons
			z = float(bias + sum(coeffs.get(k, 0.0) * crisp_inputs[k] for k in crisp_inputs))
		weights.append(w)
		zs.append(z)
	wsum = sum(weights) + 1e-12
	return float(sum(w*z for w, z in zip(weights, zs)) / wsum)
	
# -------------------
# Fuzzy C-Means (FCM)
# -------------------
def cmeans(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 150, tol: float = 1e-5, seed: Optional[int] = None):
	"""
	Simple FCM implementation.
	X: (n_samples, n_features)
	c: number of clusters
	m: fuzziness (>1)
	returns: centers (c, n_features), membership (c, n_samples)
	"""
	rng = np.random.default_rng(seed)
	X = np.asarray(X, dtype=float)
	n, d = X.shape
	# Initialize membership randomly (rows sum to 1 over clusters per sample)
	U = rng.random((c, n))
	U /= U.sum(axis=0, keepdims=True)
	
	def update_centers(U):
		Um = U ** m
		num = Um @ X
		den = Um.sum(axis=1, keepdims=True)
		return num / (den + 1e-12)
		
	def dist2(A, B):
		# squared euclidean distances between rows of A (c,d) and rows of B (n,d)
		# returns (c,n)
		return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
		
	J_prev = np.inf
	for _ in range(max_iter):
		C = update_centers(U)
		D2 = dist2(C, X) + 1e-12  # avoid zero
		# update U
		invD = D2 ** (-1.0 / (m - 1.0))
		U_new = invD / invD.sum(axis=0, keepdims=True)
		# objective
		Um = U_new ** m
		J = np.sum(Um * D2)
		if abs(J - J_prev) < tol * (1.0 + abs(J_prev)):
			U = U_new
			break
		U, J_prev = U_new, J
	return C, U
