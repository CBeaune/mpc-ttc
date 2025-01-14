# Import necessary libraries

from scipy.linalg import sqrtm, eigh, inv
import numpy as np

def compute_ellipse_parameters(sigmax, sigmay, sigmaxy, eta=0.997):
    """Compute the parameters of the ellipse."""
    a = np.sqrt(-2 * np.log(1 - eta))
    Sigma = np.array([[sigmax**2, sigmaxy], [sigmaxy, sigmay**2]])
    A = a * sqrtm(Sigma)
    w, v = eigh(A)
    v1 = np.real(v[:, 0])
    v2 = np.real(v[:, 1])
    f1 = A @ v1
    f2 = A @ v2
    φ = np.arctan2(v1[1], v1[0])
    α = φ
    return np.linalg.norm(f1), np.linalg.norm(f2), α
