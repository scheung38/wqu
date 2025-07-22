# Implements the characteristic function for the Bates model (Heston + jumps).
# Uses the Lewis (2001) Fourier inversion for option pricing.
# Prints the call price for your specified parameters.
import numpy as np
from scipy.integrate import quad

def bates_char_func(u, S0, r, T, kappa, theta, sigma_v, rho, v0, lamb, muJ, sigmaJ):
    """Characteristic function for the Bates (Heston + jumps) model."""
    # Heston part
    d = np.sqrt((rho * sigma_v * 1j * u - kappa)**2 + (sigma_v**2) * (1j * u + u**2))
    g = (kappa - rho * sigma_v * 1j * u - d) / (kappa - rho * sigma_v * 1j * u + d)
    exp_dT = np.exp(-d * T)
    C = r * 1j * u * T + (kappa * theta) / sigma_v**2 * (
        (kappa - rho * sigma_v * 1j * u - d) * T
        - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((kappa - rho * sigma_v * 1j * u - d) / sigma_v**2) * ((1 - exp_dT) / (1 - g * exp_dT))
    # Jumps part (Merton)
    jump_cf = np.exp(lamb * T * (np.exp(1j * u * muJ - 0.5 * sigmaJ**2 * u**2) - 1))
    # Combine
    return np.exp(C + D * v0 + 1j * u * np.log(S0)) * jump_cf

def lewis_call_price_bates(S0, K, T, r, kappa, theta, sigma_v, rho, v0, lamb, muJ, sigmaJ):
    """Lewis (2001) Fourier inversion for European call under Bates model."""
    def integrand(phi):
        u = phi - 1j * 0.5
        cf = bates_char_func(u, S0, r, T, kappa, theta, sigma_v, rho, v0, lamb, muJ, sigmaJ)
        numerator = np.exp(-1j * phi * np.log(K)) * cf
        denominator = 1j * phi
        return np.real(numerator / denominator)
    
    integral, _ = quad(lambda phi: integrand(phi), 0, 100, limit=500)
    call_price = S0 - np.exp(-r * T) * K / 2 + integral / np.pi
    return call_price

# Parameters
S0 = 10.65
K = 22
T = 265 / 365  # Convert days to years
r = 0.0
kappa = 0.85
theta = 0.16
sigma_v = 0.15
rho = -0.95
v0 = 0.016
lamb = 0.1
muJ = -0.05
sigmaJ = 0.9

call_price = lewis_call_price_bates(S0, K, T, r, kappa, theta, sigma_v, rho, v0, lamb, muJ, sigmaJ)
print(f"Bates Model Call Price: {call_price:.4f}")