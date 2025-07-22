import numpy as np
from scipy.integrate import quad

def bates_char_func(u, S0, r, T, kappa, theta, sigma_v, rho, v0, lamb, muJ, sigmaJ):
    d = np.sqrt((rho * sigma_v * 1j * u - kappa)**2 + (sigma_v**2) * (1j * u + u**2))
    g = (kappa - rho * sigma_v * 1j * u - d) / (kappa - rho * sigma_v * 1j * u + d)
    exp_dT = np.exp(-d * T)
    C = r * 1j * u * T + (kappa * theta) / sigma_v**2 * (
        (kappa - rho * sigma_v * 1j * u - d) * T
        - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((kappa - rho * sigma_v * 1j * u - d) / sigma_v**2) * ((1 - exp_dT) / (1 - g * exp_dT))
    jump_cf = np.exp(lamb * T * (np.exp(1j * u * muJ - 0.5 * sigmaJ**2 * u**2) - 1))
    return np.exp(C + D * v0 + 1j * u * np.log(S0)) * jump_cf

def lewis_call_price_bates(S0, K, T, r, kappa, theta, sigma_v, rho, v0, lamb, muJ, sigmaJ):
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
S0 = 2247.5
K = 2250
T = 19 / 365
r = 0.025
kappa = 1.85
theta = 0.06
sigma_v = 0.45
rho = -0.75
v0 = 0.21
lamb = 0.13
muJ = -0.4
sigmaJ = 0.3

call_price = lewis_call_price_bates(S0, K, T, r, kappa, theta, sigma_v, rho, v0, lamb, muJ, sigmaJ)
print(f"Bates Model Call Price: {call_price:.2f}")
