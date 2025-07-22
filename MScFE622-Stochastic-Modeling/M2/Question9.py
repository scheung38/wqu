import numpy as np
from scipy.integrate import quad

def merton_char_func(u, S0, r, sigma, T, lamb, muJ, sigmaJ):
    """Characteristic function for the Merton jump-diffusion model."""
    drift = r - 0.5 * sigma**2 - lamb * (np.exp(muJ + 0.5 * sigmaJ**2) - 1)
    return np.exp(
        1j * u * (np.log(S0) + drift * T)
        - 0.5 * sigma**2 * u**2 * T
        + lamb * T * (np.exp(1j * u * muJ - 0.5 * sigmaJ**2 * u**2) - 1)
    )

def lewis_put_price(S0, K, T, r, sigma, lamb, muJ, sigmaJ):
    """Lewis (2001) Fourier inversion for European put under Merton model."""
    def integrand(phi):
        u = phi - 1j * 0.5
        cf = merton_char_func(u, S0, r, sigma, T, lamb, muJ, sigmaJ)
        numerator = np.exp(-1j * phi * np.log(K)) * cf
        denominator = 1j * phi * S0 * np.exp(-r * T)
        return np.real(numerator / denominator)
    
    integral, _ = quad(lambda phi: integrand(phi), 0, 100)
    call_price = S0 - np.exp(-r * T) * K / 2 + integral / np.pi
    # Put-Call Parity: P = C - S0 + K*exp(-rT)
    put_price = call_price - S0 + K * np.exp(-r * T)
    return put_price

# Parameters
S0 = 8.75
K = 10
T = 35 / 365  # Convert days to years
r = 0.0
sigma = 0.65
lamb = 2
muJ = -0.65
sigmaJ = 0.1

put_price = lewis_put_price(S0, K, T, r, sigma, lamb, muJ, sigmaJ)
print(f"Merton Model Put Price: {put_price:.2f}")
