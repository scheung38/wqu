import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, brute
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_prepare_data(excel_path, trading_days):
    """Loads data from Excel, pivots it, and prepares it for calibration."""
    print(f"Loading data from: {excel_path}")
    df = pd.read_excel(excel_path)

    # Pivot the table to get Call and Put prices in separate columns
    df = df.pivot_table(index=['Days to maturity', 'Strike'], columns='Type', values='Price').reset_index()
    df.rename(columns={'C': 'Call', 'P': 'Put'}, inplace=True)
    df.dropna(inplace=True)  # Remove rows with missing call or put prices

    # Calculate time to maturity in years
    df['T'] = df['Days to maturity'] / trading_days
    
    min_maturity_days = df['Days to maturity'].min()
    min_maturity_years = min_maturity_days / trading_days
    
    print(f"Shortest maturity available: {min_maturity_days:.0f} days. Using this for calibration.")
    filtered_df = df[df['T'] == min_maturity_years].copy()
    
    # Calculate Mid_Price for the error function
    filtered_df['Mid_Price'] = (filtered_df['Call'] + filtered_df['Put']) / 2
    return filtered_df, min_maturity_years


def H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """Valuation of European call option in H93 model via Lewis (2001)

    Parameter definition:
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial level of variance
    Returns
    =======
    call_value: float
        present value of European call option
    """
    int_value = quad(
        lambda u: H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0),
        0,
        np.inf,
        limit=250,
    )[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * int_value)
    return call_value

def H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """
    Fourier-based approach for Lewis (2001): Integration function.
    """
    char_func_value = H93_char_func(
        u - 1j * 0.5, T, r, kappa_v, theta_v, sigma_v, rho, v0
    )
    int_func_value = (
        1 / (u**2 + 0.25) * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    )
    return int_func_value


def H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach: characteristic function.
    Parameter definitions see function BCC_call_value."""
    c1 = kappa_v * theta_v
    c2 = -np.sqrt(
        (rho * sigma_v * u * 1j - kappa_v) ** 2 - sigma_v**2 * (-u * 1j - u**2)
    )
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) / (
        kappa_v - rho * sigma_v * u * 1j - c2
    )
    H1 = r * u * 1j * T + (c1 / sigma_v**2) * (
        (kappa_v - rho * sigma_v * u * 1j + c2) * T
        - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))
    )
    H2 = (
        (kappa_v - rho * sigma_v * u * 1j + c2)
        / sigma_v**2
        * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T)))
    )
    char_func_value = np.exp(H1 + H2 * v0)
    return char_func_value


def calibration_objective(params, data, S0, r):
    """Objective function to minimize (MSE)."""
    kappa, theta, sigma, rho, v0 = params
    
    # Impose Feller condition and parameter bounds
    if 2 * kappa * theta < sigma**2 or not (0 < kappa < 20 and 0 < theta < 1 and 0 < sigma < 2 and -1 < rho < 0 and 0 < v0 < 1):
        return np.inf

    model_prices = []
    for _, row in data.iterrows():
        K = row['Strike']
        T = row['T']
        
        # Price the call option
        call_price = H93_call_value(S0, K, T, r, kappa, theta, sigma, rho, v0)
        
        # If pricing fails, return a large error
        if call_price == np.inf:
            return np.inf
            
        # Use the market price of the option that is closer to the money
        if row['Put'] > row['Call']:
            # Use Put-Call Parity for the model put price
            put_price = call_price - S0 + K * np.exp(-r * T)
            model_prices.append(put_price)
        else:
            model_prices.append(call_price)
            
    # Use the corresponding market prices for the MSE calculation
    market_prices = np.where(data['Put'] > data['Call'], data['Put'], data['Call'])
    
    mse = np.mean((market_prices - np.array(model_prices))**2)
    
    # Optional: print progress
    # print(f"Params: [k:{kappa:.2f}, t:{theta:.2f}, s:{sigma:.2f}, r:{rho:.2f}, v:{v0:.2f}] | MSE: {mse:.4f}")
    return mse

if __name__ == '__main__':
    # --- Model & Market Constants ---
    S0 = 232.90
    R = 0.015
    TRADING_DAYS = 250
    
    # --- Data Loading ---
    # Get the directory where the script is located and build the path to the data file
    script_dir = Path(__file__).parent
    EXCEL_PATH = script_dir / 'MScFE 622_Stochastic Modeling_GWP1_Option data.xlsx'
    option_data, maturity = load_and_prepare_data(EXCEL_PATH, TRADING_DAYS)

    print(f"Loaded data for maturity: {maturity:.0f} days")
    print(f"option_data: {option_data}")
    # --- Calibration ---
    # Stage 1: Brute-force global search to find a good starting point
    print("\n--- Stage 1: Starting Brute-Force Global Search ---")
    # Define parameter ranges: (kappa, theta, sigma, rho, v0)
    # ranges = (
    #     (2.0, 4.0, 1.0),      # kappa: 3 points
    #     (0.015, 0.035, 0.01), # theta: 3 points (increased)
    #     (0.15, 0.25, 0.05),   # sigma: 3 points (decreased)  
    #     (-0.7, -0.3, 0.2),    # rho: 3 points
    #     (0.01, 0.02, 0.005)   # v0: 3 points
    # )
    
    ranges = (
        (2.5, 10.6, 5.0),  # kappa_v
        (0.01, 0.041, 0.01),  # theta_v
        (0.05, 0.251, 0.1),  # sigma_v
        (-0.75, 0.01, 0.25),  # rho
        (0.01, 0.031, 0.01),
    
        # (1.0, 3.0, 1.0),      # kappa: 3 points
        # (0.005, 0.015, 0.005), # theta: 3 points  
        # (0.2, 0.4, 0.1),      # sigma: 3 points
        # (-0.7, -0.3, 0.2),    # rho: 3 points
        # (0.005, 0.015, 0.005) # v0: 3 points
    )   
    
    initial_params_from_brute = brute(
        calibration_objective,
        ranges,
        args=(option_data, S0, R),
        finish=None
    )
    
    print("--- Brute-Force Search Complete ---")
    print(f"Best parameters from brute-force search: {initial_params_from_brute}")

    # Stage 2: Local refinement using the result from Stage 1
    print("\n--- Stage 2: Starting Local Refinement (Nelder-Mead) ---")
    result = minimize(
        calibration_objective,
        initial_params_from_brute,
        args=(option_data, S0, R),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'disp': True, 'adaptive': True},
        tol=1e-6
    )

    calibrated_params = result.x
    final_mse = result.fun
    
    print("\n--- Calibration Complete ---")
    print("Calibrated Heston Parameters:")
    print(f"  kappa (mean reversion speed): {calibrated_params[0]:.4f}")
    print(f"  theta (long-term variance):   {calibrated_params[1]:.4f}")
    print(f"  sigma (vol of vol):           {calibrated_params[2]:.4f}")
    print(f"  rho (correlation):            {calibrated_params[3]:.4f}")
    print(f"  v0 (initial variance):        {calibrated_params[4]:.4f}")
    print(f"Final MSE: {final_mse:.6f}")

    # --- Results Visualization ---
    final_model_prices = []
    for i, row in option_data.iterrows():
        # call_price = heston_price_lewis(S0, row['Strike'], row['T'], R, *calibrated_params)
        call_price = H93_call_value(S0, row['Strike'], row['T'], R, *calibrated_params)
        
        if row['Put'] > row['Call']:
            put_price = call_price - S0 + row['Strike'] * np.exp(-R * row['T'])
            final_model_prices.append(put_price)
        else:
            final_model_prices.append(call_price)

    option_data['Calibrated_Price'] = final_model_prices

    plt.figure(figsize=(12, 7))
    plt.scatter(option_data['Strike'], option_data['Call'], label='Market Call Prices', marker='x', color='blue')
    plt.scatter(option_data['Strike'], option_data['Put'], label='Market Put Prices', marker='.', color='green')
    
    # Calculate complete model curves for all strikes
    model_call_prices = []
    model_put_prices = []
    
    for _, row in option_data.iterrows():
        call_price = H93_call_value(S0, row['Strike'], row['T'], R, *calibrated_params)
        put_price = call_price - S0 + row['Strike'] * np.exp(-R * row['T'])
        model_call_prices.append(call_price)
        model_put_prices.append(put_price)
    
    # Plot complete model curves
    plt.plot(option_data['Strike'], model_call_prices, 'r--', label='Heston Model (Calls)', linewidth=2)
    plt.plot(option_data['Strike'], model_put_prices, 'r-', label='Heston Model (Puts)', linewidth=2)

    plt.title(f'Heston Model Calibration vs. Market Prices (Maturity: {maturity*TRADING_DAYS:.0f} days)')
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid(True)
    plot_path = script_dir / 'calibration_fit.png'
    plt.savefig(plot_path)
    print(f"\nPlot of the calibration fit saved to: {plot_path}")
