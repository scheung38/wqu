import pandas as pd
import numpy as np
from scipy.optimize import minimize, brute
from scipy.integrate import quad
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_prepare_data_60d(excel_path, trading_days):
    """Loads data from Excel and filters for 60-day maturity options."""
    print(f"Loading data from: {excel_path}")
    df = pd.read_excel(excel_path)
    
    # Pivot the table to get Call and Put prices in separate columns
    df = df.pivot_table(index=['Days to maturity', 'Strike'], columns='Type', values='Price').reset_index()
    df.rename(columns={'C': 'Call', 'P': 'Put'}, inplace=True)
    df.dropna(inplace=True)
    
    # Calculate time to maturity in years
    df['T'] = df['Days to maturity'] / trading_days
    
    # Filter for 60-day maturity
    target_maturity_days = 60
    target_maturity_years = target_maturity_days / trading_days
    
    print("Filtering for 60-day maturity options")
    filtered_df = df[df['Days to maturity'] == target_maturity_days].copy()
    
    if filtered_df.empty:
        raise ValueError(f"No options found with {target_maturity_days} days to maturity")
    
    # Calculate Mid_Price for the error function
    filtered_df['Mid_Price'] = (filtered_df['Call'] + filtered_df['Put']) / 2
    return filtered_df, target_maturity_years

def bates_char_func(u, T, r, kappa, theta, sigma, rho, v0, lam, mu_j, sig_j):
    """Bates (1996) characteristic function with jumps."""
    # Heston part
    d = np.sqrt((rho * sigma * u * 1j - kappa)**2 - sigma**2 * (-u * 1j - u**2))
    g = (kappa - rho * sigma * u * 1j - d) / (kappa - rho * sigma * u * 1j + d)
    
    C_heston = (r * u * 1j * T + (kappa * theta) / (sigma**2) * 
                ((kappa - rho * sigma * u * 1j - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
                
    D_heston = ((kappa - rho * sigma * u * 1j - d) / (sigma**2) * 
                ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))))
    
    # Jump part - Merton jump component
    omega = np.exp(mu_j + 0.5 * sig_j**2) - 1  # compensating drift
    jump_cf = lam * T * (np.exp(1j * u * mu_j - 0.5 * sig_j**2 * u**2) - 1 - 1j * u * omega)
    
    return np.exp(C_heston + D_heston * v0 + jump_cf)

def carr_madan_fft(S0, K_array, T, r, kappa, theta, sigma, rho, v0, lam, mu_j, sig_j, 
                   alpha=1.5, N=2**15, eta=0.15):
    """Carr-Madan (1999) FFT method for option pricing."""
    
    # FFT parameters
    lambda_val = 2 * np.pi / (N * eta)
    b = N * lambda_val / 2
    
    # Strike and log-moneyness grid
    k_u = np.arange(N) * lambda_val - b  # log-strike grid
    strike_grid = S0 * np.exp(k_u)
    
    # Frequency grid
    v_j = np.arange(N) * eta
    
    # Carr-Madan integrand
    def psi(v):
        u = v - (alpha + 1) * 1j
        char_func_val = bates_char_func(u, T, r, kappa, theta, sigma, rho, v0, lam, mu_j, sig_j)
        return np.exp(-r * T) * char_func_val / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
    
    # Evaluate psi at frequency grid points
    psi_values = np.array([psi(v) for v in v_j])
    
    # Apply FFT
    fft_input = np.exp(1j * b * v_j) * psi_values * eta
    fft_result = np.fft.fft(fft_input)
    
    # Extract call prices
    call_prices = S0**(-alpha) * np.exp(-alpha * k_u) * fft_result.real / np.pi
    
    # Interpolate to desired strikes
    call_prices_interp = np.interp(K_array, strike_grid, call_prices)
    
    return call_prices_interp

def calibration_objective_bates(params, data, S0, r, T):
    """Objective function for Bates model calibration."""
    kappa, theta, sigma, rho, v0, lam, mu_j, sig_j = params
    
    # Parameter bounds and constraints
    if not (0.01 < kappa < 10 and 0.001 < theta < 1 and 0.01 < sigma < 2 and 
            -0.99 < rho < 0.99 and 0.001 < v0 < 1 and 0 <= lam < 10 and
            -2 < mu_j < 2 and 0.01 < sig_j < 2):
        return np.inf
    
    # Feller condition
    if 2 * kappa * theta < sigma**2:
        return np.inf
    
    try:
        # Get strikes
        strikes = data['Strike'].values
        
        # Price options using Carr-Madan FFT
        model_call_prices = carr_madan_fft(S0, strikes, T, r, kappa, theta, sigma, 
                                         rho, v0, lam, mu_j, sig_j)
        
        # Calculate model put prices using put-call parity
        model_put_prices = model_call_prices - S0 + strikes * np.exp(-r * T)
        
        # Use both calls and puts for better calibration
        call_errors = (data['Call'].values - model_call_prices)**2
        put_errors = (data['Put'].values - model_put_prices)**2
        
        # Weight by moneyness - out-of-money options are more liquid
        strikes = data['Strike'].values
        is_otm_call = strikes > S0
        is_otm_put = strikes < S0
        
        total_error = 0
        total_weight = 0
        
        for i in range(len(strikes)):
            call_weight = 1.0 if is_otm_call[i] else 0.5
            put_weight = 1.0 if is_otm_put[i] else 0.5
            
            total_error += call_weight * call_errors[i] + put_weight * put_errors[i]
            total_weight += call_weight + put_weight
        
        mse = total_error / total_weight
        
        # Check for invalid prices
        if (np.any(np.isnan(model_call_prices)) or np.any(np.isinf(model_call_prices)) or
            np.any(np.isnan(model_put_prices)) or np.any(np.isinf(model_put_prices))):
            return np.inf
            
        return mse
        
    except Exception as e:
        print(f"Error in pricing: {e}")
        return np.inf

if __name__ == '__main__':
    # Market constants
    S0 = 232.90
    r = 0.015
    TRADING_DAYS = 250
    
    # Load 60-day maturity data
    script_dir = Path(__file__).parent
    EXCEL_PATH = script_dir / 'MScFE 622_Stochastic Modeling_GWP1_Option data.xlsx'
    
    try:
        option_data_60d, T_60d = load_and_prepare_data_60d(EXCEL_PATH, TRADING_DAYS)
        print(f"\nLoaded {len(option_data_60d)} options with 60-day maturity")
        print(f"Time to maturity: {T_60d:.4f} years")
        print(f"Strike range: {option_data_60d['Strike'].min():.1f} to {option_data_60d['Strike'].max():.1f}")
        print("\nOption data:")
        print(option_data_60d[['Strike', 'Call', 'Put']])
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Available maturities in the data:")
        df_temp = pd.read_excel(EXCEL_PATH)
        print(sorted(df_temp['Days to maturity'].unique()))
        exit(1)
    
    # Initial parameter guess for Bates model
    # [kappa, theta, sigma, rho, v0, lambda, mu_j, sig_j]
    initial_guess = [2.5, 0.04, 0.3, -0.7, 0.04, 0.5, -0.05, 0.1]
    
    print("\n--- Starting Bates Model Calibration ---")
    print(f"Initial parameters: {initial_guess}")
    
    # Define parameter bounds for optimization
    bounds = [
        (0.5, 6.0),    # kappa: mean reversion speed
        (0.01, 0.08),  # theta: long-term variance
        (0.1, 0.8),    # sigma: vol of vol
        (-0.9, -0.1),  # rho: correlation (typically negative)
        (0.01, 0.08),  # v0: initial variance
        (0.0, 2.0),    # lambda: jump intensity (lower range)
        (-0.2, 0.1),   # mu_j: jump mean (tighter range)
        (0.05, 0.3)    # sig_j: jump volatility (tighter range)
    ]
    
    # Two-stage optimization: brute force + local refinement
    try:
        print("Stage 1: Global search using brute force...")
        
        # Coarse grid for brute force search (reduced ranges for key parameters)
        brute_ranges = [
            slice(1.0, 4.0, 0.5),     # kappa
            slice(0.02, 0.06, 0.01),  # theta  
            slice(0.2, 0.6, 0.1),     # sigma
            slice(-0.8, -0.3, 0.1),   # rho
            slice(0.02, 0.06, 0.01),  # v0
            slice(0.0, 1.0, 0.2),     # lambda
            slice(-0.1, 0.05, 0.05),  # mu_j
            slice(0.08, 0.2, 0.04)    # sig_j
        ]
        
        brute_result = brute(
            calibration_objective_bates,
            brute_ranges,
            args=(option_data_60d, S0, r, T_60d),
            full_output=True,
            finish=None  # Don't run local optimization yet
        )
        
        best_brute_params = brute_result[0]
        best_brute_score = brute_result[1]
        
        print(f"Best brute force result: MSE = {best_brute_score:.6f}")
        print(f"Best brute force params: {best_brute_params}")
        
        print("\nStage 2: Local refinement...")
        result = minimize(
            calibration_objective_bates,
            best_brute_params,
            args=(option_data_60d, S0, r, T_60d),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': True}
        )
        
        if result.success:
            calibrated_params = result.x
            final_mse = result.fun
            
            print("\n--- Bates Model Calibration Complete ---")
            print(f"Optimization successful: {result.success}")
            print(f"Final MSE: {final_mse:.6f}")
            print("\nCalibrated Bates Parameters:")
            print(f"  kappa (mean reversion):     {calibrated_params[0]:.4f}")
            print(f"  theta (long-term variance): {calibrated_params[1]:.4f} ({np.sqrt(calibrated_params[1])*100:.1f}% vol)")
            print(f"  sigma (vol of vol):         {calibrated_params[2]:.4f}")
            print(f"  rho (correlation):          {calibrated_params[3]:.4f}")
            print(f"  v0 (initial variance):      {calibrated_params[4]:.4f} ({np.sqrt(calibrated_params[4])*100:.1f}% vol)")
            print(f"  lambda (jump intensity):    {calibrated_params[5]:.4f}")
            print(f"  mu_j (jump mean):           {calibrated_params[6]:.4f}")
            print(f"  sig_j (jump volatility):    {calibrated_params[7]:.4f}")
            
        else:
            print(f"\nOptimization failed: {result.message}")
            calibrated_params = result.x
            final_mse = result.fun
            
    except Exception as e:
        print(f"Error during optimization: {e}")
        exit(1)
    
    # Generate model prices for visualization
    strikes = option_data_60d['Strike'].values
    
    try:
        model_call_prices = carr_madan_fft(S0, strikes, T_60d, r, *calibrated_params)
        model_put_prices = model_call_prices - S0 + strikes * np.exp(-r * T_60d)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot market data
        plt.scatter(option_data_60d['Strike'], option_data_60d['Call'], 
                   label='Market Call Prices', marker='x', color='blue', s=80)
        plt.scatter(option_data_60d['Strike'], option_data_60d['Put'], 
                   label='Market Put Prices', marker='o', color='green', s=80)
        
        # Plot model curves
        plt.plot(strikes, model_call_prices, 'r--', label='Bates Model (Calls)', linewidth=2)
        plt.plot(strikes, model_put_prices, 'r-', label='Bates Model (Puts)', linewidth=2)
        
        plt.title('Bates Model Calibration vs. Market Prices (60-day maturity)\nCarr-Madan FFT Approach')
        plt.xlabel('Strike Price')
        plt.ylabel('Option Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = script_dir / 'bates_calibration_60d.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        
        # Calculate and display fit quality metrics
        market_calls = option_data_60d['Call'].values
        market_puts = option_data_60d['Put'].values
        
        call_rmse = np.sqrt(np.mean((market_calls - model_call_prices)**2))
        put_rmse = np.sqrt(np.mean((market_puts - model_put_prices)**2))
        
        print("\nFit Quality Metrics:")
        print(f"Call RMSE: {call_rmse:.4f}")
        print(f"Put RMSE:  {put_rmse:.4f}")
        print(f"Overall RMSE: {np.sqrt((call_rmse**2 + put_rmse**2)/2):.4f}")
        
    except Exception as e:
        print(f"Error in final pricing or visualization: {e}")