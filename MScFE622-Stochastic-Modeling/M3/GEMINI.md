# Team Member A - Step 2 Part B Task

## Task Overview
Team Member A will repeat the previous Task (a) in Step 2 using Carr-Madan (1999) approach to Bates (1996) model.

## Specific Task Requirements

### Step 2b: Carr-Madan Approach Implementation
- Implement the Carr-Madan (1999) approach to the Bates (1996) model
- **Key Change**: Use 60-day maturity instrument (instead of the maturity from Step 1)
- Follow all other instructions from Task (b) in Step 1

### Implementation Details
- **Model**: Bates (1996) - Heston model with jumps
- **Approach**: Carr-Madan (1999) Fast Fourier Transform method
- **Target Maturity**: 60-day maturity instrument
- **Calibration**: Follow the same calibration procedures as established in Step 1 Task (b)

### Key Components to Implement
1. **Carr-Madan FFT Method**: 
   - Fast Fourier Transform approach for option pricing
   - Efficient computation of option prices across strike range
   
2. **Bates Model Parameters**:
   - Heston parameters: v0, kappa, theta, sigma, rho
   - Jump parameters: jump intensity, jump size mean, jump size variance

3. **60-Day Maturity Focus**:
   - Extract and use 60-day maturity data from market data
   - Calibrate model specifically to this maturity bucket

### Reference Materials
- Carr, P. and Madan, D. (1999) - FFT approach
- Bates, D. (1996) - Heston model with jumps
- Follow calibration methodology from Step 1 Task (b)

### Expected Deliverables
- Carr-Madan FFT implementation for Bates model
- Model calibration to 60-day maturity options
- Comparison of results with previous approaches

