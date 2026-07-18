import numpy as np
from scipy.stats import norm, t
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def fiducial_likelihood(returns, mu, sigma):
    """
    Compute the fiducial likelihood (structural equation) for a model.
    For a normal model, the fiducial likelihood is the same as the ordinary likelihood.
    """
    if sigma <= 0:
        return 0.0
    ll = -0.5 * np.sum(((returns - mu) / sigma)**2 + np.log(2 * np.pi * sigma**2))
    return np.exp(ll)

def generalized_fiducial_distribution(returns, n_samples=100):
    """
    Compute the generalized fiducial distribution for the parameters (mu, sigma).
    This is the fiducial analog of the posterior distribution.
    """
    if len(returns) < 5:
        return np.array([]), np.array([])
    # Compute sample statistics
    n = len(returns)
    mu_hat = np.mean(returns)
    sigma_hat = np.std(returns, ddof=1)
    # Generate fiducial samples
    mu_samples = []
    sigma_samples = []
    # Use pivot equations: Z = sqrt(n) * (mu_hat - mu) / sigma_hat ~ t(n-1)
    # And V = (n-1) * sigma_hat^2 / sigma^2 ~ chi^2(n-1)
    for _ in range(n_samples):
        # Sample from t-distribution
        t_sample = t.rvs(df=n-1)
        # Sample from chi-square
        chi2_sample = np.random.chisquare(df=n-1)
        # Back-transform to get parameter samples
        mu_sample = mu_hat - t_sample * sigma_hat / np.sqrt(n)
        sigma_sample = sigma_hat * np.sqrt((n-1) / chi2_sample)
        mu_samples.append(mu_sample)
        sigma_samples.append(sigma_sample)
    return np.array(mu_samples), np.array(sigma_samples)

def fiducial_confidence(returns, n_samples=100):
    """
    Compute the fiducial confidence that the mean return is positive.
    This is the proportion of fiducial samples with mu > 0.
    """
    if len(returns) < 5:
        return 0.5  # neutral confidence
    # Generate fiducial distribution
    mu_samples, sigma_samples = generalized_fiducial_distribution(returns, n_samples)
    if len(mu_samples) == 0:
        return 0.5
    # Compute confidence that mu > 0
    confidence = np.mean(mu_samples > 0)
    return confidence

def fiducial_score(returns, macro_df, n_samples=100):
    """
    Compute per-ETF fiducial confidence score.
    Higher score = more confident that the ETF has positive expected return.
    """
    if len(returns) < 10 or macro_df is None or len(macro_df) < 10:
        return 0.5
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 10:
        return 0.5
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)[-1]
    # Base confidence
    confidence = fiducial_confidence(returns, n_samples)
    # Adjust confidence based on macro factor
    # Higher macro factor -> more uncertainty, confidence moves toward 0.5
    adjusted_confidence = 0.5 + (confidence - 0.5) * (1 - macro_factor * 0.3)
    # Clip to [0,1]
    adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
    return float(adjusted_confidence)
