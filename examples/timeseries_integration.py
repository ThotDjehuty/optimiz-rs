"""
Time-Series Integration Helpers - Example Usage
===============================================

Demonstrates the 6 time-series utility functions for financial data analysis:
1. prepare_for_hmm_py: Feature engineering for regime detection
2. rolling_hurst_exponent_py: Mean-reversion detection
3. rolling_half_life_py: Pairs trading metrics
4. return_statistics_py: Risk analysis
5. create_lagged_features_py: ML feature creation
6. rolling_correlation_py: Correlation analysis

These helpers bridge OptimizR's optimization capabilities with time-series analysis,
particularly useful for regime-switching models and pairs trading strategies.
"""

import optimizr
import numpy as np


def example_prepare_for_hmm():
    """Example: Feature engineering for Hidden Markov Models"""
    print("\n=== Example 1: prepare_for_hmm_py ===")
    
    # Simulate stock prices
    prices = [100.0, 101.5, 99.8, 102.3, 103.7, 104.2, 103.1, 105.8, 107.2, 106.5]
    
    # Create feature matrix with 1 and 2-period lags
    features = optimizr.prepare_for_hmm_py(prices, [1, 2])
    
    print(f"Input: {len(prices)} price points")
    print(f"Output: {len(features)} rows x {len(features[0])} columns")
    print("\nFeature columns:")
    print("  [0] Simple returns")
    print("  [1] Log returns")
    print("  [2] Volatility proxy (squared returns)")
    print("  [3] Lagged returns (lag=1)")
    print("  [4] Lagged returns (lag=2)")
    print(f"\nFirst row: {[f'{x:.4f}' for x in features[0]]}")
    print("\n💡 Use this with OptimizR's HMM for regime detection!")


def example_rolling_hurst():
    """Example: Detecting mean-reversion with Hurst exponent"""
    print("\n=== Example 2: rolling_hurst_exponent_py ===")
    
    # Generate mean-reverting returns
    np.random.seed(42)
    returns = list(np.random.randn(20) * 0.02)
    
    # Compute rolling Hurst exponent
    window = 10
    hurst_values = optimizr.rolling_hurst_exponent_py(returns, window)
    
    print(f"Returns: {len(returns)} observations")
    print(f"Rolling Hurst (window={window}): {len(hurst_values)} values")
    print(f"\nHurst values: {[f'{h:.3f}' for h in hurst_values[:5]]}...")
    print("\nInterpretation:")
    print("  H < 0.5: Mean-reverting (good for pairs trading)")
    print("  H = 0.5: Random walk")
    print("  H > 0.5: Trending")
    
    avg_hurst = np.mean(hurst_values)
    if avg_hurst < 0.5:
        print(f"\n📊 Average H = {avg_hurst:.3f} → Mean-reverting behavior detected!")
    elif avg_hurst > 0.5:
        print(f"\n📊 Average H = {avg_hurst:.3f} → Trending behavior detected!")
    else:
        print(f"\n📊 Average H = {avg_hurst:.3f} → Random walk behavior")


def example_rolling_half_life():
    """Example: Mean-reversion speed for pairs trading"""
    print("\n=== Example 3: rolling_half_life_py ===")
    
    # Simulate spread between two cointegrated assets
    np.random.seed(42)
    spread = list(100 + np.cumsum(np.random.randn(30) * 0.5))
    
    # Compute rolling half-life
    window = 15
    half_lives = optimizr.rolling_half_life_py(spread, window)
    
    print(f"Spread: {len(spread)} observations")
    print(f"Rolling half-life (window={window}): {len(half_lives)} values")
    print(f"\nHalf-life values: {[f'{hl:.2f}' for hl in half_lives[:5]]}...")
    print("\nInterpretation:")
    print("  Lower half-life → Faster mean reversion")
    print("  Higher half-life → Slower mean reversion")
    
    avg_hl = np.mean(half_lives)
    print(f"\n📊 Average half-life: {avg_hl:.2f} periods")
    print(f"   → Spread reverts to mean in ~{avg_hl:.0f} periods on average")


def example_return_statistics():
    """Example: Comprehensive risk metrics"""
    print("\n=== Example 4: return_statistics_py ===")
    
    # Sample returns from a trading strategy
    returns = [0.02, -0.01, 0.015, 0.025, -0.005, 0.01, -0.02, 0.03, 0.005, -0.015]
    
    # Compute statistics
    mean, std, skew, kurt, sharpe = optimizr.return_statistics_py(returns)
    
    print(f"Returns: {len(returns)} observations")
    print("\nStatistics:")
    print(f"  Mean return:       {mean:.4f} ({mean*100:.2f}%)")
    print(f"  Volatility (std):  {std:.4f}")
    print(f"  Skewness:          {skew:.4f} {'(left-tailed)' if skew < 0 else '(right-tailed)'}")
    print(f"  Kurtosis:          {kurt:.4f} {'(fat tails)' if kurt > 0 else '(thin tails)'}")
    print(f"  Sharpe ratio:      {sharpe:.4f}")
    
    print("\n📊 Risk Assessment:")
    if sharpe > 2.0:
        print("   ✅ Excellent risk-adjusted returns")
    elif sharpe > 1.0:
        print("   ✓ Good risk-adjusted returns")
    else:
        print("   ⚠️ Moderate risk-adjusted returns")


def example_lagged_features():
    """Example: Create features for ML models"""
    print("\n=== Example 5: create_lagged_features_py ===")
    
    # Time series to predict
    returns = [0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.025, 0.01, -0.01, 0.02]
    
    # Create lagged feature matrix
    lags = [1, 2, 3]
    features = optimizr.create_lagged_features_py(returns, lags, include_original=True)
    
    print(f"Original series: {len(returns)} observations")
    print(f"Lagged features: {len(features)} rows x {len(features[0])} columns")
    print("\nFeature columns:")
    print(f"  [0] Original value (t)")
    print(f"  [1] Lag-1 (t-1)")
    print(f"  [2] Lag-2 (t-2)")
    print(f"  [3] Lag-3 (t-3)")
    print(f"\nFirst row: {[f'{x:.4f}' for x in features[0]]}")
    print("\n💡 Use this for ML prediction models (LSTM, Random Forest, etc.)")


def example_rolling_correlation():
    """Example: Pairs trading correlation analysis"""
    print("\n=== Example 6: rolling_correlation_py ===")
    
    # Two potentially cointegrated assets
    np.random.seed(42)
    asset1_returns = list(np.random.randn(25) * 0.02)
    asset2_returns = list(np.random.randn(25) * 0.02 + np.array(asset1_returns) * 0.6)
    
    # Compute rolling correlation
    window = 10
    correlations = optimizr.rolling_correlation_py(asset1_returns, asset2_returns, window)
    
    print(f"Asset 1 returns: {len(asset1_returns)} observations")
    print(f"Asset 2 returns: {len(asset2_returns)} observations")
    print(f"Rolling correlation (window={window}): {len(correlations)} values")
    print(f"\nCorrelation values: {[f'{c:.3f}' for c in correlations[:5]]}...")
    
    avg_corr = np.mean(correlations)
    print(f"\n📊 Average correlation: {avg_corr:.3f}")
    if avg_corr > 0.7:
        print("   → Strong positive correlation (good for pairs trading)")
    elif avg_corr > 0.3:
        print("   → Moderate correlation")
    else:
        print("   → Weak correlation (not ideal for pairs trading)")


def example_integrated_workflow():
    """Example: Complete pairs trading analysis workflow"""
    print("\n" + "="*70)
    print("=== Integrated Workflow: Pairs Trading Analysis ===")
    print("="*70)
    
    # Generate synthetic pair of assets
    np.random.seed(42)
    n = 50
    asset1 = list(100 + np.cumsum(np.random.randn(n) * 0.5))
    asset2 = list(100 + np.cumsum(np.random.randn(n) * 0.5 + 
                                   (np.array(asset1) - 100) * 0.6))
    
    # Compute spread
    spread = [a1 - a2 for a1, a2 in zip(asset1, asset2)]
    
    # Step 1: Check mean-reversion with Hurst exponent
    print("\n1. Mean-reversion check (Hurst exponent):")
    hurst_values = optimizr.rolling_hurst_exponent_py(spread, 20)
    avg_hurst = np.mean(hurst_values)
    print(f"   Average Hurst: {avg_hurst:.3f}")
    mean_reverting = avg_hurst < 0.5
    print(f"   Mean-reverting: {'✅ Yes' if mean_reverting else '❌ No'}")
    
    # Step 2: Estimate reversion speed
    print("\n2. Mean-reversion speed (half-life):")
    half_lives = optimizr.rolling_half_life_py(spread, 20)
    avg_hl = np.mean(half_lives)
    print(f"   Average half-life: {avg_hl:.2f} periods")
    print(f"   Reversion time: ~{avg_hl:.0f} periods")
    
    # Step 3: Check correlation stability
    print("\n3. Correlation stability:")
    returns1 = [(asset1[i] - asset1[i-1])/asset1[i-1] for i in range(1, len(asset1))]
    returns2 = [(asset2[i] - asset2[i-1])/asset2[i-1] for i in range(1, len(asset2))]
    correlations = optimizr.rolling_correlation_py(returns1, returns2, 15)
    avg_corr = np.mean(correlations)
    print(f"   Average correlation: {avg_corr:.3f}")
    print(f"   Correlation stability: {'✅ High' if avg_corr > 0.7 else '⚠️ Moderate' if avg_corr > 0.3 else '❌ Low'}")
    
    # Step 4: Risk metrics for spread returns
    print("\n4. Spread risk metrics:")
    spread_returns = [(spread[i] - spread[i-1]) for i in range(1, len(spread))]
    mean, std, skew, kurt, sharpe = optimizr.return_statistics_py(spread_returns)
    print(f"   Mean: {mean:.4f}, Volatility: {std:.4f}")
    print(f"   Sharpe: {sharpe:.3f}")
    
    # Final recommendation
    print("\n" + "="*70)
    print("📊 Trading Recommendation:")
    if mean_reverting and avg_hl < 20 and avg_corr > 0.5:
        print("✅ STRONG PAIR: Good candidate for pairs trading")
        print(f"   - Fast mean reversion ({avg_hl:.1f} periods)")
        print(f"   - Stable correlation ({avg_corr:.2f})")
        print(f"   - Predictable behavior (H={avg_hurst:.2f})")
    elif mean_reverting and avg_corr > 0.3:
        print("⚠️ MODERATE PAIR: Consider with caution")
        print(f"   - Mean reversion detected")
        print(f"   - Moderate correlation ({avg_corr:.2f})")
    else:
        print("❌ WEAK PAIR: Not recommended for pairs trading")
        print(f"   - Low correlation or trending behavior")
    print("="*70)


if __name__ == "__main__":
    print("=" * 70)
    print("OptimizR Time-Series Integration Helpers")
    print("=" * 70)
    
    # Run all examples
    example_prepare_for_hmm()
    example_rolling_hurst()
    example_rolling_half_life()
    example_return_statistics()
    example_lagged_features()
    example_rolling_correlation()
    
    # Integrated workflow
    example_integrated_workflow()
    
    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)
