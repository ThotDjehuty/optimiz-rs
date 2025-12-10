//! Backtesting Engine
//! ==================
//!
//! Backtest optimal switching strategies with transaction costs

use crate::optimal_control::{OptimalControlError, Result};

/// Trade type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeType {
    Buy,
    Sell,
    CloseLong,
    CloseShort,
}

/// Individual trade
#[derive(Debug, Clone)]
pub struct Trade {
    pub timestamp: usize,
    pub trade_type: TradeType,
    pub price: f64,
    pub position: i32,
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total return
    pub total_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Win rate
    pub win_rate: f64,
    /// PnL curve
    pub pnl: Vec<f64>,
    /// All trades
    pub trades: Vec<Trade>,
    /// Average holding period
    pub avg_holding_period: f64,
    /// Profit factor
    pub profit_factor: f64,
}

/// Backtest optimal switching strategy
///
/// Rules:
/// - Buy when spread < lower_bound
/// - Sell when spread > upper_bound  
/// - Exit when spread crosses mean (theta)
pub fn backtest_optimal_switching(
    spread: &[f64],
    lower_bound: f64,
    upper_bound: f64,
    transaction_cost: f64,
) -> Result<BacktestResult> {
    if spread.is_empty() {
        return Err(OptimalControlError::InsufficientData(1));
    }
    
    let theta = spread.iter().sum::<f64>() / spread.len() as f64;
    
    let mut position: i32 = 0; // -1 (short), 0 (flat), +1 (long)
    let mut cash = 0.0;
    let mut pnl = Vec::with_capacity(spread.len());
    let mut trades = Vec::new();
    
    for (t, &z) in spread.iter().enumerate() {
        let current_pnl = cash + position as f64 * z;
        pnl.push(current_pnl);
        
        // Entry signals
        if position == 0 {
            if z < lower_bound {
                // Buy spread (expect mean-reversion up)
                position = 1;
                cash -= z * (1.0 + transaction_cost);
                trades.push(Trade {
                    timestamp: t,
                    trade_type: TradeType::Buy,
                    price: z,
                    position,
                });
            } else if z > upper_bound {
                // Short spread (expect mean-reversion down)
                position = -1;
                cash += z * (1.0 - transaction_cost);
                trades.push(Trade {
                    timestamp: t,
                    trade_type: TradeType::Sell,
                    price: z,
                    position,
                });
            }
        }
        // Exit signals (cross mean)
        else if position == 1 && z > theta {
            // Close long
            cash += z * (1.0 - transaction_cost);
            position = 0;
            trades.push(Trade {
                timestamp: t,
                trade_type: TradeType::CloseLong,
                price: z,
                position,
            });
        } else if position == -1 && z < theta {
            // Close short
            cash -= z * (1.0 + transaction_cost);
            position = 0;
            trades.push(Trade {
                timestamp: t,
                trade_type: TradeType::CloseShort,
                price: z,
                position,
            });
        }
    }
    
    // Close any open position at end
    if position != 0 {
        let final_price = spread[spread.len() - 1];
        let tc = transaction_cost * position.signum() as f64;
        cash += position as f64 * final_price * (1.0 - tc);
        position = 0;
    }
    
    // Calculate metrics
    let total_return = pnl.last().copied().unwrap_or(0.0);
    
    // Returns
    let mut returns = Vec::with_capacity(pnl.len() - 1);
    for i in 1..pnl.len() {
        let prev = pnl[i - 1].abs() + 1e-10;
        returns.push((pnl[i] - pnl[i - 1]) / prev);
    }
    
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>() / returns.len() as f64;
    let std_return = variance.sqrt();
    
    let sharpe_ratio = if std_return > 1e-10 {
        mean_return / std_return * 252.0f64.sqrt()
    } else {
        0.0
    };
    
    // Maximum drawdown
    let mut cummax = pnl[0];
    let mut max_dd = 0.0;
    for &p in &pnl {
        cummax = cummax.max(p);
        let dd = (p - cummax) / (cummax.abs() + 1e-10);
        max_dd = max_dd.min(dd);
    }
    
    // Win rate
    let mut wins = 0;
    let mut losses = 0;
    let mut i = 0;
    while i + 1 < trades.len() {
        if trades[i].trade_type == TradeType::Buy || trades[i].trade_type == TradeType::Sell {
            if i + 1 < trades.len() {
                let entry = trades[i].price;
                let exit = trades[i + 1].price;
                let pnl_trade = if trades[i].trade_type == TradeType::Buy {
                    exit - entry
                } else {
                    entry - exit
                };
                
                if pnl_trade > 0.0 {
                    wins += 1;
                } else {
                    losses += 1;
                }
                i += 2;
            } else {
                break;
            }
        } else {
            i += 1;
        }
    }
    
    let win_rate = if wins + losses > 0 {
        wins as f64 / (wins + losses) as f64
    } else {
        0.0
    };
    
    // Average holding period
    let mut holding_periods = Vec::new();
    let mut i = 0;
    while i + 1 < trades.len() {
        if trades[i].trade_type == TradeType::Buy || trades[i].trade_type == TradeType::Sell {
            if i + 1 < trades.len() {
                let period = trades[i + 1].timestamp - trades[i].timestamp;
                holding_periods.push(period as f64);
                i += 2;
            } else {
                break;
            }
        } else {
            i += 1;
        }
    }
    
    let avg_holding_period = if !holding_periods.is_empty() {
        holding_periods.iter().sum::<f64>() / holding_periods.len() as f64
    } else {
        0.0
    };
    
    // Profit factor
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;
    for i in 1..pnl.len() {
        let daily_pnl = pnl[i] - pnl[i - 1];
        if daily_pnl > 0.0 {
            gross_profit += daily_pnl;
        } else {
            gross_loss += daily_pnl.abs();
        }
    }
    
    let profit_factor = if gross_loss > 1e-10 {
        gross_profit / gross_loss
    } else {
        gross_profit
    };
    
    Ok(BacktestResult {
        total_return,
        sharpe_ratio,
        max_drawdown: max_dd,
        num_trades: trades.len(),
        win_rate,
        pnl,
        trades,
        avg_holding_period,
        profit_factor,
    })
}

/// Backtest simple mean-reversion strategy
pub fn backtest_mean_reversion(
    spread: &[f64],
    z_score_entry: f64,
    z_score_exit: f64,
    transaction_cost: f64,
) -> Result<BacktestResult> {
    if spread.len() < 20 {
        return Err(OptimalControlError::InsufficientData(20));
    }
    
    // Calculate rolling mean and std
    let window = 20;
    let mut positions = vec![0i32; spread.len()];
    let mut signals = Vec::new();
    
    for i in window..spread.len() {
        let window_data = &spread[i - window..i];
        let mean = window_data.iter().sum::<f64>() / window as f64;
        let variance = window_data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / window as f64;
        let std = variance.sqrt();
        
        if std < 1e-10 {
            continue;
        }
        
        let z = (spread[i] - mean) / std;
        
        if z < -z_score_entry {
            signals.push((i, TradeType::Buy, spread[i]));
        } else if z > z_score_entry {
            signals.push((i, TradeType::Sell, spread[i]));
        } else if z.abs() < z_score_exit {
            signals.push((i, TradeType::CloseLong, spread[i]));
        }
    }
    
    // Convert to BacktestResult format
    let mean_spread = spread.iter().sum::<f64>() / spread.len() as f64;
    let std_spread = (spread.iter()
        .map(|x| (x - mean_spread).powi(2))
        .sum::<f64>() / spread.len() as f64)
        .sqrt();
    
    let lower_bound = mean_spread - z_score_entry * std_spread;
    let upper_bound = mean_spread + z_score_entry * std_spread;
    
    backtest_optimal_switching(spread, lower_bound, upper_bound, transaction_cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_backtest_optimal_switching() {
        // Simple mean-reverting spread
        let spread = vec![
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0,
            1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0,
        ];
        
        let result = backtest_optimal_switching(&spread, -1.5, 1.5, 0.001).unwrap();
        
        assert!(result.num_trades > 0);
        assert!(result.pnl.len() == spread.len());
    }
}
