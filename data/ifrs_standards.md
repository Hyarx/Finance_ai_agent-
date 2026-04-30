# Market Risk — Key Concepts

## GARCH Model
GARCH(1,1) captures volatility clustering in financial returns.
sigma_t^2 = omega + alpha * epsilon_(t-1)^2 + beta * sigma_(t-1)^2
- alpha + beta close to 1: high volatility persistence
- alpha high: strong reaction to market shocks
- beta high: slow mean reversion of volatility

## Value at Risk (VaR)
VaR at confidence level alpha: maximum loss not exceeded
with probability alpha over a given time horizon.
- VaR 99% 1-day: regulatory standard (Basel II)
- VaR 95% 1-day: common internal risk measure

## Expected Shortfall (ES)
ES at level alpha: average loss in the worst (1-alpha)% scenarios.
- ES 97.5%: Basel III standard (replaces VaR 99%)
- Always more conservative than VaR
- Better captures tail risk

## Regulatory Standards
- Basel II: VaR 99% for market risk capital
- Basel III: shift to ES 97.5%
- Capital charge = 3 x ES 97.5% (minimum multiplier)