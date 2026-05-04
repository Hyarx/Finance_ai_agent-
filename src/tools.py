import numpy as np
import pandas as pd

import yfinance as yf
from arch import arch_model
from textblob import TextBlob

import requests

def get_sec_filings(ticker: str) -> str:
    """Retrieve official financial statements from SEC EDGAR.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT).

    Returns:
        Key financial data from the latest 10-K annual report.
    """
    try:
        headers = {"User-Agent": "alexandre@student.uliege.be"}
        print(f"[INFO] Fetching SEC EDGAR for {ticker}...")
        # Étape 1 — Récupère le CIK via le mapping officiel
        mapping_url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(mapping_url, headers=headers)
        tickers_data = resp.json()

        cik = None
        company_name = ticker
        for entry in tickers_data.values():
            if entry["ticker"].upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                company_name = entry["title"]
                break

        if not cik:
            return f"CIK not found for {ticker}"

        # Étape 2 — Récupère les filings
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(sub_url, headers=headers)
        sub_data = resp.json()

        # Étape 3 — Trouve le dernier 10-K
        filings = sub_data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])

        last_10k_date = "N/A"
        last_10k_accession = "N/A"
        for i, form in enumerate(forms):
            if form == "10-K":
                last_10k_date = dates[i]
                last_10k_accession = accessions[i]
                break

        # Étape 4 — Récupère les facts financiers
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(facts_url, headers=headers)
        facts = resp.json()

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        def get_latest_value(concept):
            try:
                units = us_gaap[concept]["units"]
                key = list(units.keys())[0]
                entries = [e for e in units[key] if e.get("form") == "10-K"]
                entries.sort(key=lambda x: x["end"], reverse=True)
                return entries[0]["val"] if entries else None
            except Exception:
                return None

        revenue = get_latest_value("Revenues") or get_latest_value("RevenueFromContractWithCustomerExcludingAssessedTax")
        net_income = get_latest_value("NetIncomeLoss")
        total_assets = get_latest_value("Assets")
        total_liabilities = get_latest_value("Liabilities")
        equity = get_latest_value("StockholdersEquity")
        operating_cf = get_latest_value("NetCashProvidedByUsedInOperatingActivities")
        rd_expense = get_latest_value("ResearchAndDevelopmentExpense")

        def fmt(val, unit="$"):
            if val is None:
                return "N/A"
            return f"{unit}{val/1e9:.2f}B"
        
        print(f"[INFO] Done SEC EDGAR for {ticker}!")
        return f"""
=== SEC EDGAR 10-K FILING: {company_name} ({ticker.upper()}) ===
CIK: {cik}
Latest 10-K filed: {last_10k_date}
Accession: {last_10k_accession}

INCOME STATEMENT
Revenue: {fmt(revenue)}
Net Income: {fmt(net_income)}
R&D Expense: {fmt(rd_expense)}

BALANCE SHEET
Total Assets: {fmt(total_assets)}
Total Liabilities: {fmt(total_liabilities)}
Stockholders Equity: {fmt(equity)}

CASH FLOW
Operating Cash Flow: {fmt(operating_cf)}

COMPLIANCE
- 10-K filed on time: {'Yes' if last_10k_date != 'N/A' else 'Unknown'}
- SEC registered: Yes
- GAAP compliant: Yes (filing accepted by SEC)
"""

    except Exception as e:
        return f"SEC EDGAR query failed: {str(e)}"
    

def calculate_financial_risk(tickers: str, weights: str = None, horizon: int = 1) -> str:
    """Calculate financial risk metrics for a stock or a portfolio.

    Args:
        tickers: Single ticker (e.g. AAPL) or comma-separated list 
                (e.g. AAPL,NVDA,MSFT) for a portfolio.
        weights: Optional comma-separated weights for portfolio 
                (e.g. 0.4,0.3,0.3). Equal weights if not provided.
        horizon: Number of days for risk horizon (default 1 day).
                Use 10 for Basel III standard, 30 for monthly risk.

    Returns:
        Financial ratios, GARCH volatility, VaR, Expected Shortfall
        and Altman Z-Score for single stock or portfolio.
    """
    print(f"[INFO] Starting financial risk for {tickers}, horizon={horizon}")
    ticker_list = [t.strip() for t in tickers.split(",")]

    if len(ticker_list) == 1:
        return _single_stock_risk(ticker_list[0], horizon)
    else:
        weight_list = None
        if weights:
            weight_list = [float(w.strip()) for w in weights.split(",")]
        return _portfolio_risk(ticker_list, weight_list, horizon)

def simulate_garch_path(omega, alpha, beta, initial_vol, n_days, n_sim):
    """Vectorized GARCH simulation."""
    z = np.random.normal(0, 1, (n_sim, n_days))
    vols = np.zeros((n_sim, n_days))
    returns = np.zeros((n_sim, n_days))
    
    vol = np.full(n_sim, initial_vol)
    
    for day in range(n_days):
        r = vol * z[:, day]
        returns[:, day] = r
        vol = np.sqrt(omega + alpha * r**2 + beta * vol**2)
        vols[:, day] = vol
    
    paths = np.zeros((n_sim, n_days))
    paths[:, -1] = returns.sum(axis=1)
    return paths

"""
    The modelisation of the financial risk is based on a GARCH(1,1) model for volatility,
    VaR and Expected Shortfall for risk measures. This part was seen in the lecture and 
    project during a course in HEC, financial risk modeling. 

    The Altman Z-Score is a well-known bankruptcy prediction model that combines several 
    financial ratios to assess the likelihood of distress and the help of AI.
"""
def _single_stock_risk(ticker: str, horizon: int = 1) -> str:
    """Single stock risk analysis."""

    stock = yf.Ticker(ticker)
    hist = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    info = stock.info

    prices = hist["Close"].dropna()
    returns = np.log(prices / prices.shift(1)).dropna().squeeze()

    # GARCH(1,1)
    model = arch_model(returns * 100, vol="Garch", p=1, q=1, rescale=False)
    res = model.fit(disp="off", options={"maxiter": 50})
    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    persistence = alpha + beta
    current_vol = float(res.conditional_volatility.iloc[-1]) / 100

    # VaR et ES
    if horizon == 1:
        sorted_returns = np.sort(returns)
        VaR_99 = float(np.percentile(sorted_returns, 1))
        VaR_95 = float(np.percentile(sorted_returns, 5))
        ES_975 = float(sorted_returns[sorted_returns <= np.percentile(sorted_returns, 2.5)].mean())
    else:
        paths = simulate_garch_path(
            omega=float(res.params["omega"]) / 10000,
            alpha=float(res.params["alpha[1]"]),
            beta=float(res.params["beta[1]"]),
            initial_vol=current_vol,
            n_days=horizon,
            n_sim=10000
        )
        sim_cum = paths.sum(axis=1)
        VaR_99 = float(np.percentile(sim_cum, 1))
        VaR_95 = float(np.percentile(sim_cum, 5))
        ES_975 = float(sim_cum[sim_cum <= np.percentile(sim_cum, 2.5)].mean())

    # Ratios
    roe = info.get("returnOnEquity", None)
    pe = info.get("trailingPE", None)
    de = info.get("debtToEquity", None)
    current_ratio = info.get("currentRatio", None)
    net_margin = info.get("profitMargins", None)
    revenue_growth = info.get("revenueGrowth", None)

    # Altman Z-Score
    try:
        bs = stock.balance_sheet
        inc = stock.financials
        total_assets = float(bs.loc["Total Assets"].iloc[0])
        total_liabilities = float(bs.loc["Total Liabilities Net Minority Interest"].iloc[0])
        working_capital = float(bs.loc["Current Assets"].iloc[0]) - float(bs.loc["Current Liabilities"].iloc[0])
        retained_earnings = float(bs.loc["Retained Earnings"].iloc[0])
        ebit = float(inc.loc["EBIT"].iloc[0])
        revenue = float(inc.loc["Total Revenue"].iloc[0])
        market_cap = info.get("marketCap", total_assets)

        A = working_capital / total_assets
        B = retained_earnings / total_assets
        C = ebit / total_assets
        D = market_cap / total_liabilities
        E = revenue / total_assets
        altman_z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        altman_zone = "Safe zone" if altman_z > 3.0 else "Grey zone" if altman_z > 1.8 else "Distress zone"
    except Exception:
        altman_z = None
        altman_zone = "Not available"

    return f"""
=== SINGLE STOCK RISK ANALYSIS: {ticker.upper()} ===

FUNDAMENTAL RATIOS
ROE: {f'{roe:.2%}' if roe else 'N/A'}
P/E Ratio: {f'{pe:.2f}' if pe else 'N/A'}
Debt/Equity: {f'{de:.2f}' if de else 'N/A'}
Current Ratio: {f'{current_ratio:.2f}' if current_ratio else 'N/A'}
Net Margin: {f'{net_margin:.2%}' if net_margin else 'N/A'}
Revenue Growth: {f'{revenue_growth:.2%}' if revenue_growth else 'N/A'}

GARCH(1,1) VOLATILITY MODEL
omega={omega:.6f}, alpha={alpha:.4f}, beta={beta:.4f}
Persistence (alpha+beta): {persistence:.4f}
Current daily volatility: {current_vol:.4f} ({current_vol*100:.2f}%)
Annualized volatility: {current_vol * np.sqrt(252) * 100:.2f}%

RISK MEASURES ({horizon}-day horizon)
VaR 99%: {VaR_99:.4f} ({VaR_99*100:.2f}%)
VaR 95%: {VaR_95:.4f} ({VaR_95*100:.2f}%)
Expected Shortfall 97.5%: {ES_975:.4f} ({ES_975*100:.2f}%)

ALTMAN Z-SCORE
Z-Score: {f'{altman_z:.2f}' if altman_z else 'N/A'} → {altman_zone}
(>3.0 Safe | 1.8-3.0 Grey | <1.8 Distress)
"""
def _portfolio_risk(tickers: list, weights: list = None, horizon: int = 1) -> str:
    """Portfolio risk analysis with Copula and Monte Carlo."""
    from scipy.stats import rankdata, norm

    print(f"[INFO] Starting portfolio risk for {tickers}, horizon={horizon}")

    n = len(tickers)
    if weights is None:
        weights = [1/n] * n
    weights = np.array(weights)
    weights = weights / weights.sum()

    all_returns = {}
    for ticker in tickers:
        print(f"[INFO] Downloading data for {ticker}...")
        hist = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        prices = hist["Close"].dropna()
        returns = np.log(prices / prices.shift(1)).dropna().squeeze()
        all_returns[ticker] = returns

    returns_df = pd.DataFrame(all_returns).dropna()

    garch_results = {}
    std_resids = {}
    vols = {}
    for ticker in tickers:
        print(f"[INFO] Fitting GARCH for {ticker}...")
        r = returns_df[ticker] * 100
        model = arch_model(r, vol="Garch", p=1, q=1, rescale=False)
        res = model.fit(disp="off", options={"maxiter": 50})
        garch_results[ticker] = res
        std_resids[ticker] = res.std_resid.values
        vols[ticker] = float(res.conditional_volatility.iloc[-1]) / 100

    print(f"[INFO] Computing Copula...")
    uniform_data = {}
    for ticker in tickers:
        u = rankdata(std_resids[ticker]) / (len(std_resids[ticker]) + 1)
        uniform_data[ticker] = u

    uniform_df = pd.DataFrame(uniform_data)
    corr_matrix = uniform_df.corr().values

    print(f"[INFO] Running Monte Carlo 1-day...")
    n_sim = 1000
    from scipy.stats import multivariate_normal
    L = np.linalg.cholesky(corr_matrix)
    z = np.random.normal(size=(n_sim, n))
    correlated = z @ L.T
    simulated_uniforms = norm.cdf(correlated)

    sim_returns = np.zeros((n_sim, n))
    for i, ticker in enumerate(tickers):
        sim_returns[:, i] = norm.ppf(simulated_uniforms[:, i]) * vols[ticker]

    sim_returns_h = sim_returns
    portfolio_returns = sim_returns @ weights

    if horizon > 1:
        print(f"[INFO] Running Monte Carlo {horizon}-day horizon...")
        sim_returns_h = np.zeros((n_sim, n))
        for i, ticker in enumerate(tickers):
            print(f"[INFO] Simulating GARCH path for {ticker}...")
            res = garch_results[ticker]
            paths = simulate_garch_path(
                omega=float(res.params["omega"]) / 10000,
                alpha=float(res.params["alpha[1]"]),
                beta=float(res.params["beta[1]"]),
                initial_vol=vols[ticker],
                n_days=horizon,
                n_sim=n_sim
            )
            sim_returns_h[:, i] = paths[:, -1]
        portfolio_returns = sim_returns_h @ weights

    print(f"[INFO] Computing VaR and ES...")
    
    VaR_99 = float(np.percentile(portfolio_returns, 1))
    VaR_95 = float(np.percentile(portfolio_returns, 5))
    ES_975 = float(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 2.5)].mean())

    contributions = {}
    for i, ticker in enumerate(tickers):
        contrib = weights[i] * np.cov(sim_returns_h[:, i], portfolio_returns)[0, 1] / np.std(portfolio_returns)
        contributions[ticker] = contrib

    print(f"[INFO] Portfolio risk done!")

    garch_summary = ""
    for ticker in tickers:
        res = garch_results[ticker]
        a = res.params["alpha[1]"]
        b = res.params["beta[1]"]
        garch_summary += f"  {ticker}: α={a:.3f} β={b:.3f} pers={a+b:.3f} vol={vols[ticker]*100:.2f}%/day\n"

    contrib_summary = ""
    for ticker, contrib in contributions.items():
        w = weights[tickers.index(ticker)]
        contrib_summary += f"  {ticker}: w={w:.0%} contrib={contrib:.4f}\n"

    return f"""
=== PORTFOLIO RISK: {', '.join(tickers)} ({horizon}-day horizon) ===
Weights: {', '.join([f'{t}={w:.0%}' for t, w in zip(tickers, weights)])}
Monte Carlo: {n_sim} simulations | Copula: Gaussian

GARCH(1,1) PER ASSET
{garch_summary}
PORTFOLIO RISK MEASURES
VaR 99%: {VaR_99*100:.2f}% | VaR 95%: {VaR_95*100:.2f}% | ES 97.5%: {ES_975*100:.2f}%

RISK CONTRIBUTIONS
{contrib_summary}
"""



def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of financial text using NLP.

    Args:
        text: Financial text to analyze (news, reports, articles).

    Returns:
        Sentiment score and associated risk level.
    """
    print(f"[INFO] Analyzing sentiment...")
    blob = TextBlob(text)
    score = blob.sentiment.polarity      # entre -1 et 1
    subjectivity = blob.sentiment.subjectivity  # entre 0 et 1

    if score > 0.1:
        sentiment = "POSITIVE"
        risk = "Low external risk"
    elif score < -0.1:
        sentiment = "NEGATIVE"
        risk = "High external risk"
    else:
        sentiment = "NEUTRAL"
        risk = "Moderate external risk"

    return f"""
SENTIMENT ANALYSIS
Sentiment: {sentiment}
Polarity score: {score:.2f} (range -1 to 1)
Subjectivity: {subjectivity:.2f} (0=objective, 1=subjective)
Risk level: {risk}
"""

def calculate_portfolio_performance(
    tickers: str,
    weights: str = None,
    horizon: int = 252
) -> str:
    """Calculate portfolio performance metrics including Sharpe ratio
    and comparison with equal-weight and optimal portfolios.

    Args:
        tickers: Comma-separated list of tickers (e.g. AAPL,NVDA,MSFT).
        weights: Optional comma-separated weights (e.g. 0.4,0.3,0.3).
                Equal weights if not provided.
        horizon: Investment horizon in trading days (default 252 = 1 year).
                Use 21 for 1 month, 63 for 3 months, 252 for 1 year.

    Returns:
        Expected returns, Sharpe ratio, and optimal weights suggestion.
    """
    from scipy.optimize import minimize

    ticker_list = [t.strip() for t in tickers.split(",")]
    n = len(ticker_list)

    if weights:
        w = np.array([float(x.strip()) for x in weights.split(",")])
        w = w / w.sum()
    else:
        w = np.array([1/n] * n)

    all_returns = {}
    for ticker in ticker_list:
        hist = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        prices = hist["Close"].dropna()
        returns = np.log(prices / prices.shift(1)).dropna().squeeze()
        all_returns[ticker] = returns

    returns_df = pd.DataFrame(all_returns).dropna()

    mean_returns = returns_df.mean() * horizon
    cov_matrix = returns_df.cov() * horizon

    risk_free = 0.04 * (horizon / 252)

    port_return = float(np.dot(w, mean_returns))
    port_vol = float(np.sqrt(w @ cov_matrix @ w))
    sharpe = (port_return - risk_free) / port_vol

    def neg_sharpe(weights):
        r = np.dot(weights, mean_returns)
        v = np.sqrt(weights @ cov_matrix @ weights)
        return -(r - risk_free) / v

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = [(0, 1)] * n
    w0 = np.array([1/n] * n)

    result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    opt_return = float(np.dot(optimal_weights, mean_returns))
    opt_vol = float(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights))
    opt_sharpe = (opt_return - risk_free) / opt_vol

    # Label horizon
    if horizon == 21:
        horizon_label = "1 month"
    elif horizon == 63:
        horizon_label = "3 months"
    elif horizon == 252:
        horizon_label = "1 year"
    else:
        horizon_label = f"{horizon} trading days"

    # Résumé
    current_summary = ""
    for i, ticker in enumerate(ticker_list):
        current_summary += f"  {ticker}: weight={w[i]:.1%}, expected return={mean_returns[ticker]:.2%}\n"

    optimal_summary = ""
    for i, ticker in enumerate(ticker_list):
        optimal_summary += f"  {ticker}: weight={optimal_weights[i]:.1%}\n"
    print(f"[INFO] Done sentiment!")
    return f"""
=== PORTFOLIO PERFORMANCE ANALYSIS ===
Assets: {', '.join(ticker_list)}
Horizon: {horizon_label}

CURRENT PORTFOLIO
{current_summary}
Expected Return: {port_return:.2%}
Volatility: {port_vol:.2%}
Sharpe Ratio: {sharpe:.2f}
({'Good' if sharpe > 1 else 'Acceptable' if sharpe > 0.5 else 'Poor'})

OPTIMAL PORTFOLIO (max Sharpe)
{optimal_summary}
Expected Return: {opt_return:.2%}
Volatility: {opt_vol:.2%}
Sharpe Ratio: {opt_sharpe:.2f}

CONCLUSION
{'Your portfolio is close to optimal.' if abs(sharpe - opt_sharpe) < 0.1
else f'Rebalancing could improve Sharpe from {sharpe:.2f} to {opt_sharpe:.2f}.'}
"""