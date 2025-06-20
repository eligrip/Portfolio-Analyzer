import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import requests
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
try:
    import secrets
except ImportError:
    secrets = None

FMP_API_KEY = getattr(secrets, 'FMP_API_KEY', None)
FMP_BASE = 'https://financialmodelingprep.com/api/v3'


def load_portfolio(csv_path):
    df = pd.read_csv(csv_path)
    if {'Ticker', 'Shares', 'avg_price'}.issubset(df.columns):
        df = df.rename(columns={'Shares': 'Quantity', 'avg_price': 'Purchase Price'})
        df['Purchase Date'] = None
    elif {'Ticker', 'Shares'}.issubset(df.columns):
        df = df.rename(columns={'Shares': 'Quantity'})
        df['Purchase Price'] = np.nan
        df['Purchase Date'] = None
    elif {'Ticker', 'Purchase Price', 'Quantity'}.issubset(df.columns):
        if 'Purchase Date' not in df.columns:
            df['Purchase Date'] = None
    else:
        raise ValueError("CSV must have Ticker, Shares, avg_price or Ticker, Shares or Ticker, Purchase Price, Quantity columns.")
    return df

def fetch_fundamentals_fmp(ticker):
    quote_url = f"{FMP_BASE}/quote/{ticker}?apikey={FMP_API_KEY}"
    resp = requests.get(quote_url)
    quote = resp.json()
    if quote and isinstance(quote, list) and len(quote) > 0:
        quote = quote[0]
    else:
        quote = {}
    fundamentals_url = f"{FMP_BASE}/ratios-ttm/{ticker}?apikey={FMP_API_KEY}"
    ratios = requests.get(fundamentals_url).json()
    ratios = ratios[0] if ratios and isinstance(ratios, list) else {}
    profile_url = f"{FMP_BASE}/profile/{ticker}?apikey={FMP_API_KEY}"
    profile = requests.get(profile_url).json()
    profile = profile[0] if profile and isinstance(profile, list) else {}
    income_url = f"{FMP_BASE}/income-statement/{ticker}?limit=4&apikey={FMP_API_KEY}"
    income = requests.get(income_url).json()
    balance_url = f"{FMP_BASE}/balance-sheet-statement/{ticker}?limit=4&apikey={FMP_API_KEY}"
    balance = requests.get(balance_url).json()
    cashflow_url = f"{FMP_BASE}/cash-flow-statement/{ticker}?limit=4&apikey={FMP_API_KEY}"
    cashflow = requests.get(cashflow_url).json()
    hist_url = f"{FMP_BASE}/historical-price-full/{ticker}?serietype=line&apikey={FMP_API_KEY}"
    hist = requests.get(hist_url).json().get('historical', [])
    hist = hist[::-1]
    price_now = hist[-1]['close'] if hist else np.nan
    price_1m = hist[-21]['close'] if len(hist) > 21 else np.nan
    price_3m = hist[-63]['close'] if len(hist) > 63 else np.nan
    price_1y = hist[0]['close'] if len(hist) > 0 else np.nan
    momentum_1m = (price_now - price_1m) / price_1m if price_1m else np.nan
    momentum_3m = (price_now - price_3m) / price_3m if price_3m else np.nan
    momentum_1y = (price_now - price_1y) / price_1y if price_1y else np.nan
    closes = np.array([h['close'] for h in hist]) if hist else np.array([])
    returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([])
    volatility = returns.std() * np.sqrt(252) if returns.size else np.nan
    roll_max = np.maximum.accumulate(closes) if closes.size else np.array([])
    drawdown = ((closes / roll_max - 1).min() if closes.size else np.nan)
    sharpe = ((returns.mean() * 252 - 0.02) / (returns.std() * np.sqrt(252)) if returns.size else np.nan)
    high_52w = np.max(closes) if closes.size else np.nan
    low_52w = np.min(closes) if closes.size else np.nan
    beta = profile.get('beta', np.nan)
    return {
        'P/E': ratios.get('peRatioTTM'),
        'PEG': ratios.get('pegRatioTTM'),
        'ROIC': ratios.get('returnOnCapitalEmployedTTM'),
        'Gross Margin %': ratios.get('grossProfitMarginTTM'),
        'FCF Yield': ratios.get('freeCashFlowYieldTTM'),
        'Debt/Equity': ratios.get('debtEquityRatioTTM'),
        'EPS (ttm)': ratios.get('epsTTM'),
        'Sector': profile.get('sector'),
        '1Y Revenue Growth': ratios.get('revenueGrowthTTM'),
        '1Y Margin Trend': ratios.get('grossProfitMarginTTM'),
        'Current Price': quote.get('price', np.nan),
        'Market Cap': quote.get('marketCap', np.nan),
        'Momentum 1M': momentum_1m,
        'Momentum 3M': momentum_3m,
        'Momentum 1Y': momentum_1y,
        'Volatility': volatility,
        'Drawdown': drawdown,
        'Sharpe': sharpe,
        '52w High': high_52w,
        '52w Low': low_52w,
        'Beta': beta,
    }

def fetch_news(ticker, max_items=3):
    url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            items = root.findall('.//item')
            news = []
            for item in items[:max_items]:
                title = item.find('title').text if item.find('title') is not None else ''
                link = item.find('link').text if item.find('link') is not None else ''
                news.append(f"- {title} ({link})")
            return news if news else ["No news found."]
        else:
            return ["No news found."]
    except Exception as e:
        return [f"Error fetching news: {e}"]

def score_position(fundamentals):
    score = 0
    notes = []
    if fundamentals['P/E'] and fundamentals['P/E'] < 15:
        score += 1
        notes.append('Low P/E')
    if fundamentals['PEG'] and fundamentals['PEG'] < 1.5:
        score += 1
        notes.append('Low PEG')
    if fundamentals['ROIC'] and fundamentals['ROIC'] > 0.15:
        score += 1
        notes.append('High ROIC')
    if fundamentals['Gross Margin %'] and fundamentals['Gross Margin %'] > 0.4:
        score += 1
        notes.append('Strong Margins')
    if fundamentals['Debt/Equity'] and fundamentals['Debt/Equity'] < 1:
        score += 1
        notes.append('Low Debt')
    return score, notes

def sector_exposure(df):
    sector_counts = df['Sector'].value_counts()
    plt.figure(figsize=(8,6))
    sector_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Sector Exposure')
    plt.ylabel('')
    plt.show()

def diversification_metrics(df):
    weights = df['Weight']
    hhi = (weights ** 2).sum()
    entropy = -np.sum(weights * np.log(weights + 1e-12))
    return hhi, entropy

def actionable_output(df):
    for idx, row in df.iterrows():
        print(f"{row['Ticker']} (Weight: {row['Weight']:.2%}): Score {row['Score']} - {row['Notes']}")
        print(f"  Cost Basis: ${row['Cost Basis']:.2f} | Current Value: ${row['Current Value']:.2f} | Gain/Loss: ${row['Gain/Loss']:.2f} ({row['Return %']:.2f}%)")
        print(f"  Momentum 1M: {row['Momentum 1M']:.2%} | 3M: {row['Momentum 3M']:.2%} | 1Y: {row['Momentum 1Y']:.2%}")
        print(f"  Volatility: {row['Volatility']:.2%} | Drawdown: {row['Drawdown']:.2%} | Sharpe: {row['Sharpe']:.2f} | Beta: {row['Beta']:.2f}")
        print(f"  52w High: {row['52w High']:.2f} | 52w Low: {row['52w Low']:.2f}")
        print("  News:")
        for news_item in row['News']:
            print(f"    {news_item}")
        print()

def generate_daily_report(df):
    hhi, entropy = diversification_metrics(df)
    report = []
    report.append(f"Portfolio Report for {datetime.now().strftime('%Y-%m-%d')}")
    report.append("")
    report.append(f"Portfolio Diversity (HHI): {hhi:.3f} (lower is more diverse)")
    report.append(f"Portfolio Entropy: {entropy:.3f} (higher is more diverse)")
    report.append("")
    for idx, row in df.iterrows():
        report.append(f"{row['Ticker']} (Weight: {row['Weight']:.2%}): Score {row['Score']} - {row['Notes']}")
        report.append(f"  Cost Basis: ${row['Cost Basis']:.2f} | Current Value: ${row['Current Value']:.2f} | Gain/Loss: ${row['Gain/Loss']:.2f} ({row['Return %']:.2f}%)")
        report.append(f"  Momentum 1M: {row['Momentum 1M']:.2%} | 3M: {row['Momentum 3M']:.2%} | 1Y: {row['Momentum 1Y']:.2%}")
        report.append(f"  Volatility: {row['Volatility']:.2%} | Drawdown: {row['Drawdown']:.2%} | Sharpe: {row['Sharpe']:.2f} | Beta: {row['Beta']:.2f}")
        report.append(f"  52w High: {row['52w High']:.2f} | 52w Low: {row['52w Low']:.2f}")
        report.append("  News:")
        for news_item in row['News']:
            report.append(f"    {news_item}")
        report.append("")
    return '\n'.join(report)

def analyze_portfolio(csv_path):
    portfolio = load_portfolio(csv_path)
    fundamentals_list = []
    news_list = []
    for ticker in portfolio['Ticker']:
        try:
            fundamentals = fetch_fundamentals_fmp(ticker)
        except Exception as e:
            fundamentals = {k: None for k in [
                'P/E', 'PEG', 'ROIC', 'Gross Margin %', 'FCF Yield', 'Debt/Equity', 'EPS (ttm)', 'Sector',
                '1Y Revenue Growth', '1Y Margin Trend', 'Current Price', 'Market Cap',
                'Momentum 1M', 'Momentum 3M', 'Momentum 1Y', 'Volatility', 'Drawdown', 'Sharpe', '52w High', '52w Low', 'Beta'
            ]}
        fundamentals_list.append(fundamentals)
        news = fetch_news(ticker)
        news_list.append(news)
        time.sleep(3)
    fundamentals_df = pd.DataFrame(fundamentals_list)
    portfolio = pd.concat([portfolio, fundamentals_df], axis=1)
    portfolio['Current Value'] = portfolio['Current Price'] * portfolio['Quantity']
    portfolio['Cost Basis'] = portfolio['Purchase Price'] * portfolio['Quantity']
    portfolio['Gain/Loss'] = portfolio['Current Value'] - portfolio['Cost Basis']
    portfolio['Return %'] = (portfolio['Gain/Loss'] / portfolio['Cost Basis']) * 100
    total_value = portfolio['Current Value'].sum()
    if total_value == 0 or np.isnan(total_value):
        portfolio['Weight'] = 0
    else:
        portfolio['Weight'] = portfolio['Current Value'] / total_value
    scores_notes = portfolio.apply(lambda row: score_position(row), axis=1)
    portfolio['Score'] = [sn[0] for sn in scores_notes]
    portfolio['Notes'] = [', '.join(sn[1]) for sn in scores_notes]
    portfolio['News'] = news_list
    portfolio = portfolio.sort_values('Weight', ascending=False).reset_index(drop=True)
    return portfolio

def send_email(report, subject="Daily Portfolio Report", to_email="eligripenstraw@gmail.com"):
    if not secrets:
        return
    msg = MIMEMultipart()
    msg['From'] = secrets.GMAIL_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(report, 'plain'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(secrets.GMAIL_USER, secrets.GMAIL_PASSWORD)
            server.sendmail(secrets.GMAIL_USER, to_email, msg.as_string())
    except Exception as e:
        pass

def run_daily(csv_path):
    portfolio = analyze_portfolio(csv_path)
    actionable_output(portfolio)
    sector_exposure(portfolio)
    hhi, entropy = diversification_metrics(portfolio)
    report = generate_daily_report(portfolio)
    send_email(report)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    run_daily(sys.argv[1])

# Scheduling instructions (not code):
# To schedule this script to run every day at 9am Eastern, add the following line to your crontab (convert 9am ET to your local time if needed):
# 0 9 * * * /usr/bin/python3 /path/to/portfolio_analyzer.py /path/to/portfolio.csv
# Or use 'cron -e' to edit your crontab. 