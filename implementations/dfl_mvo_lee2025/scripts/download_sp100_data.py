from pathlib import Path
from io import StringIO

import requests
import pandas as pd
import yfinance as yf


START_DATE = "2010-01-01"
END_DATE = "2025-01-01"

OUTPUT_DIR = Path("data/raw")
RAW_PATH = OUTPUT_DIR / "sp100_adjusted_close_raw.csv"
CLEAN_PATH = OUTPUT_DIR / "sp100_adjusted_close.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "sp100_data_diagnostics.csv"
CONSTITUENTS_PATH = OUTPUT_DIR / "sp100_constituents.csv"
COMPLETE_TICKERS_PATH = OUTPUT_DIR / "sp100_complete_tickers.csv"


def get_sp100_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/S%26P_100"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    table = next(table for table in tables if "Symbol" in table.columns)

    tickers = table["Symbol"].astype(str).str.strip().tolist()
    return [ticker.replace(".", "-") for ticker in tickers]


def download_prices(tickers: list[str]) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=True,
        threads=True,
        group_by="column",
    )

    if data.empty:
        raise RuntimeError("다운로드된 가격 데이터가 없습니다.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()

    prices = prices.reindex(columns=tickers)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    return prices.sort_index()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = get_sp100_tickers()

    pd.DataFrame({"ticker": tickers}).to_csv(
        CONSTITUENTS_PATH,
        index=False,
    )

    print(f"S&P100 constituents: {len(tickers)}")

    prices = download_prices(tickers)
    prices.to_csv(RAW_PATH)

    diagnostics = pd.DataFrame({
        "missing_count": prices.isna().sum(),
        "missing_ratio": prices.isna().mean(),
        "first_valid_date": [
            prices[ticker].first_valid_index()
            for ticker in prices.columns
        ],
        "last_valid_date": [
            prices[ticker].last_valid_index()
            for ticker in prices.columns
        ],
    })

    diagnostics.to_csv(DIAGNOSTICS_PATH)

    # 현재 DFL dataset은 NaN을 허용하지 않으므로
    # 전체 기간에 가격 데이터가 존재하는 종목만 사용
    complete_tickers = diagnostics.index[
        diagnostics["missing_count"] == 0
    ].tolist()

    clean_prices = prices[complete_tickers].copy()
    clean_prices.to_csv(CLEAN_PATH)

    pd.DataFrame({
        "ticker": complete_tickers
    }).to_csv(
        COMPLETE_TICKERS_PATH,
        index=False,
    )

    dropped_tickers = [
        ticker
        for ticker in tickers
        if ticker not in complete_tickers
    ]

    print()
    print("=" * 80)
    print("S&P100 DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Original assets : {len(tickers)}")
    print(f"Complete assets : {len(complete_tickers)}")
    print(f"Dropped assets  : {len(dropped_tickers)}")
    print(f"Trading days    : {len(clean_prices)}")
    print(
        f"Date range      : "
        f"{clean_prices.index.min().date()} ~ "
        f"{clean_prices.index.max().date()}"
    )
    print(
        f"Missing values  : "
        f"{int(clean_prices.isna().sum().sum())}"
    )
    print(f"Saved shape     : {clean_prices.shape}")

    print("\nDropped tickers:")
    print(dropped_tickers)

    print("\nSaved:")
    print(CLEAN_PATH)
    print(DIAGNOSTICS_PATH)
    print(COMPLETE_TICKERS_PATH)


if __name__ == "__main__":
    main()