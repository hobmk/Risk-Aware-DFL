from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


# 우선 고정된 ticker 목록을 직접 관리

DOW30_TICKERS = [
    "AAPL",
    "AMGN",
    "AMZN",
    "AXP",
    "BA",
    "CAT",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "GS",
    "HD",
    "HON",
    "IBM",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "NVDA",
    "PG",
    "SHW",
    "TRV",
    "UNH",
    "V",
    "VZ",
    "WMT",
]


def download_adjusted_close(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=True,
        threads=True,
        group_by="column",
    )

    if data.empty:
        raise RuntimeError("다운로드된 데이터가 없습니다.")

    # 여러 ticker를 다운로드하면 MultiIndex column이 생성된다.
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise RuntimeError(
                f"Close 열을 찾을 수 없습니다: {data.columns}"
            )

        close = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            raise RuntimeError(
                f"Close 열을 찾을 수 없습니다: {data.columns}"
            )

        close = data[["Close"]].copy()

        if len(tickers) == 1:
            close.columns = tickers

    close.index = pd.to_datetime(close.index)
    close.index.name = "Date"

    close = close.sort_index()
    close = close.reindex(columns=tickers)

    return close


def print_data_diagnostics(prices: pd.DataFrame) -> None:
    print("=" * 70)
    print("다운로드 결과")
    print("=" * 70)
    print(f"기간: {prices.index.min().date()} ~ {prices.index.max().date()}")
    print(f"거래일 수: {len(prices)}")
    print(f"자산 수: {prices.shape[1]}")

    missing = prices.isna().sum()
    missing_ratio = prices.isna().mean()

    diagnostics = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_ratio": missing_ratio,
            "first_valid_date": [
                prices[ticker].first_valid_index()
                for ticker in prices.columns
            ],
            "last_valid_date": [
                prices[ticker].last_valid_index()
                for ticker in prices.columns
            ],
        }
    )

    print()
    print(diagnostics.to_string())

    problematic = diagnostics[
        diagnostics["missing_count"] > 0
    ]

    print()
    if problematic.empty:
        print("모든 자산에 완전한 가격 데이터가 있습니다.")
    else:
        print(
            f"결측치가 있는 자산: "
            f"{problematic.index.tolist()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--universe",
        choices=["dow30"],
        default="dow30",
    )

    parser.add_argument(
        "--start-date",
        default="2010-01-01",
    )

    parser.add_argument(
        "--end-date",
        default="2025-01-01",
        help="yfinance의 end는 포함되지 않습니다.",
    )

    parser.add_argument(
        "--output",
        default="data/raw/dow30_adjusted_close.csv",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.universe == "dow30":
        tickers = DOW30_TICKERS
    else:
        raise ValueError(f"지원하지 않는 universe: {args.universe}")

    prices = download_adjusted_close(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print_data_diagnostics(prices)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prices.to_csv(output_path)

    print()
    print(f"저장 완료: {output_path}")
    print(f"저장 shape: {prices.shape}")


if __name__ == "__main__":
    main()