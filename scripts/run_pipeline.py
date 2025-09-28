import argparse
from hidden_liquidity.core import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Hidden liquidity quick run")
    parser.add_argument("quotes_csv", help="path to quotes csv")
    parser.add_argument("symbol", help="ticker symbol such as AAPL")
    args = parser.parse_args()
    res = run_pipeline(args.quotes_csv, args.symbol)
    print("symbol,", res["symbol"])
    print("implied_h,", round(res["implied_h"], 4))
    print("loss,", round(res["loss"], 6))
    print("u_empirical shape,", res["u_empirical"].shape)
    print("u_model shape,", res["u_model"].shape)

if __name__ == "__main__":
    main()