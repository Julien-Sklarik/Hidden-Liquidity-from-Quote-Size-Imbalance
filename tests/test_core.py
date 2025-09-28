import pandas as pd
from hidden_liquidity.core import run_pipeline

def test_pipeline_on_sample():
    res = run_pipeline("data/sample_quotes.csv", "AAPL")
    assert "implied_h" in res
    assert res["u_empirical"].shape == (10,10)
    assert res["u_model"].shape == (10,10)
    assert res["loss"] >= 0.0