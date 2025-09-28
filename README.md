Hidden Liquidity from Quote Size Imbalance

A research project that infers a hidden liquidity parameter from best quote sizes and mid price changes in limit order books. Built from work conducted at UC Berkeley and expanded into a clean reusable package.


What this project delivers

    one ready to use Python package with clean functions for data prep, signal estimation, and model fit
    one simple runner script for quick checks on a ticker
    one example dataset to test the full path
    one research notebook that documents the approach end to end


Why it matters for trading

    Liquidity imbalance at the inside market contains predictive content for the next mid price move.
    This repo extracts a stable signal u i j and fits a structural parameter h that captures the intensity of hidden liquidity.
    The result can feed low latency decision rules or a slower horizon book tilt signal.


Quick start

    create a fresh virtual environment
    pip install .
    python scripts/run_pipeline.py data/sample_quotes.csv AAPL

You should see an implied value for h and shapes for the empirical and model u matrices.


Data spec

    input columns expected
        DATE string in yyyy mm dd
        TIME_M string with microsecond time
        SYM_ROOT string ticker
        EX one of T P Z
        BID float
        ASK float
        BIDSIZ float
        OFRSIZ float ask size

    cleaning rules
        keep periods between 10 in the morning and just before 4 in the afternoon
        drop zero quotes or non positive spread


Library map

    src hidden_liquidity core.py      pipeline functions
    scripts run_pipeline.py           command line quick check
    data sample_quotes.csv            synthetic example
    notebooks research_hidden_liquidity_ucb.ipynb    cleaned research notebook
    tests test_core.py                unit test


Suggested next steps

    plug in your own WRDS or proprietary quotes
    extend the model fit to per exchange or per symbol daily
    add a simple trading rule such as
        go long when u i j is above one half
        short otherwise
        then track hit ratio by bucket
