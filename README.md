# Fiducial Inference – Generalized for ETFs

Implements generalized fiducial inference (Fisher), a third way between frequentist and Bayesian statistics. Provides confidence in parameter estimates without prior distributions. Particularly well-suited for models with non-identifiable parameters. The per‑ETF score is the fiducial confidence that the expected return is positive.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Generalized fiducial distribution via pivotal quantities
- Fiducial confidence = P(mu > 0 | data)
- Macro-adjusted confidence
- Score = confidence (higher = stronger signal)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-fiducial-inference-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High confidence → strong signal for long position.
- Low confidence → weak or negative signal.

## Requirements

See `requirements.txt`.
