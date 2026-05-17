# Shipment Delay Predictor

End-to-end MLOps system that predicts whether an e-commerce shipment will be
delayed. XGBoost classifier + SHAP explanations + Gemini-generated narrative,
served behind a FastAPI REST API with an interactive demo page.

> **Target column caveat**: the raw dataset column `Reached.on.Time_Y.N = 1`
> means the shipment was **DELAYED** (did NOT reach on time). The naming is
> counterintuitive; we rename it to `delayed` internally everywhere.

## Architecture

```
       Train.csv (Kaggle)
              │
              ▼
   src/preprocess.py  ── fits sklearn Pipeline ──► model/preprocessor.pkl
                          │
                          ▼  (X_train, X_val, X_test as .npy)
   src/train.py  ──XGBoost + Optuna (50 trials)──► model/model.pkl
                                                   model/model_metadata.json
                          │
                          ▼ (artifacts loaded at startup)
   app/main.py  ──FastAPI──┬─► /predict (single + batch + file upload)
                           ├─► /explain  (SHAP + Gemini narration)
                           ├─► /worst-case, /best-case
                           ├─► /sensitivity (per-feature probability sweep)
                           ├─► /dataset-sample (for viz)
                           └─► /demo  ──► app/static/demo.html (interactive UI)
```

## Quick start (local)

```bash
git clone <your-repo-url> && cd shipment-delay-predictor

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt   # dev install

# macOS only: XGBoost needs OpenMP
brew install libomp

# Add your Gemini key
cp .env.example .env
# edit .env -> GEMINI_API_KEY=...

# (Optional) re-run training. The repo already ships trained model.pkl
python -m src.preprocess
python -m src.train

# Start the API + demo
uvicorn app.main:app --reload
# open http://localhost:8000/demo
```

## API reference

All examples assume the server is running at `http://localhost:8000`.

### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","model_loaded":true,"threshold_default":0.5,"gemini_configured":true}
```

### `POST /predict` — JSON body
```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "warehouse_block":"F","mode_of_shipment":"Road",
  "customer_care_calls":4,"customer_rating":2,"cost_of_product":180,
  "prior_purchases":3,"product_importance":"High","gender":"M",
  "discount_offered":45,"weight_in_gms":4200,"threshold":0.4
}'
```

### `POST /predict` — file upload (single or batch)
```bash
echo '[{"warehouse_block":"A","mode_of_shipment":"Flight",...}]' > shipments.json
curl -X POST http://localhost:8000/predict -F "file=@shipments.json"
```

### `POST /explain` — SHAP + Gemini
Same body as `/predict`. Returns top 5 SHAP factors, a natural-language
explanation, and 3 suggested operational actions. **Gracefully degrades** to a
deterministic fallback if Gemini is unavailable or rate-limited.

### `POST /worst-case`, `POST /best-case`
No body needed. Returns predictions for canonical worst/best feature combinations.

### `POST /sensitivity`
Same body as `/predict`. Sweeps each numeric feature across its observed range
(holding all others fixed) and returns the min/max probability per feature.
Shows which features are the biggest levers **for this specific shipment**.

### `GET /dataset-sample?n=300`
Returns 300 random training rows. Used by the demo's parallel-coordinates plot.

## CLI

```bash
# Inline args
python scripts/predict_cli.py predict --weight 4200 --discount 45 --mode Road \
  --warehouse F --calls 4 --rating 2 --prior 3 --importance High \
  --cost 180 --gender M --threshold 0.4

# From file
python scripts/predict_cli.py predict --file shipments.json
python scripts/predict_cli.py explain --file shipment.json
python scripts/predict_cli.py sensitivity --file shipment.json
python scripts/predict_cli.py worst-case
python scripts/predict_cli.py best-case

# Point at the deployed API
python scripts/predict_cli.py predict --file s.json --url https://your-app.onrender.com
# or export once
export API_HOST=https://your-app.onrender.com
```

## Testing

```bash
pytest -q
```

12 smoke tests covering every endpoint. The Gemini call is stubbed so tests are
hermetic and don't require an API key.

## Model performance

Trained on 10,999 rows (70/15/15 stratified split). XGBoost tuned with Optuna
(50 trials, F1 on validation).

| Metric        | Validation | Test  |
|---------------|-----------:|------:|
| F1 @ 0.5      |      0.677 | 0.666 |
| ROC-AUC       |      0.724 | 0.727 |
| PR-AUC        |      0.840 | 0.844 |

The dominant feature by gain is the engineered `high_discount` flag (>40%
discount). The dataset has weak signal on most other features — public Kaggle
leaderboards on this dataset peak around F1 0.70.

Run extensive model behavior tests:
```bash
python -m scripts.test_battery
```

## Deploying to Render (CI/CD)

This repo is configured for **Render's git-connected auto-deploy**:

1. **Push this repo to GitHub** (see step-by-step below)
2. Sign in at [render.com](https://render.com) — free tier is enough
3. Click **New → Blueprint** and pick your GitHub repo
4. Render reads `render.yaml` and provisions a web service
5. In the service's **Environment** tab, add a secret env var:
   - `GEMINI_API_KEY` = `<your key>`
   - (`GEMINI_MODEL` defaults to `gemini-2.5-flash-lite` — override if you want)
6. Click **Manual Deploy** for the first build
7. Every subsequent `git push origin main` auto-deploys

### GitHub Actions CI
On every push and PR, `.github/workflows/ci.yml` installs deps and runs `pytest`.
Render's deploy is separate — it doesn't depend on the GH Actions run.

### Free-tier caveats
- 512 MB RAM (this app uses ~250–350 MB at runtime — fits, but tight)
- Cold start of ~30s after 15 min idle
- First build takes 5–10 min (SHAP + numba pull in a lot)

## Repo layout

```
shipment-delay-predictor/
├── app/                    FastAPI app + demo HTML
│   ├── main.py
│   ├── predictor.py
│   ├── schemas.py
│   └── static/demo.html
├── src/                    ML pipeline
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── scripts/
│   ├── predict_cli.py
│   └── test_battery.py
├── tests/test_api.py
├── data/
│   ├── raw/Train.csv       (gitignored — pulled via Kaggle CLI)
│   └── processed/*.npy, *.csv
├── model/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
├── .github/workflows/ci.yml
├── render.yaml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```
