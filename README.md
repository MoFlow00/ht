# Discovery Hub

A Telegram channel crawler, data-processing pipeline, and link discovery website.

## Project map

### Website
- `index.html`: GitHub Pages frontend. Contains the Discovery Hub interface, search, Exclude terms, Grid/List views, and admin login.
- The website calls the Cloudflare Worker API. GitHub Pages does not execute `worker.js`.

### Cloudflare API
- `worker.js`: Cloudflare Worker API. Reads and updates the KV namespace bound as `DATA_STORE`.
- `TELE_API_V2.md`: API routes, KV keys, and deployment configuration.

Deploy `worker.js` separately through the Cloudflare Worker dashboard. Keep the existing `DATA_STORE` binding and the `ADMIN_API_KEY` secret.

### Crawler and data pipeline
- `run_crawler.py`: Collects Telegram channel data and writes temporary output under `crawler_output/`.
- `update_pipeline.py`: Processes crawler output, applies cleanup rules and scoring, then syncs data to Hugging Face and Cloudflare KV.
- `purge_rules.py`: Shared matching and normalization helpers used by the data pipeline.
- `purge_rules.json`: Configurable terms used by the cleanup pipeline.
- `explorer_scoring.py`: Discovery classification, ranking, and trend-scoring helpers.
- `interests.json`: Interest keywords used by the pipeline.
- `ai_models_catalog.json`: Model catalog used by scoring/classification.
- `PURGE_RULES.md`: Notes on cleanup-rule behavior.

### Tests
- `tests/test_explorer_scoring.py`: Scoring and classification tests.
- `tests/test_purge_rules.py`: Cleanup-rule tests.
- `tests/test_view_history.py`: View-history regression tests.
- `tests/worker.test.js`: Cloudflare Worker API tests.

### Automation
- `.github/workflows/crawler_pipeline.yml`: Scheduled crawler and data-sync workflow. Runs every 30 minutes and also supports manual runs.
- `.github/workflows/tests.yml`: Runs Python tests, Worker tests, and JavaScript syntax checks on pushes and pull requests.

## Generated folders

These are runtime output folders, not source code. They are ignored by Git:
- `crawler_output/`
- `data_sync/`
- Python `__pycache__/` and `*.pyc`

Do not commit generated datasets or crawler output unless there is a specific reason.

## Required GitHub Actions secrets

The crawler workflow reads:
- `HF_TOKEN`
- `HF_REPO_ID`
- `CF_API_TOKEN`
- `CF_ACCOUNT_ID`
- `CF_KV_NAMESPACE_ID`

The Cloudflare Worker uses the secret:
- `ADMIN_API_KEY`

Do not put secret values in source files or commit them to Git.

## Deployment flow

1. Push website changes to the branch configured in GitHub Pages. GitHub Pages publishes `index.html`.
2. Deploy `worker.js` separately to Cloudflare Workers.
3. Keep the Worker KV binding and secrets configured in Cloudflare.
4. Check GitHub Actions before deploying changes that affect the crawler or Worker.

## Working rule

Keep the website, Worker API, crawler, and data pipeline as separate deployable parts. A change to `index.html` does not deploy the Worker, and a Worker deployment does not publish the website.
