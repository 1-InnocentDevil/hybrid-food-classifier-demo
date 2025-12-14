# Hybrid Knowledge-Based Expert System (Demo)

Public demonstration of a hybrid pipeline for classifying informal food descriptions using:
- BERT encoder + prototype classifier
- PPO embedding refinement (Stable-Baselines3)
- Fuzzy and symbolic knowledge-based adjustments (KBS)
- Online inference API with logging, plus optional offline updates

## Data
This repository includes a small synthetic dataset for demonstration only:
- `data/RefinedData.parquet`

The labels in the synthetic dataset must match `config/class_order.json` exactly.

## Repository layout
- `src/` main scripts
- `config/` class order, fuzzy terms, symbolic rules, optional prototypes
- `data/` synthetic demo Parquet
- `logs/` created at runtime by the server (not committed)
- `reports/` created by evaluation/mining scripts (not committed)

## 1) Freeze split
python .\src\splitter.py

## 2) Train BERT → PPO
python .\src\ReinforcementLearningWithRefinedData_parquet.py

## 3) Baseline evaluations
python .\src\eval_end_to_end.py --variant bert_proto
python .\src\eval_end_to_end.py --variant ppo
python .\src\eval_end_to_end.py --variant ppo_kbs
python .\src\eval_end_to_end.py --variant kbs_fuzzy_only
python .\src\eval_end_to_end.py --variant kbs_sym_only

## 4) Run API server
uvicorn src.server:app --reload --host 127.0.0.1 --port 8000

## 5) Ingest logs
python .\src\ingest_logs.py --predict "$ROOT\logs\predict.jsonl" --feedback "$ROOT\logs\feedback.jsonl"

## 6) Mine KBS candidates
python .\src\mine_kbs_candidates.py
python .\src\mine_kbs_candidates.py --apply-fuzzy --apply-symbolic

## 7) Reload server
Invoke-WebRequest -Method POST http://127.0.0.1:8000/admin/reload

## 8) Offline PPO retrain
python .\src\offline_retrain.py

## 9) Post-update evaluations
python .\src\eval_end_to_end.py --variant bert_proto
python .\src\eval_end_to_end.py --variant ppo
python .\src\eval_end_to_end.py --variant ppo_kbs
python .\src\eval_end_to_end.py --variant kbs_fuzzy_only
python .\src\eval_end_to_end.py --variant kbs_sym_only

Notes
The included dataset is synthetic and intended for demonstration.
Do not commit real user logs, private datasets, or model checkpoints.
