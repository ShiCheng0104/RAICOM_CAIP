# FP-FraudSim Streaming Risk MVP

This folder adds a minimal offline-training and online-streaming loop for the existing FP-FraudSim data.

## 1. Train LightGBM

```powershell
pip install -r requirements-fraudsim.txt
python -m fraudsim.training.train --dataset fp_fraudsim_injected --model lightgbm
```

Artifacts are written to:

```text
models/lightgbm/latest/model.pkl
models/lightgbm/latest/feature_config.json
models/lightgbm/latest/metrics.json
models/leaderboard.json
```

Use `--dataset fp_fraudsim` to train on the base dataset.

## 2. Start Services

```powershell
docker compose up --build kafka redis model-api flink-jobmanager flink-taskmanager flink-risk-job
```

API health:

```text
http://localhost:8000/health
```

Dashboard:

```text
http://localhost:8000/dashboard
```

Flink UI:

```text
http://localhost:8081
```

## 3. Load Profiles and Replay Stream

```powershell
docker compose --profile tools run --rm load-profiles
docker compose --profile tools run --rm producer
```

The producer writes `transaction_events`. The Flink job writes scored messages to `risk_results` and high-risk messages to `alert_events`.

## 4. Inspect Kafka Output

Inside the Kafka container:

```powershell
docker exec -it fraudsim-kafka kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic risk_results --from-beginning
```

High-risk alerts:

```powershell
docker exec -it fraudsim-kafka kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic alert_events --from-beginning
```

## Notes

- The first implemented model adapter is `lightgbm`.
- Future models should implement the same adapter shape and will automatically share feature building, metrics, and leaderboard output.
- The dashboard discovers `models/*/latest/model.pkl` and can hot-load a model through the API.
- The PyFlink job computes the MVP window features in Python state and is intended for demo and competition validation, not high-throughput production.
