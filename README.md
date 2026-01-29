## Coffee Bean Quality Classification (MLOps Final Project)

This repository contains an end‑to‑end MLOps project for **coffee bean quality classification**.  
It includes data preparation and model training in notebooks, an MLflow‑tracked PyTorch model, a Streamlit web app for inference, a FastAPI service, Dockerized deployment, and observability with Prometheus, Loki, Grafana, and Promtail.

### Features

- **Image classification**: Classifies coffee bean images into `Defect`, `Longberry`, `Peaberry`, and `Premium`.
- **Modern UI**: Streamlit app with a custom dark theme and rich visualization of prediction confidences.
- **Experiment tracking**: Uses **MLflow** with a remote tracking server hosted on DagsHub.
- **Reproducible training**: Jupyter notebook for model training stored under `notebook/`.
- **Deployment ready**: Dockerfile, `docker-compose.yml`, and Vercel configuration for easy deployment.
- **Monitoring & logging**: Prometheus, Loki, Grafana, and Promtail configs for metrics and log aggregation.

### Tech Stack

- **Language**: Python 3.8.16
- **Modeling**: PyTorch, torchvision, scikit-learn
- **Serving**:
  - Streamlit UI (`script/app2.py`)
  - FastAPI service (`script/app.py`)
- **Experiment tracking**: MLflow (remote tracking on DagsHub)
- **Infrastructure / DevOps**: Docker, Docker Compose, GitHub Actions, Vercel
- **Monitoring**: Prometheus, Loki, Grafana, Promtail

### Project Structure

- `dataset/` – Train/val/test images for the classifier.
- `model/` – Saved model weights (`*.pth`) from training runs.
- `notebook/project_akhir.ipynb` – Training and experimentation notebook.
- `script/app2.py` – Streamlit UI for coffee bean image classification.
- `script/app.py` – Minimal FastAPI application (CORS‑enabled).
- `script/train.py`, `script/test.py`, `script/send_metric.py` – Training, evaluation, and metric utilities (if used in your workflow).
- `requirements.txt` – Core dependencies for running the app and model.
- `requirements-deploy.txt` – Dependencies for deployment environments.
- `docker-compose.yml`, `docker-compose.monitoring.yml` – Dockerized services and monitoring stack.
- `grafana/`, `loki/`, `promtail/`, `prometheus.yml` – Monitoring and logging configuration.
- `mlflow/` – Local MLflow artifacts / metadata (remote tracking is configured in the app).
- `vercel.json`, `script/vercel.json`, `vercel/` – Vercel deployment configuration and artifacts.

### Setup

#### 1. Clone the repository

```bash
git clone <your-repo-url>.git
cd project-akhir
```

#### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate  # on Windows (PowerShell)
# source .venv/bin/activate    # on macOS / Linux
```

#### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables

The Streamlit app (`script/app2.py`) uses a remote MLflow tracking server on DagsHub and expects DagsHub credentials to be available as environment variables.  
These should be stored in a **local `.env` file** (not committed to git).

Required variables:

- `DAGSHUB_USERNAME` – Your DagsHub username.
- `DAGSHUB_TOKEN` – A personal access token from DagsHub.

Example `.env` (do **not** commit this file):

```bash
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_token
```

The MLflow tracking URI is configured inside the app:

- `MLFLOW_TRACKING_URI = "https://dagshub.com/salsazufar/project-akhir-mlops.mlflow"`

### Running the Apps

#### Streamlit inference app (recommended entrypoint)

```bash
streamlit run script/app2.py
```

Then open the URL printed in the terminal (typically `http://localhost:8501`).

#### FastAPI service (simple demo API)

```bash
uvicorn script.app:app --reload
```

The root endpoint will be available at `http://127.0.0.1:8000/`.

### Monitoring & Logging (Optional)

To start the monitoring stack (Prometheus, Loki, Grafana, Promtail), you can use Docker Compose:

```bash
docker-compose -f docker-compose.monitoring.yml up
```

This will bring up the observability components defined in the corresponding config files.



