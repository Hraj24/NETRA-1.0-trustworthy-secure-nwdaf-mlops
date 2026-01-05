# NETRA

**NETRA** is a trustworthy, secure, and explainable MLOps framework designed for **Network Data Analytics Function (NWDAF)** in **5G/6G network environments**.  
It focuses on building reliable ML pipelines for network intelligence with emphasis on **trust, explainability, drift handling, and safe deployment**.

---

## 🔍 Key Objectives

- Enable trustworthy ML lifecycle management for NWDAF
- Provide explainable AI (XAI) for network decisions
- Detect data & concept drift in live network telemetry
- Support secure, reproducible, containerized deployment
- Bridge the gap between research ML models and production-grade MLOps

---

## ✨ Core Features

- 📊 NWDAF-ready Network Data Analytics
- 🔐 Trustworthy ML-Ops pipeline
- 🧠 Explainable AI using SHAP
- 🔁 Drift detection & monitoring
- 🐳 Docker & Docker Compose based deployment
- 📈 Interactive frontend dashboard
- ⚙️ FastAPI backend for inference & explainability

---

## 🏗️ Project Structure

```
NETRA/
│
├── data/                    # Datasets & processed data
├── notebooks/               # Jupyter notebooks (EDA, experiments)
├── reports/                 # SHAP plots & analysis outputs
├── models/                  # Trained ML models & artifacts
├── logs/                    # Runtime logs (ignored in git)
│
├── nwdaf-dashboard/         # Frontend (Vite + React)
│   ├── src/
│   ├── public/
│   └── Dockerfile
│
├── src/                     # Backend source code (FastAPI)
│   ├── api/
│   ├── services/
│   ├── utils/
│
├── Dockerfile.backend       # Backend Dockerfile
├── docker-compose.yml       # Multi-service orchestration
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Tech Stack

### Backend
- Python
- FastAPI
- Scikit-learn
- SHAP
- Uvicorn

### Frontend
- React (Vite)
- Axios
- Nginx

### MLOps & Infra
- Docker
- Docker Compose
- GitHub

---

## ⚙️ Getting Started (Docker)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Hraj24/NETRA-1.0-trustworthy-secure-nwdaf-mlops.git
cd NETRA-1.0-trustworthy-secure-nwdaf-mlops
```

### 2️⃣ Build & run
```bash
docker compose up --build
```

### 3️⃣ Access services

| Service | URL |
|-------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

---

## 🧪 API Endpoints

| Method | Endpoint | Description |
|------|---------|-------------|
| POST | /predict | Run ML prediction |
| POST | /explain | SHAP-based explanation |
| GET  | /health | Health check |

---

## 🧠 Explainability

NETRA integrates **SHAP** to:
- Explain individual predictions
- Visualize global & local feature importance
- Improve trust in ML-driven network decisions

Plots are stored in:
```
reports/
```

---

## 📈 Drift Monitoring

- Tracks distribution changes in network data
- Logs drift events for further analysis
- Enables safer model lifecycle management

---

## 🔐 Trust & Security

- Reproducible Docker builds
- Clear separation of training, inference, and monitoring
- Designed for future extensions like Federated Learning and secure model rollout

---

## 👤 Author

**Harsh Raj**  
GitHub: https://github.com/Hraj24

---

## 📜 License

Academic & research use. License can be added later.
