"""
Service HTTP minimal autour de NIDSPredictor, pensé pour être appelé par le
filtre `http` de Logstash (un POST par événement, ou par petit lot).

Lancement : uvicorn commun.service:app --host 0.0.0.0 --port 8000
(depuis MD_4/, pour que le package `commun` soit importable)

Exemple Logstash (filtre http, un event à la fois) :
  http {
    url => "http://localhost:8000/predict"
    body => '{"event": %{[@metadata][raw]}, "source": "%{[event][module]}"}'
    target_body => "[ml]"
  }
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional

from commun.inference import NIDSPredictor

app = FastAPI(title="NIDS Predictor", version="1.0")
predictor: Optional[NIDSPredictor] = None


class PredictRequest(BaseModel):
    event: dict[str, Any]
    source: Optional[str] = None


@app.on_event("startup")
def _load_models():
    global predictor
    predictor = NIDSPredictor()


@app.get("/health")
def health():
    return {"status": "ok", "sources_disponibles": list(predictor.medianes_par_source.keys()) if predictor else []}


@app.post("/predict")
def predict(req: PredictRequest):
    return predictor.predict(req.event, source=req.source)
