"""
Service HTTP autour de NIDSPredictor.

Deux usages :
1. Appelé par le filtre `http` de Logstash (un POST /predict par événement) --
   voir logstash/pipeline/nids.conf.
2. Import manuel de données externes via /predict_csv (ou le formulaire sur
   /) : un fichier CSV est analysé en une fois (predict_batch, efficace) et
   les résultats sont indexés dans Elasticsearch (index nids-import), séparé
   de l'index nids-ml-* alimenté par le flux Logstash pour ne pas mélanger
   données de test/production et données importées ponctuellement.

Lancement : uvicorn commun.service:app --host 0.0.0.0 --port 8000
"""

import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import requests
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from commun.inference import NIDSPredictor

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="NIDS Predictor", version="1.0")
predictor: Optional[NIDSPredictor] = None


@app.exception_handler(Exception)
async def _json_error_handler(request: Request, exc: Exception):
    # Sans ce handler, une exception non geree renvoie une page HTML par
    # defaut -- le front (fetch().json()) plante alors avec une erreur
    # "unexpected character" qui masque la vraie cause. Vu en conditions
    # reelles sur un import volumineux (413 Elasticsearch non gere).
    logger.exception("Erreur non geree sur %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})

ES_HOST = os.environ.get("ES_HOST", "http://elasticsearch:9200")
# Reutilise l'utilisateur logstash_internal (deja autorise en ecriture sur
# nids-* via setup/roles/logstash_writer.json) -- ES exige une authentification
# depuis l'activation de xpack.security.
ES_USER = os.environ.get("ES_USER", "")
ES_PASSWORD = os.environ.get("ES_PASSWORD", "")
ES_AUTH = (ES_USER, ES_PASSWORD) if ES_USER else None
IMPORT_INDEX = "nids-import"
SOURCES_SUPPORTEES = ["netflow", "cicflowmeter", "zeek"]


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


BULK_BATCH_SIZE = 2000


def _bulk_index(index: str, docs: list[dict]) -> dict:
    """Indexe par lots (une requete _bulk trop volumineuse est rejetee par
    Elasticsearch avec 413 Request Entity Too Large -- vu en conditions
    reelles sur un import NetFlow de plusieurs dizaines de milliers de lignes)."""
    indexed = 0
    errors: list = []
    for start in range(0, len(docs), BULK_BATCH_SIZE):
        batch = docs[start:start + BULK_BATCH_SIZE]
        lines = []
        for doc in batch:
            lines.append(json.dumps({"index": {"_index": index}}))
            lines.append(json.dumps(doc, default=str))
        body = "\n".join(lines) + "\n"
        resp = requests.post(
            f"{ES_HOST}/_bulk",
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
            auth=ES_AUTH,
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        batch_errors = [item["index"]["error"] for item in result.get("items", []) if item.get("index", {}).get("error")]
        indexed += len(batch) - len(batch_errors)
        errors.extend(batch_errors)
    return {"indexed": indexed, "errors": len(errors), "error_detail": errors[:3]}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), source: str = Form(...)):
    if source not in SOURCES_SUPPORTEES:
        return JSONResponse(status_code=400, content={"error": f"source doit etre l'une de {SOURCES_SUPPORTEES}"})

    raw_bytes = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"CSV illisible: {e}"})

    if raw_df.empty:
        return JSONResponse(status_code=400, content={"error": "CSV vide"})

    res = predictor.predict_batch(raw_df, source=source)

    import_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    docs = []
    for i in res.index:
        row = res.loc[i].to_dict()
        is_attack = row.get("is_attack")
        is_attack = bool(is_attack) if pd.notna(is_attack) else None
        docs.append({
            "@timestamp": now,
            "import_id": import_id,
            "import_filename": file.filename,
            "data_origin": "import",
            "event": {"module": source},
            "ml_status": row.get("ml_status"),
            "is_attack": is_attack,
            "anomaly_score": float(row["anomaly_score"]) if pd.notna(row.get("anomaly_score")) else None,
            "attack_category": row.get("attack_category") if pd.notna(row.get("attack_category")) else None,
            "confidence": float(row["confidence"]) if pd.notna(row.get("confidence")) else None,
        })

    index_result = _bulk_index(IMPORT_INDEX, docs)

    total = len(docs)
    n_attacks = sum(1 for d in docs if d["is_attack"] is True)
    n_non_reconnu = sum(1 for d in docs if d["ml_status"] == "format_non_reconnu")
    categories: dict[str, int] = {}
    for d in docs:
        if d["attack_category"]:
            categories[d["attack_category"]] = categories.get(d["attack_category"], 0) + 1

    return {
        "import_id": import_id,
        "fichier": file.filename,
        "source": source,
        "total_evenements": total,
        "attaques_detectees": n_attacks,
        "evenements_normaux": total - n_attacks - n_non_reconnu,
        "format_non_reconnu": n_non_reconnu,
        "repartition_categories": categories,
        "indexation": index_result,
    }


UPLOAD_PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>NIDS - Import de donnees</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  form { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-top: 20px; }
  label { display: block; margin-top: 12px; font-weight: 600; font-size: 0.9rem; }
  select, input[type=file] { margin-top: 4px; padding: 6px; width: 100%; box-sizing: border-box; border: 1px solid #ccc; border-radius: 6px; }
  button {
    margin-top: 20px; padding: 12px 28px; width: 100%;
    background: linear-gradient(135deg, #0066cc, #004c99);
    color: white; border: none; border-radius: 8px; cursor: pointer;
    font-size: 1.05rem; font-weight: 600; letter-spacing: 0.02em;
    box-shadow: 0 2px 8px rgba(0, 102, 204, 0.35);
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.2s ease;
  }
  button:hover:not(:disabled) {
    background: linear-gradient(135deg, #0074e6, #0058b3);
    box-shadow: 0 4px 14px rgba(0, 102, 204, 0.45);
    transform: translateY(-1px);
  }
  button:active:not(:disabled) { transform: translateY(0); box-shadow: 0 2px 6px rgba(0, 102, 204, 0.35); }
  button:disabled { background: #999; box-shadow: none; cursor: not-allowed; }
  button::before { content: "\1F50D  "; }
  #result { margin-top: 20px; padding: 16px; border-radius: 8px; background: #f4f4f4; white-space: pre-wrap; font-family: monospace; font-size: 0.85rem; display: none; }
  .attack { color: #c00; font-weight: bold; }
  .ok { color: #080; font-weight: bold; }
  a.dash {
    display: inline-block; margin-top: 16px; padding: 10px 20px;
    background: #fff; color: #06c; border: 2px solid #06c; border-radius: 8px;
    text-decoration: none; font-weight: 600; transition: background 0.15s ease, color 0.15s ease;
  }
  a.dash:hover { background: #06c; color: #fff; }
</style>
</head>
<body>
  <h1>NIDS - Import et analyse de donnees</h1>
  <p>Importez un fichier CSV (NetFlow, CICFlowMeter ou Zeek) : chaque ligne est analysee par le modele
  (detection d'attaque + categorisation) et indexee dans Elasticsearch (index <code>nids-import</code>).</p>
  <form id="f">
    <label>Fichier CSV</label>
    <input type="file" name="file" id="file" accept=".csv" required>
    <label>Format des donnees</label>
    <select name="source" id="source">
      <option value="netflow">NetFlow</option>
      <option value="cicflowmeter">CICFlowMeter</option>
      <option value="zeek">Zeek (conn.log)</option>
    </select>
    <button type="submit" id="btn">Analyser</button>
  </form>
  <div id="result"></div>
  <script>
    const f = document.getElementById('f');
    const btn = document.getElementById('btn');
    const result = document.getElementById('result');
    f.addEventListener('submit', async (e) => {
      e.preventDefault();
      btn.disabled = true; btn.textContent = 'Analyse en cours...';
      result.style.display = 'block';
      result.textContent = 'Analyse en cours, patientez...';
      try {
        const fd = new FormData(f);
        const resp = await fetch('/predict_csv', { method: 'POST', body: fd });
        const data = await resp.json();
        if (!resp.ok) {
          result.textContent = 'Erreur : ' + (data.error || JSON.stringify(data));
        } else {
          let txt = `Fichier : ${data.fichier}\\nSource : ${data.source}\\n\\n`;
          txt += `Total evenements analyses : ${data.total_evenements}\\n`;
          txt += `Attaques detectees : ${data.attaques_detectees}\\n`;
          txt += `Evenements normaux : ${data.evenements_normaux}\\n`;
          txt += `Format non reconnu : ${data.format_non_reconnu}\\n\\n`;
          txt += `Repartition par categorie :\\n`;
          for (const [cat, n] of Object.entries(data.repartition_categories)) {
            txt += `  - ${cat} : ${n}\\n`;
          }
          txt += `\\nIndexe dans Elasticsearch : ${data.indexation.indexed}/${data.total_evenements} document(s)`;
          if (data.indexation.errors > 0) txt += ` (${data.indexation.errors} erreur(s))`;
          txt += `\\n\\nID d'import : ${data.import_id}`;
          result.textContent = txt;

          // Lien direct vers le dashboard, filtre sur CET import precis
          // (evite de melanger avec les imports precedents -- le dashboard
          // seul, sans filtre, ne peut pas deviner "le plus recent").
          const g = encodeURIComponent("(time:(from:now-15m,to:now))");
          const query = `import_id:${data.import_id}`;
          const a = encodeURIComponent(`(query:(language:kuery,query:'${query}'))`);
          const kibanaLink = document.getElementById('kibana-link');
          kibanaLink.href = `http://localhost:5601/app/dashboards#/view/nids-import-dashboard?_g=${g}&_a=${a}`;
          kibanaLink.textContent = 'Voir CET import dans Kibana →';
          kibanaLink.style.display = 'inline-block';
        }
      } catch (err) {
        result.textContent = 'Erreur reseau : ' + err;
      }
      btn.disabled = false; btn.textContent = 'Analyser';
    });
  </script>
  <a class="dash" id="kibana-link" href="http://localhost:5601/app/dashboards#/view/nids-import-dashboard" target="_blank">Voir les resultats dans Kibana &rarr;</a>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def upload_page():
    return UPLOAD_PAGE
