from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import io
import asyncio

# =========================
# Configurações
# =========================
MODEL_PATHS = {
    "pt": "model/sentiment_pt.joblib",
    "es": "model/sentiment_es.joblib"
}

THRESHOLDS = {
    "pt": 0.624,
    "es": 0.488
}

DEFAULT_LANG = "pt"

# =========================
# App
# =========================
app = FastAPI(
    title="Sentiment Analysis API (PT + ES)",
    description="API otimizada para análise de sentimento (JSON + CSV)",
    version="1.2.0"
)

# =========================
# Carregar modelos (1x)
# =========================
models = {}
idx_neg = {}

for lang, path in MODEL_PATHS.items():
    model = joblib.load(path)
    models[lang] = model
    idx_neg[lang] = list(model.classes_).index("Negativo")

# =========================
# Utilidades
# =========================
def clean_text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower()

def validate_lang(lang: Optional[str]) -> str:
    if lang in models:
        return lang
    return DEFAULT_LANG

# =========================
# Schemas JSON
# =========================
class TextRequest(BaseModel):
    text: str
    lang: Optional[str] = DEFAULT_LANG

class BatchRequest(BaseModel):
    texts: List[str]
    lang: Optional[str] = DEFAULT_LANG

# =========================
# Endpoint UNIFICADO
# =========================
@app.post("/sentiment/analyze")
async def analyze_sentiment(
    request: Request,
    file: UploadFile = File(None),
    lang: Optional[str] = DEFAULT_LANG
):

    lang = validate_lang(lang)

    # =====================
    # CASO CSV (BATCH + STREAM)
    # =====================
    if file is not None:
        if not file.filename.endswith(".csv"):
            raise HTTPException(400, "Arquivo deve ser CSV")

        df = pd.read_csv(file.file)

        if "text" not in df.columns:
            raise HTTPException(400, "CSV deve conter coluna 'text'")

        texts = clean_text_series(df["text"]).tolist()

        probs = models[lang].predict_proba(texts)[:, idx_neg[lang]]
        labels = [
            "Negativo" if p >= THRESHOLDS[lang] else "Positivo"
            for p in probs
        ]

        df["idioma"] = lang
        df["previsao"] = labels
        df["probabilidade"] = probs.round(4)

        async def stream_csv():
            buffer = io.StringIO()
            header_written = False

            for start in range(0, len(df), 500):
                chunk = df.iloc[start:start + 500]
                chunk.to_csv(
                    buffer,
                    index=False,
                    header=not header_written
                )
                header_written = True
                yield buffer.getvalue()
                buffer.seek(0)
                buffer.truncate(0)
                await asyncio.sleep(0)

        return StreamingResponse(
            stream_csv(),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=sentiment_resultado.csv"
            }
        )

    # =====================
    # CASO JSON
    # =====================
    data = await request.json()

    # JSON single
    if "text" in data:
        text = data["text"].strip().lower()

        if len(text) < 5:
            raise HTTPException(400, "Texto inválido")

        prob = models[lang].predict_proba([text])[0][idx_neg[lang]]
        label = "Negativo" if prob >= THRESHOLDS[lang] else "Positivo"

        return {
            "idioma": lang,
            "previsao": label,
            "probabilidade": round(float(prob), 4)
        }

    # JSON batch
    if "texts" in data:
        texts = [t.lower() for t in data["texts"]]
        probs = models[lang].predict_proba(texts)[:, idx_neg[lang]]

        return JSONResponse([
            {
                "idioma": lang,
                "previsao": "Negativo" if p >= THRESHOLDS[lang] else "Positivo",
                "probabilidade": round(float(p), 4)
            }
            for p in probs
        ])

    raise HTTPException(400, "Formato inválido")

# =========================
# Health check
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

