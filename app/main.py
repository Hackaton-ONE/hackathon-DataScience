from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from typing import Optional
import pandas as pd
import joblib
import io

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
    version="1.2.1"
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# =========================
# Carregar modelos
# =========================
models = {}
idx_neg = {}

for lang, path in MODEL_PATHS.items():
    model = joblib.load(path)
    models[lang] = model
    idx_neg[lang] = list(model.classes_).index("Negativo")

# =========================
# Utils
# =========================
def validate_lang(lang: Optional[str]) -> str:
    return lang if lang in models else DEFAULT_LANG

def clean_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower()

# =========================
# Endpoint
# =========================
@app.post("/sentiment/analyze")
async def analyze_sentiment(
    request: Request,
    file: UploadFile = File(None),
    lang: Optional[str] = DEFAULT_LANG
):
    lang = validate_lang(lang)

    # =====================
    # CSV
    # =====================
    if file:
        if not file.filename.endswith(".csv"):
            raise HTTPException(400, "Arquivo deve ser CSV")

        df = pd.read_csv(file.file)

        if "text" not in df.columns:
            raise HTTPException(400, "CSV deve conter coluna 'text'")

        texts = clean_series(df["text"]).tolist()

        probs = models[lang].predict_proba(texts)[:, idx_neg[lang]]

        df["idioma"] = lang
        df["previsao"] = ["Negativo" if p >= THRESHOLDS[lang] else "Positivo" for p in probs]
        df["probabilidade"] = probs.round(4)

        def stream_csv():
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            yield buffer.read()

        return StreamingResponse(
            stream_csv(),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=sentiment_resultado.csv"
            }
        )

    # =====================
    # JSON
    # =====================
    data = await request.json()

    if "text" in data:
        text = data["text"].strip().lower()

        if len(text) < 5:
            raise HTTPException(400, "Texto inválido")

        prob = models[lang].predict_proba([text])[0][idx_neg[lang]]

        return {
            "idioma": lang,
            "previsao": "Negativo" if prob >= THRESHOLDS[lang] else "Positivo",
            "probabilidade": round(float(prob), 4)
        }

    if "texts" in data:
        texts = [t.strip().lower() for t in data["texts"] if isinstance(t, str)]
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
# Health
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}
