from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import re
import io
import asyncio
from langdetect import detect

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

# =========================
# App
# =========================
app = FastAPI(
    title="Sentiment Analysis API (PT + ES)",
    description="Classificação de sentimento com suporte a JSON e CSV",
    version="1.0.0"
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
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    return text

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang.startswith("es"):
            return "es"
    except:
        pass
    return "pt"  # fallback seguro

def predict(text: str, lang: Optional[str] = None):
    text = clean_text(text)

    if len(text) < 5:
        raise ValueError("Texto muito curto")

    if not lang or lang == "auto":
        lang = detect_language(text)

    model = models.get(lang)
    if not model:
        raise ValueError(f"Idioma '{lang}' não suportado")

    prob_neg = model.predict_proba([text])[0][idx_neg[lang]]
    label = "Negativo" if prob_neg >= THRESHOLDS[lang] else "Positivo"

    return {
        "idioma": lang,
        "previsao": label,
        "probabilidade": round(float(prob_neg), 4)
    }

# =========================
# Schemas JSON
# =========================
class TextRequest(BaseModel):
    text: str
    lang: Optional[str] = "auto"

class BatchRequest(BaseModel):
    texts: List[str]
    lang: Optional[str] = "auto"

# =========================
# Endpoint UNIFICADO
# =========================
@app.post("/sentiment/analyze")
async def analyze_sentiment(
    request: Request,
    file: UploadFile = File(None),
    lang: Optional[str] = "auto"
):
    # =====================
    # CASO CSV (STREAMING)
    # =====================
    if file is not None:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Arquivo deve ser CSV")

        df = pd.read_csv(file.file)

        if "text" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV deve conter coluna 'text'"
            )

        async def stream_csv():
            buffer = io.StringIO()
            header_written = False

            for idx, row in df.iterrows():
                try:
                    result = predict(row["text"], lang)
                except ValueError:
                    result = {
                        "idioma": None,
                        "previsao": "Erro",
                        "probabilidade": None
                    }

                out_row = row.to_dict()
                out_row.update(result)

                pd.DataFrame([out_row]).to_csv(
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
        try:
            return JSONResponse(
                predict(data["text"], data.get("lang", lang))
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # JSON batch
    if "texts" in data and isinstance(data["texts"], list):
        results = []
        for text in data["texts"]:
            try:
                results.append(predict(text, data.get("lang", lang)))
            except ValueError:
                results.append({
                    "idioma": None,
                    "previsao": "Erro",
                    "probabilidade": None
                })
            await asyncio.sleep(0)

        return JSONResponse(results)

    raise HTTPException(
        status_code=400,
        detail="Formato inválido. Envie JSON ou CSV."
    )

# =========================
# Health check
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}
