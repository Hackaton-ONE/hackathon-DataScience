from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from functools import lru_cache
from typing import Optional
import pandas as pd
from pandas.api.types import is_numeric_dtype
import json
import joblib
import shutil
import tempfile
import os

# --- Configurações ---
MODEL_PATHS = {
    "pt": "model/sentiment_pt.joblib",
    "es": "model/sentiment_es.joblib"
}

THRESHOLDS = {
    "pt": 0.624,
    "es": 0.488
}

DEFAULT_LANG = "pt"

# --- App e Middleware ---
app = FastAPI(
    title="Sentiment Analysis API (PT + ES)",
    description="API otimizada para análise de sentimento (JSON + CSV) com Cache e Streaming",
    version="1.3.3"
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# --- Carregar Modelos (Global) ---
models = {}
idx_neg = {}

print("Carregando modelos...")
for lang, path in MODEL_PATHS.items():
    try:
        model = joblib.load(path)
        models[lang] = model
        idx_neg[lang] = list(model.classes_).index("Negativo")
        print(f"Modelo {lang} carregado.")
    except Exception as e:
        print(f"Erro ao carregar modelo {lang}: {e}")

# --- Utils e Lógica de Negócio ---

def validate_lang(lang: Optional[str]) -> str:
    if lang and lang.lower() in models:
        return lang.lower()
    return DEFAULT_LANG

def clean_text(text: str) -> str:
    return str(text).strip().lower()

def clean_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower()

# --- Cache para Predições Únicas ---
@lru_cache(maxsize=10000)
def predict_cached(text: str, lang: str):
    prob = models[lang].predict_proba([text])[0][idx_neg[lang]]
    return prob

# --- Gerador para Streaming de CSV (COM LIMPEZA DE COLUNAS) ---
def process_csv_in_chunks(file_path: str, lang: str, chunk_size=5000):
    first_chunk = True
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            iterator = pd.read_csv(
                f, 
                chunksize=chunk_size, 
                sep=None,      
                engine='python', 
                on_bad_lines='skip' 
            )

            for chunk in iterator:
                chunk.columns = [c.strip().lower() for c in chunk.columns]

                main_text_col = None

                if not main_text_col:
                    for col in chunk.columns:
                        if is_numeric_dtype(chunk[col]):
                            continue
                        
                        avg_len = chunk[col].astype(str).str.len().mean()
                        if avg_len > 20:
                            main_text_col = col
                            break
                
                if not main_text_col:
                    main_text_col = chunk.columns[0]

                texts = clean_series(chunk[main_text_col]).tolist()
                
                if not texts:
                    continue

                probs = models[lang].predict_proba(texts)[:, idx_neg[lang]]

                out_df = pd.DataFrame()
                out_df["text"] = chunk[main_text_col]
                out_df["idioma"] = lang
                out_df["previsao"] = ["Negativo" if p >= THRESHOLDS[lang] else "Positivo" for p in probs]
                out_df["probabilidade"] = probs.round(4)
                
                yield out_df.to_csv(header=first_chunk, index=False)
                first_chunk = False

    except Exception as e:
        print(f"ERRO NO STREAMING: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Arquivo temporário removido: {file_path}")

# --- Endpoint Principal ---
@app.post("/sentiment/analyze")
async def analyze_sentiment(
    request: Request,
    file: UploadFile = File(None),
    lang: Optional[str] = DEFAULT_LANG
):
    lang = validate_lang(lang)
    data = None

    # --- LÓGICA DE ARQUIVO (UPLOAD) ---
    if file:
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    shutil.copyfileobj(file.file, tmp)
                    tmp_path = tmp.name
            except Exception as e:
                 raise HTTPException(500, f"Erro ao salvar arquivo temporário: {e}")
            
            return StreamingResponse(
                process_csv_in_chunks(tmp_path, lang),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=sentiment_resultado.csv"}
            )

        elif filename.endswith(".json"):
            try:
                content = await file.read()
                data = json.loads(content)
            except Exception as e:
                raise HTTPException(400, f"Arquivo JSON inválido ou corrompido: {e}")

        else:
            raise HTTPException(400, "O arquivo deve ser .csv ou .json.")

    # --- LÓGICA DE BODY (RAW JSON) ---
    if data is None:
        try:
            body = await request.json()
            data = body
        except Exception:
            pass

    # --- VALIDAÇÃO FINAL ---
    if data is None:
        raise HTTPException(400, "Envie um arquivo CSV/JSON ou um JSON válido no corpo.")

    # --- PROCESSAMENTO DO JSON (TEXT ou TEXTS) ---
    
    if "text" in data:
        text = clean_text(data["text"])
        if len(text) < 2:
            raise HTTPException(400, "Texto muito curto.")
        
        prob = predict_cached(text, lang)

        return {
            "idioma": lang,
            "previsao": "Negativo" if prob >= THRESHOLDS[lang] else "Positivo",
            "probabilidade": round(float(prob), 4)
        }

    if "texts" in data:
        raw_texts = data["texts"]
        if not isinstance(raw_texts, list):
             raise HTTPException(400, "'texts' deve ser uma lista.")

        texts = [clean_text(t) for t in raw_texts if isinstance(t, str)]
        
        if not texts:
            return JSONResponse([])

        probs = models[lang].predict_proba(texts)[:, idx_neg[lang]]

        results = []
        for p in probs:
            results.append({
                "idioma": lang,
                "previsao": "Negativo" if p >= THRESHOLDS[lang] else "Positivo",
                "probabilidade": round(float(p), 4)
            })
            
        return JSONResponse(results)

    raise HTTPException(400, "JSON deve conter campo 'text' ou 'texts'.")

# --- Health Check ---
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}

