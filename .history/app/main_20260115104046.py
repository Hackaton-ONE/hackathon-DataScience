from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from functools import lru_cache
from typing import Optional, Union, List, Any
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
    "pt": 0.624486,
    "es": 0.488833
}

DEFAULT_LANG = "pt"

# --- App e Middleware ---
app = FastAPI(
    title="Sentiment Analysis API (PT + ES)",
    description="API otimizada para análise de sentimento (JSON + CSV) com Cache e Streaming",
    version="1.4.2"
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

# --- Lógica de Extração de Texto (Padrão + Fallback de Conteúdo) ---
def extract_text_from_json(data: Any) -> Union[str, List[str], None]:
    """
    1. Prioriza chaves oficiais ('text', 'texts').
    2. Fallback: Procura por CONTEÚDO (string > 5 chars) ignorando o nome da chave.
    """
    
    # --- Caso 1: Dicionário Único ---
    if isinstance(data, dict):
        # --- A. Prioridade Absoluta (Padrão da API) ---
        if "text" in data and isinstance(data["text"], str):
            return data["text"]
        if "texts" in data and isinstance(data["texts"], list):
            return data["texts"]

        # --- B. Fallback Inteligente (Busca por CONTEÚDO) ---
        # --- Se não achou 'text'/'texts', pega o primeiro campo que parece um texto útil. ---
        for k, v in data.items():
            # --- Se for string longa (> 5 chars), assume que é o texto ---
            if isinstance(v, str) and len(v) > 5:
                return v
            #  --- Se for lista de strings ---
            if isinstance(v, list) and v and isinstance(v[0], str):
                return v

    # --- Caso 2: Lista de Objetos (ex: [{"id":1, "msg":"..."}, ...]) ---
    elif isinstance(data, list):
        if not data: 
            return []
        
        # --- Analisa o primeiro item para descobrir qual chave usar ---
        sample = data[0]
        if isinstance(sample, dict):
            found_key = None
            
            # --- 1. Tenta achar 'text' ---
            if "text" in sample:
                found_key = "text"
            else:
                # --- 2. Fallback: Procura chave com valor > 5 chars ---
                for k, v in sample.items():
                    if isinstance(v, str) and len(v) > 5:
                        found_key = k
                        break

            # --- Se achou a chave, extrai de todos os itens da lista ---
            if found_key:
                return [item.get(found_key, "") for item in data if isinstance(item, dict)]

    return None

# --- Gerador para Streaming de CSV ---
def process_csv_in_chunks(file_path: str, lang: str, chunk_size=5000):
    first_chunk = True
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            iterator = pd.read_csv(f, chunksize=chunk_size, sep=None, engine='python', on_bad_lines='skip')
            for chunk in iterator:
                chunk.columns = [c.strip().lower() for c in chunk.columns]
                main_text_col = None
                
                # --- LÓGICA PURAMENTE MATEMÁTICA ---
                
                for col in chunk.columns:
                    # --- 1. Ignora colunas numéricas (IDs, Preços) ---
                    if is_numeric_dtype(chunk[col]): 
                        continue
                    
                    # --- 2. Calcula a média de caracteres da coluna inteira ---
                    avg_len = chunk[col].astype(str).str.len().mean()
                    
                    # --- 3. Se a média for maior que 20, assumimos que é o texto ---
                    if avg_len > 20:
                        main_text_col = col
                        break
                
                # --- Fallback: Se nenhuma coluna tiver média > 20, pega a primeira ---
                if not main_text_col: 
                    main_text_col = chunk.columns[0]

                texts = clean_series(chunk[main_text_col]).tolist()
                if not texts: continue

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
        if os.path.exists(file_path): os.remove(file_path)
        print(f"Arquivo temporário removido: {file_path}")

# --- Endpoint Principal ---
@app.post("/sentiment/analyze")
async def analyze_sentiment(
    request: Request,
    file: UploadFile = File(None),
    lang: Optional[str] = DEFAULT_LANG
):
    lang = validate_lang(lang)
    json_payload = None

    # --- 1. ARQUIVO (CSV ou JSON Upload) ---
    if file:
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    shutil.copyfileobj(file.file, tmp)
                    tmp_path = tmp.name
            except Exception as e:
                 raise HTTPException(500, f"Erro ao salvar arquivo temporário: {e}")
            
            base_name = os.path.splitext(file.filename)[0]
            output_filename = f"{base_name}_resultado.csv"
            return StreamingResponse(
                process_csv_in_chunks(tmp_path, lang),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={output_filename}"}
            )

        elif filename.endswith(".json"):
            try:
                content = await file.read()
                json_payload = json.loads(content)
            except Exception as e:
                raise HTTPException(400, f"Arquivo JSON inválido: {e}")
        else:
            raise HTTPException(400, "O arquivo deve ser .csv ou .json.")

    # --- 2. BODY (Raw JSON) ---
    if json_payload is None:
        try:
            body = await request.json()
            json_payload = body
        except Exception:
            pass

    # --- VALIDAÇÃO ---
    if json_payload is None:
        raise HTTPException(400, "Envie um arquivo CSV/JSON ou um JSON válido no corpo.")

    # --- 3. PROCESSAMENTO SMART JSON ---
    extracted_data = extract_text_from_json(json_payload)

    if extracted_data is None:
         raise HTTPException(400, "JSON inválido. Use 'text', 'texts' ou um campo com texto longo.")

    # --- A. Texto Único (String) ---
    if isinstance(extracted_data, str):
        text = clean_text(extracted_data)
        if len(text) < 2:
            raise HTTPException(400, "Texto muito curto.")
        
        prob = predict_cached(text, lang)
        return {
            "idioma": lang,
            "previsao": "Negativo" if prob >= THRESHOLDS[lang] else "Positivo",
            "probabilidade": round(float(prob), 4)
        }

    # --- B. Lote de Textos (Lista) ---
    elif isinstance(extracted_data, list):
        texts = [clean_text(t) for t in extracted_data if isinstance(t, str)]
        
        if not texts:
            return JSONResponse([])

        probs = models[lang].predict_proba(texts)[:, idx_neg[lang]]
        results = []
        
        for i, p in enumerate(probs):
            results.append({
                "idioma": lang,
                "previsao": "Negativo" if p >= THRESHOLDS[lang] else "Positivo",
                "probabilidade": round(float(p), 4)
            })
            
        return JSONResponse(results)

    raise HTTPException(500, "Erro interno no processamento dos dados.")

# --- Health Check ---
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}

# --- INICIALIZAÇÃO ---
if __name__ == "__main__":
    import uvicorn
    # Roda na porta 8000 para bater com o Java
    uvicorn.run(app, host="0.0.0.0", port=8000)