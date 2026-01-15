from fastapi import FastAPI
from pydantic import BaseModel
import re
import joblib
import uvicorn
import os

app = FastAPI()

# Carrega o modelo com segurança
try:
    model = joblib.load("modelo_sentimento_pt.joblib")
    print("✅ Modelo carregado!")
except:
    print("⚠️ Modelo não encontrado. O código vai quebrar se tentar prever.")
    model = None

class Input(BaseModel):
    text: str

def clean_text(text):
    text = str(text).lower()
    text = text.replace("lojas americanas", "loja") 
    text = text.replace("americanas", "loja")
    text = text.replace("americana", "loja")
    text = re.sub(r"r?\$ ?\d+([.,]\d+)?", "dinheiro", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"[^a-z0-9à-úç ]", "", text)
    text = " ".join(text.split())
    return text

# MUDANÇA IMPORTANTE: A rota que o Java chama é esta:
@app.post("/sentiment/analyze")
def predict_sentiment(input: Input):
    text = clean_text(input.text)

    if model:
        label = model.predict([text])[0]
        
        # Tenta pegar probabilidade
        try:
            proba_all = model.predict_proba([text])[0]
            prob = float(max(proba_all))
        except:
            prob = 1.0 # Se o modelo não tiver probabilidade

        if str(label).lower().startswith("pos"):
            label_clean = "Positivo"
        else:
            label_clean = "Negativo"
            
        return {
            "previsao": label_clean,
            "probabilidade": round(prob, 4)
        }
    else:
        return {"previsao": "Erro", "probabilidade": 0.0}

# Permite rodar no IntelliJ
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)