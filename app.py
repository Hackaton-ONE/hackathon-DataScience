from fastapi import FastAPI
from pydantic import BaseModel
import re
import joblib

app = FastAPI()

model = joblib.load("modelo_sentimento_pt.joblib")

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

@app.post("/predict")
def predict_sentiment(input: Input):
    text = clean_text(input.text)

    label = model.predict([text])[0]

    proba_all = model.predict_proba([text])[0]
    prob = float(max(proba_all))

    if str(label).lower().startswith("pos"):
        label_clean = "Positivo"
    else:
        label_clean = "Negativo"

    return {
        "previsao": label_clean,
        "probabilidade": round(prob, 4)
    }

