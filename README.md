# hackathon-DataScience

# API de Análise de Sentimento (Português)

Este projeto contém uma **API de Análise de Sentimento** em português, construída com **FastAPI** e utilizando um modelo treinado com **TF-IDF + Naive Bayes**, salvo em `joblib`.

A API recebe um texto e retorna a **classificação de sentimento** (positivo/negativo) e a **probabilidade associada**.

---

## 1. Funcionalidades

- Classificação automática de sentimento em PT-BR  
- Limpeza de texto (normalização, remoção de links, moedas, símbolos etc.)  
- API FastAPI com documentação automática  
- Carregamento de modelo `.joblib`  
- Testes simples em Doc UI / cURL 

---

## 2. Estrutura do Projeto
```
hackathon-DataScience/
├── app.py # API FastAPI
├── modelo_sentimento_pt.joblib # Modelo treinado
├── model_pt.ipynb # Notebook de treino
├── requirements.txt # Dependências da API
└── README.md # Documentação
```

---

## 3. Pré-requisitos

- Python 3.9+
- pip atualizado
- (Opcional, recomendado) Virtualenv

---

## 4. Instalação Local

### 1. Clonar o repositório
```bash
git clone https://github.com/Hackaton-ONE/hackathon-DataScience.git
cd hackathon-DataScience
```

### 2. Criar ambiente virtual (opcional)
```
python -m venv venv
venv\Scripts\activate          # Windows
# ou
source venv/bin/activate      # Linux/Mac
```

### 3. Instalar dependências
```
pip install -r requirements.txt
```

---

## 5. Executar a API
```
uvicorn app:app --reload
```
A API ficará disponível em:

http://127.0.0.1:8000

---

## 6. Testando a API

### Testar pelo Swagger 
1. Acesse:
--> http://127.0.0.1:8000/docs

2. Abra o endpoint POST /predict

3. Envie algo como:
```
{
"text": "Eu amei esse produto!"
}
```

### Testar via cURL
```
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"O atendimento foi horrível\"}"
```

---

## 7. Exemplo de Resposta da API
```
{
  "previsao": "Positivo",
  "probabilidade": 0.8735
}
```
