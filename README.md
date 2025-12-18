# 🧠 SentimentAPI - Microserviço de Data Science

Este projeto contém a API de Análise de Sentimentos.
O serviço recebe um texto e retorna se o sentimento é **Positivo**, **Neutro** ou **Negativo**.

## 🚀 Como Rodar
1. Instale as dependências:
   pip install -r requirements.txt
2. Inicie o Servidor:
   python app.py

## 🔌 Como usar (Endpoint)
* **URL:** POST http://localhost:5000/predict
* **JSON de Envio:** {"text": "O produto é ótimo"}
* **Resposta:** {"previsao": "Positivo", "probabilidade": 0.98}