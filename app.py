from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# --- CONFIGURAÇÃO ---
# Carrega o modelo assim que a API liga
# O arquivo .pkl deve estar na mesma pasta que este script
try:
    print("Carregando modelo...")
    pipeline = joblib.load('sentiment_model.pkl')
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    pipeline = None

@app.route('/')
def home():
    return "SentimentAPI está Online! Use o endpoint /predict para analisar textos."

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Modelo não está carregado.'}), 500

    # 1. Receber o JSON do Java/Postman
    dados = request.get_json()
    
    # Validação: O JSON tem o campo 'text'?
    if not dados or 'text' not in dados:
        return jsonify({'error': 'Nenhum texto fornecido. Envie {"text": "seu texto"}'}), 400
    
    texto_usuario = dados['text']

    # 2. Fazer a Previsão
    # O pipeline já cuida da limpeza e vetorização (Lembra do nosso notebook?)
    try:
        # predict espera uma lista, por isso colocamos [texto_usuario]
        predicao = pipeline.predict([texto_usuario])[0]
        
        # Pega as probabilidades (certeza do modelo)
        # proba retorna algo como [0.1, 0.2, 0.7] (Neg, Neu, Pos)
        probs = pipeline.predict_proba([texto_usuario])[0]
        probabilidade_maxima = max(probs)
        
        # 3. Retornar a resposta
        resultado = {
            'text': texto_usuario,
            'previsao': predicao,
            'probabilidade': float(round(probabilidade_maxima, 4)),
            'status': 'sucesso'
        }
        return jsonify(resultado)

    except Exception as e:
        return jsonify({'error': f"Erro na predição: {str(e)}"}), 500

if __name__ == '__main__':
    # Roda o servidor na porta 5000
    app.run(host='0.0.0.0', port=5000, debug=True)