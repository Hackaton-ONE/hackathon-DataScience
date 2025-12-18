# 🛒 SentimentAPI - Análise de Sentimentos (Olist MVP)

> **Status:** 🟢 MVP Funcional | **Versão:** 1.0

## 📖 Sobre o Projeto
Este é um microserviço de Inteligência Artificial desenvolvido para analisar reviews de e-commerce. O sistema recebe um comentário em texto e classifica automaticamente o sentimento do cliente como **Positivo** ou **Negativo**.

O objetivo é permitir que o time de suporte identifique clientes insatisfeitos em tempo real, antes que o problema escale.

---

## 🎯 Por que escolhemos o Dataset Olist?
Para o treinamento deste modelo, optamos pelo **Brazilian E-Commerce Public Dataset by Olist** 

* **Dados Reais:** Utilizamos 100.000 reviews reais, preservando gírias, erros de português e abreviações comuns no Brasil.
* **Diversidade de Vocabulário:** Como a Olist é um marketplace (vários vendedores), o vocabulário é muito mais rico e variado do que o de um e-commerce de nicho.
* **Foco na Dor:** O dataset possui uma alta concentração de problemas logísticos (atraso, produto errado), tornando o modelo especialista em detectar falhas de entrega.

---

## 🛠️ Arquitetura Técnica
O pipeline de dados foi construído para ser leve e rápido (baixa latência):

1.  **Pré-processamento:** Limpeza de texto, remoção de stopwords e normalização.
2.  **Vetorização:** TF-IDF (Term Frequency-Inverse Document Frequency) para transformar texto em números.
3.  **Modelo:** Regressão Logística. Escolhida por ser explicável (não é "caixa preta") e extremamente rápida para inferência em CPU.
4.  **Interface:**
    * **Backend:** Flask (API REST)
    * **Frontend:** Streamlit (Dashboard de Teste)

---

## 🚀 Como Rodar o Projeto

### Pré-requisitos
* Python 3.8 ou superior.
* Pip (Gerenciador de pacotes).

### 1. Instalação
Clone o repositório e instale as dependências listadas:

```bash
pip install -r requirements.txt
(Dica para Windows: Se o comando acima falhar, tente py -m pip install -r requirements.txt)

2. Executando a Aplicação
Você tem duas formas de interagir com a IA:

🅰️ Modo Visual (Dashboard Streamlit)
Ideal para demonstrações e testes manuais rápidos. Uma interface gráfica abrirá no seu navegador.
streamlit run dashboard.py
(Windows: py -m streamlit run dashboard.py)

🅱️ Modo API (Servidor Flask)
Ideal para integração com o Backend (Java/Node/etc). O servidor ficará ouvindo na porta 5000.
python app.py
(Windows: py app.py)

🔌 Documentação da API
Se você rodar o Modo API, utilize os seguintes endpoints:

POST /predict
Recebe um texto e retorna a classificação.
Exemplo de Corpo (JSON):
{
  "comentario": "O produto chegou muito rápido e a qualidade é excelente!"
}

Exemplo de Resposta (JSON):
{
  "sentimento": "Positivo",
  "confianca": "0.98",
  "status": 200
}


## 📂 Estrutura do Repositório

| Arquivo | Função |
| :--- | :--- |
| `app.py` | API Flask (Back-end) para integração com outros sistemas via JSON. |
| `dashboard.py` | Interface gráfica em Streamlit para testes visuais e rápidos. |
| `sentiment_model.pkl` | O modelo de IA treinado e serializado (Cérebro da aplicação). |
| `SentimentAPI_Pipeline_Treinamento_v1.ipynb` | Notebook contendo todo o processo de limpeza de dados e treinamento. |
| `requirements.txt` | Lista de todas as bibliotecas necessárias para rodar o projeto. |
| `README.md` | Documentação oficial com instruções de uso. |