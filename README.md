# hackathon-DataScience

# API de Análise de Sentimento (PT + ES)

API de Análise de Sentimentos desenvolvida como MVP para hackathon, com foco em integração entre Data Science e Back-End.

A aplicação classifica textos como Positivo ou Negativo, com suporte a Português e Espanhol, aceitando JSON ou CSV, inclusive arquivos grandes via processamento assíncrono e streaming.

---

## Funcionalidades

- Classificação de sentimento (Positivo / Negativo)

- Suporte multilíngue: Português (PT) e Espanhol (ES)

- Um único endpoint para JSON e CSV

- Processamento em lote (batch)

- Streaming assíncrono (ideal para arquivos grandes)

- Limpeza automática de texto

- Modelos leves (≈ 9 MB) carregados uma única vez

---

## Estrutura do Projeto
```
hackathon-DataScience/
api/
├── main.py # API FastAPI
├── model/ # Modelo treinado em PT e ES
│   ├── sentiment_pt.joblib
│   └── sentiment_es.joblib
├── notebooks/ # Notebook de treino em PT e ES
│   ├── notebook_pt.ipynb
│   └── notebook_es.ipynb
├── requirements.txt # Dependências da API
README.md # Documentação
```

---

## Tecnologias utilizadas

- Python 3.12

- FastAPI

- scikit-learn

- Pandas

- Joblib

- langdetect

- Uvicorn

---

## Instalação e execução

### Clonar o repositório
```bash
git clone https://github.com/Hackaton-ONE/hackathon-DataScience.git
cd hackathon-DataScience
```


### Instalar dependências
```
pip install -r requirements.txt
```

---

## Executar a API
```
uvicorn main:app --reload
```
Acesse a documentação interativa:

[http://127.0.0.1:8000](http://localhost:8000/docs)

---

## Endpoint principal

### POST /sentiment/analyze

Este endpoint aceita JSON ou CSV e retorna o resultado no mesmo formato.

---
## Exemplos de uso

### JSON (single) 

```
POST /sentiment/analyze
{
  "text": "O atendimento foi excelente",
  "lang": "pt"
}
```

Resposta:

```

{
  "idioma": "pt",
  "previsao": "Positivo",
  "probabilidade": 0.93
}

```
---

### JSON (batch) 

```
POST /sentiment/analyze
{
  "texts": [
    "Produto excelente",
    "Péssimo atendimento",
    "El servicio fue horrible"
  ]
}
```

Resposta:

```

[
  {"idioma":"pt","previsao":"Positivo","probabilidade":0.95},
  {"idioma":"pt","previsao":"Negativo","probabilidade":0.91},
  {"idioma":"es","previsao":"Negativo","probabilidade":0.92}
]

```

---
### CSV (upload + download)

Arquivo de entrada (comentarios.csv)

```
text
Produto excelente
Péssimo atendimento
El producto llegó tarde
```

Requisição:

```
curl -X POST "http://localhost:8000/sentiment/analyze?lang=auto" \
  -F "file=@comentarios.csv" \
  --output resultado.csv
```

Arquivo de saída (resultado.csv)

```
text,idioma,previsao,probabilidade
Produto excelente,pt,Positivo,0.96
Péssimo atendimento,pt,Negativo,0.91
El producto llegó tarde,es,Negativo,0.88
```

---

## Modelos de Machine Learning

- Técnica: TF-IDF + Logistic Regression

- Treinados separadamente para PT e ES

- Serializados com joblib

- Otimizados para baixa latência em API

## Validações implementadas

- Verificação de formato (JSON / CSV)

- Validação da coluna text em CSV

- Checagem de texto mínimo

- Tratamento de erros amigável

## Escalabilidade

- Processamento assíncrono

- Streaming linha a linha para CSVs grandes

- Arquitetura preparada para persistência e métricas
