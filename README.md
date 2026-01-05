# hackathon-DataScience

# API de Análise de Sentimento (PT + ES)

API de Análise de Sentimentos desenvolvida como **MVP para hackathon**, com foco em **integração entre Data Science e Back-End**, **alto desempenho** e **processamento em lote**.

A aplicação classifica textos como **Positivo** ou **Negativo**, com suporte a **Português (PT)** e **Espanhol (ES)**, aceitando **JSON ou CSV**, inclusive arquivos grandes via **batch prediction** e **streaming assíncrono**.

---

## Principais características

- Classificação de sentimento (Positivo / Negativo)
- Suporte multilíngue: **Português (PT)** e **Espanhol (ES)**
- **Um único endpoint** para JSON e CSV
- **Batch prediction** (máximo desempenho)
- **Streaming assíncrono** (ideal para arquivos grandes)
- Modelos **leves (~9 MB)** carregados **uma única vez**
- Baixa latência e alta previsibilidade
- Código simples, limpo e pronto para produção

---

## Estrutura do Projeto
```
hackathon-DataScience/
api/
├── main.py # API FastAPI
├── model/ # Modelos treinados
│ ├── sentiment_pt.joblib
│ └── sentiment_es.joblib
├── notebooks/ # Notebooks de treino
│ ├── notebook_pt.ipynb
│ └── notebook_es.ipynb
├── requirements.txt # Dependências
README.md # Documentação
```

---

## Tecnologias utilizadas

- Python 3.11+
- FastAPI
- scikit-learn
- Pandas
- Joblib
- Uvicorn

---

## ⚙️ Instalação e execução

### Clonar o repositório

```bash
git clone https://github.com/Hackaton-ONE/hackathon-DataScience.git
cd hackathon-DataScience/api
```

## Criar ambiente virtual (opcional, recomendado)

```
python -m venv venv
venv\Scripts\activate  # Windows
```

### Instalar dependências
```
pip install -r requirements.txt
```

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

O idioma pode ser definido via query param:

```
?lang=pt
?lang=es
```
Caso não seja informado, o idioma padrão é Português (pt).

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
    "Entrega atrasou muito"
  ],
  "lang": "pt"
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
curl -X POST "http://localhost:8000/sentiment/analyze?lang=pt" \
  -F "file=@comentarios.csv" \
  -o resultado.csv

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

- Modelos treinados separadamente para PT e ES

- Serialização com joblib

- Otimizados para:

  - Baixa latência

  - Execução em batch

  - Uso em APIs REST

## Validações implementadas

- Verificação de formato (JSON / CSV)

- Validação da coluna text em CSV

- Validação de texto mínimo

- Tratamento de erros amigável

Respostas consistentes

## Performance

- Batch prediction (vetorizado)

- Streaming de CSV em chunks

- Sem processamento desnecessário por linha

- Ideal para arquivos grandes (>100k linhas)

### Exemplo real:

  - ~1 MB CSV → ~0.6–0.8 segundos
  
  - CPU local, sem paralelismo extra
