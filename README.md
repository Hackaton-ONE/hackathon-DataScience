# Hackathon-DataScience

# Sentiment Analysis API (High-Performance MVP)

> Uma API robusta, vetorizada e assíncrona para análise de sentimentos em PT/ES, capaz de processar **+14.000 requisições/segundo**.

---

## Índice:

- [Visão Geral](#visão-geral)
  - [Principais Diferenciais](#principais-diferenciais)
- [Benchmark de Performance](#benchmark-de-performance)
- [Stack Tecnológico](#stack-tecnológico)
- [Instalação e Execução](#instalação-e-execução)
- [Endpoint](#endpoint)
  - [JSON (Single)](#a-json-texto-únicosingle---ideal-para-chatbots)
  - [JSON (Batch)](#b-json-batch---lista-de-textos)
  - [CSV (Streaming / cURL)](#c-arquivo-csv-big-data--bi)
- [Decisões de Arquitetura](#decisões-de-arquitetura)

---

## Visão Geral

Esta aplicação foi desenhada para resolver o gargalo comum em deploy de modelos de ML: **latência e escalabilidade**.

Diferente de abordagens tradicionais (loops linha-a-linha), esta API utiliza **processamento vetorial em lote**, **streaming de dados** e **cache em memória**, permitindo a análise de arquivos gigantescos com consumo mínimo de memória RAM e latência baixíssima.

## Principais Diferenciais

* **Dual Language:** Suporte nativo a Português (PT) e Espanhol (ES).
* **True Streaming:** Processa arquivos CSV muito maiores que a memória RAM disponível (leitura em *chunks*).
* **High Performance:** Vetorização com Scikit-Learn e NumPy.
* **Smart Caching:** Cache LRU para predições repetidas (latência zero para textos frequentes).
* **Robustez:** Parser de CSV "blindado" contra erros de codificação e colunas duplicadas.
* **Compressão:** GZip middleware para otimização de banda de rede.

---

## Benchmark de Performance

Testes de carga realizados em ambiente local (CPU padrão):

| Carga (Linhas) | Tamanho (CSV) | Tempo Total | Throughput (Req/s) |
| :--- | :--- | :--- | :--- |
| **50.000** | ~14 MB | 3.6 segundos | **~13.800** |
| **100.000** | ~28 MB | 6.9 segundos | **~14.500** |

> *Nota: A arquitetura mantém o consumo de memória constante, independentemente do tamanho do arquivo de entrada.*

---

## Stack Tecnológico

* **Python 3.11+**
* **FastAPI** (High-performance web framework)
* **Scikit-Learn** (Inferência Vetorizada)
* **Pandas** (Manipulação eficiente de dados)
* **Joblib** (Serialização de Modelos)

---

## Instalação e Execução

### 1. Setup

```bash
# Clonar o repositório
git clone [https://github.com/Hackaton-ONE/hackathon-DataScience.git](https://github.com/Hackaton-ONE/hackathon-DataScience.git)
cd hackathon-DataScience/api

# Criar ambiente virtual
python -m venv venv

# Ativar (Windows)
venv\Scripts\activate
# Ativar (Linux/Mac)
source venv/bin/activate

# Instalar dependências otimizadas
pip install -r requirements.txt
```
## Executar a API
```
uvicorn main:app --reload
```
Acesse a documentação interativa (Swagger UI): [http://127.0.0.1:8000](http://localhost:8000/docs)

---

## Endpoint

> `POST /sentiment/analyze`

Ponto único de entrada que aceita tanto JSON quanto Arquivos CSV.

O idioma padrão é `pt`, mas pode ser alterado via query param `?lang=es`.

### Exemplos de uso:

### A. JSON (Texto Único/Single - Ideal para Chatbots)

**Requisição:**

```JSON
// Request
{
  "text": "O atendimento foi excelente e a entrega rápida!",
  "lang": "pt"
}
```

**Resposta:**

```JSON
// Response
{
  "idioma": "pt",
  "previsao": "Positivo",
  "probabilidade": 0.9850
}
```

### B. JSON (Batch - Lista de Textos)

**Requisição:**

```JSON
// Request
{
  "texts": ["Adorei o produto", "Demorou muito", "Qualidade média"]
}
```

**Resposta:**

```JSON
// Response
{
[
  {
    "idioma":"pt",
    "previsao":"Positivo",
    "probabilidade":0.1698
  },
    {
      "idioma":"pt",
      "previsao":"Positivo",
      "probabilidade":0.537
    },
    {
      "idioma":"pt",
      "previsao":"Positivo",
      "probabilidade":0.2081
    }
]
}
```

---

### C. Arquivo CSV (Big Data / BI)

Ideal para processar históricos de atendimento.

O arquivo é processado via streaming e o download inicia imediatamente.

## Exemplo via cURL:

**Requisição:**

```Bash
curl -X POST "http://localhost:8000/sentiment/analyze?lang=pt" \
  -F "file=@meu_dataset_gigante.csv" \
  -o resultado_analise.csv
```

**Formato do CSV de Saída:** A API limpa automaticamente colunas duplicadas e retorna um CSV enxuto:

**Resposta:**

```Snippet de código
text,idioma,previsao,probabilidade
"Adorei o produto",pt,Positivo,0.98
"Péssimo serviço",pt,Negativo,0.92
```

---

## Decisões de Arquitetura

1 - **Vetorização vs Loops:** Utilizamos `model.predict_proba(lista_inteira)` ao invés de iterar linha por linha.
  Isso delega o cálculo matemático para as bibliotecas em C (NumPy/BLAS), acelerando o processo em até 100x.

2 - **Streaming & Generators:** Para CSVs, utilizamos Python Generators (`yield`).
  A API lê blocos de 5.000 linhas, processa, devolve e limpa da memória. Isso impede erros de Out of Memory (OOM).
  
3 - **IO Bound Optimization:** O uso de arquivos temporários e leitura otimizada (`chunksize`) garante que a CPU nunca fique ociosa esperando leitura de disco.
