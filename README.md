# Hackathon-DataScience

# Sentiment Analysis ML

> By: Vicente Venancio Pascoal </br>
> Modelo de Machine Learning utilizando Logistic Regression para predição de sentimentos.

---

## Índice:

- [Visão Geral](#visão-geral)
  - [Principais Diferenciais](#principais-diferenciais)
- [Stack Tecnológico](#stack-tecnológico)
- [Instalação e Execução](#instalação-e-execução)
- [Endpoint](#endpoint)
- [Decisões de Arquitetura](#decisões-de-arquitetura)

---

## Visão Geral

Esta aplicação foi desenhada para fazer predição de reviews de e-commerce. O modelo recebe um comentário relacionado a reviews de lojas e produtos os classifica automaticamente.

O objetivo é análise de sentimentos multilíngue (PT e ES) para ajudar empresas a identificar e priorizar automaticamente feedbacks negativos em grande volume, permitindo ação rápida sobre problemas críticos de clientes.

## Principais Diferenciais

* Modelo otimizado para análise de sentimentos em português brasileiro, considerando gírias, variações linguísticas e textos informais.
* Pipeline de pré-processamento avançado, incluindo normalização, tratamento de emojis e limpeza contextual do texto.
* Avaliação robusta do desempenho do modelo com métricas como Precision, Recall e F1-score.
* Modelos com classificação em múltiplas categorias de sentimento, indo além da abordagem binária.
* Implementação de múltiplos modelos de Machine Learning, permitindo a comparação entre abordagens clássicas e modernas para análise de sentimentos.

---

## Stack Tecnológico

* **Python 3.11+**
* **Scikit-Learn** (Inferência Vetorizada)
* **imblearn** (Criação de Pipelines)
* **Pandas** (Manipulação eficiente de dados)

---

## Instalação e Execução

### Setup

```bash
# Clonar o repositório
git clone [https://github.com/Hackaton-ONE/hackathon-DataScience.git](https://github.com/Hackaton-ONE/hackathon-DataScience.git)
cd hackathon-DataScience

# Trocar para branch
git checkout Modelo_Sentimento_Vicente
```
## Executar Código
1. **Carregamento dataset:** Pra fazer o carregamento do dataset primeiro você deve baixá-lo no seguinte link [DATASET](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets). Após entrar no link baixe a versão **concatenated.csv**. </br> Crie uma pasta chamada dataset e o coloque nela.
2. **Execução:** Por estar dentro de um notebook a execução do código é bem fácil. A única coisa é clicar em `Run All`

---

## Explicação código

### Carregamento do dataset 
Nesta etapa nós carregamos o dataset através da biblioteca pandas

```python
df = pd.read_csv("./datasets/concatenated.csv")
```

### Exclusão de colunas 
Para um melhor aprendizado do nosso modelo precisamos excluir algumas colunas

```python
df.drop(columns=["dataset", "original_index", "review_text_processed", "review_text_tokenized", "rating", "kfold_polarity", "kfold_rating"], inplace=True)
```

### Tratamento Coluna alvo

```python
df["polarity"].replace({1: "positivo", 0: "negativo", np.nan: "neutro"}, inplace=True)
df["sentimento"] = df["polarity"]
df.drop(columns=["polarity"], inplace=True)
```

### Tratamento dataset

```python
def clean_text(text):
    text = str(text).lower()
    text = text.replace("lojas americanas", "loja")
    text = text.replace("americanas", "loja")
    text = text.replace("americana", "loja")
    text = re.sub(r"r?$ ?\d+([.,]\d+)?", "dinheiro", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"[^a-z0-9à-úç ]", "", text)
    text = " ".join(text.split())
    return text

nltk.download("stopwords")
stop_pt = stopwords.words("portuguese")

df.dropna(subset=['review_text', 'sentimento'], inplace=True)
df['review_text'] = df['review_text'].astype(str).apply(clean_text)
```

### Separação do dataset para Treinamento
Aqui separamos o dataset para realização do treinamento e validação

```python
x = df["review_text"]
y = df["sentimento"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
```

### Treinamenmento do Modelo Base
Nesta etapa criamos um pipeline para realizar o treinamento e por fim testamos a acurácia

```python
pipeline = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(stop_words=stopwords.words("portuguese"))),
        ("clf", LogisticRegression(class_weight="balanced", solver="saga")),
    ]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.4f}")
```

### Modelo multiclasse com GridSearch
Aqui utilizamos uma técnica para realizar o treinamento com alguns hiperparâmetros 

```python
param_grid = {
    'tfidf__ngram_range': [(1,2)],
    'clf__C': [1, 10],
    'clf__penalty': ['l2']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid = param_grid,
    cv = 3,
    n_jobs = -1, 
    verbose = 1
)

grid_search.fit(X_train, y_train)
```

Analise do modelo usando classification report 

```python
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("classification_report:", classification_report(y_test, y_pred))
```

### Modelo base multiclasse com pequena amostra do Dataset
Para esta etapa foi criado uma amostra do dataset com 200 mil comentários para cada classe, com o intuito de melhorar o treinamento.

```python
df_small = (
    df
    .groupby("sentimento", group_keys=False)
    .apply(lambda x: x.sample(200_000, random_state=42))
)
```

divisão da amostra 

```python
x = df_small["review_text"]
y = df_small["sentimento"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Amostras usadas para treinamento: {len(x_train)}")
```

treinamento e analise de métricas

```python
pipeline.fit(x_train, y_train)

y_predict = pipeline.predict(x_test)

print("Classification Report:", classification_report(y_test, y_predict))
```

### Modelo base multiclasse com GridSearch e pequena amostra do Dataset 

```python
param_grid = {
    "tfidf__ngram_range": [(1,2)],
    "clf__C": [1, 10],
    "clf__penalty": ["l2"]
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid = param_grid,
    scoring = "balanced_accuracy",
    cv = 3,
    n_jobs = -1,
    verbose = 2 
)

grid_search.fit(x_train, y_train)
```

analise do modelo 

```python
best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test)

print(f"Classification report: {classification_report(y_test, y_pred)}")
```

### Modelo binário 
Analisando os modelos anteriores vimos que o modelo gera uma confusão grande entre as classificações, principalmente o neutro. Pensando nisso excluimos essa classe e tranformamos o modelo em binário.

---
## Decisões de Arquitetura

1. - **Vetorização vs Loops:** Utilizamos `model.predict_proba(lista_inteira)` ao invés de iterar linha por linha.
  Isso delega o cálculo matemático para as bibliotecas em C (NumPy/BLAS), acelerando o processo em até 100x.

2. - **Streaming & Generators:** Para CSVs, utilizamos Python Generators (`yield`).
  A API lê blocos de 5.000 linhas, processa, devolve e limpa da memória. Isso impede erros de Out of Memory (OOM).
  
3. - **IO Bound Optimization:** O uso de arquivos temporários e leitura otimizada (`chunksize`) garante que a CPU nunca fique ociosa esperando leitura de disco.

4. - **Smart Column Detection (Heurística):**
   Sabemos que datasets reais são bagunçados. Implementamos uma `heurística` que, na ausência de cabeçalhos padrão,
  varre o arquivo em busca de colunas não-numéricas com média de caracteres > 20.
  Isso permite processar CSVs "sujos" ou sem padronização sem quebrar a pipeline.
