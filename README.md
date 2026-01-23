# Sentiment Analysis ML

> **Autor:** Vicente Venancio Pascoal  
> Modelo de Machine Learning utilizando Logistic Regression para predição de sentimentos em reviews de e-commerce.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Principais Diferenciais](#principais-diferenciais)
- [Stack Tecnológico](#stack-tecnológico)
- [Instalação e Execução](#instalação-e-execução)
  - [Pré-requisitos](#pré-requisitos)
  - [Setup](#setup)
  - [Executar Código](#executar-código)
- [Arquitetura e Desenvolvimento](#arquitetura-e-desenvolvimento)
  - [1. Carregamento do Dataset](#1-carregamento-do-dataset)
  - [2. Exclusão de Colunas](#2-exclusão-de-colunas)
  - [3. Tratamento da Coluna Alvo](#3-tratamento-da-coluna-alvo)
  - [4. Pré-processamento de Texto](#4-pré-processamento-de-texto)
  - [5. Separação do Dataset](#5-separação-do-dataset)
  - [6. Modelo Base Multiclasse](#6-modelo-base-multiclasse)
  - [7. Modelo com GridSearchCV](#7-modelo-com-gridsearchcv)
  - [8. Amostragem Balanceada](#8-amostragem-balanceada)
  - [9. Modelo Binário](#9-modelo-binário)
- [Resultados](#resultados)

---

## Visão Geral

Esta aplicação foi desenvolvida para realizar predição de sentimentos em reviews de e-commerce. O modelo recebe comentários relacionados a avaliações de lojas e produtos, classificando-os automaticamente em categorias de sentimento.

**Objetivo:** Análise de sentimentos multilíngue (PT e ES) para auxiliar empresas a identificar e priorizar automaticamente feedbacks negativos em grande volume, permitindo ação rápida sobre problemas críticos reportados por clientes.

## Principais Diferenciais

- **Otimização para Português Brasileiro:** Modelo treinado considerando gírias, variações linguísticas e textos informais característicos do idioma
- **Pipeline de Pré-processamento Avançado:** Inclui normalização, tratamento de emojis e limpeza contextual do texto
- **Avaliação Robusta:** Métricas detalhadas incluindo Precision, Recall e F1-score para cada classe
- **Classificação Multiclasse:** Capacidade de identificar sentimentos positivos, negativos e neutros
- **Abordagem Binária Otimizada:** Modelo binário com acurácia de 87% para casos de uso específicos
- **Comparação de Modelos:** Implementação de múltiplas abordagens de ML com análise comparativa de desempenho

---

## Stack Tecnológico

- **Python 3.11+**
- **Scikit-Learn** - Inferência e modelagem vetorizada
- **imbalanced-learn** - Criação de pipelines balanceados
- **Pandas** - Manipulação eficiente de dados
- **NLTK** - Processamento de linguagem natural
- **NumPy** - Computação numérica

---

## Instalação e Execução

### Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)
- Git

### Setup

```bash
# Clonar o repositório
git clone https://github.com/Hackaton-ONE/hackathon-DataScience.git
cd hackathon-DataScience

# Trocar para a branch do projeto
git checkout Modelo_Sentimento_Vicente

# Instalar dependências (recomenda-se usar ambiente virtual)
pip install -r requirements.txt
```

### Executar Código

1. **Carregamento do Dataset:**
   - Baixe o dataset no link: [DATASET - Kaggle](https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets)
   - Selecione a versão **concatenated.csv**
   - Crie uma pasta chamada `datasets` na raiz do projeto
   - Coloque o arquivo baixado dentro da pasta `datasets`

2. **Execução:**
   - Abra o notebook Jupyter
   - Execute todas as células clicando em `Run All` ou `Executar Tudo`

---

## Arquitetura e Desenvolvimento

### 1. Carregamento do Dataset

Carregamento inicial dos dados utilizando a biblioteca Pandas:

```python
df = pd.read_csv("./datasets/concatenated.csv")
```

### 2. Exclusão de Colunas

Remoção de colunas desnecessárias para otimizar o treinamento:

```python
df.drop(columns=["dataset", "original_index", "review_text_processed", 
                 "review_text_tokenized", "rating", "kfold_polarity", 
                 "kfold_rating"], inplace=True)
```

### 3. Tratamento da Coluna Alvo

Padronização da coluna de polaridade para categorias textuais:

```python
df["polarity"].replace({1: "positivo", 0: "negativo", np.nan: "neutro"}, inplace=True)
df["sentimento"] = df["polarity"]
df.drop(columns=["polarity"], inplace=True)
```

### 4. Pré-processamento de Texto

Pipeline de limpeza e normalização dos textos:

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

### 5. Separação do Dataset

Divisão dos dados para treinamento e validação:

```python
x = df["review_text"]
y = df["sentimento"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
```

### 6. Modelo Base Multiclasse

Implementação do pipeline inicial com TF-IDF e Logistic Regression:

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

**Resultado:** Acurácia de 69%

### 7. Modelo com GridSearchCV

Otimização de hiperparâmetros utilizando Grid Search:

```python
param_grid = {
    'tfidf__ngram_range': [(1,2)],
    'clf__C': [1, 10],
    'clf__penalty': ['l2']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification Report:", classification_report(y_test, y_pred))
```

**Resultado:** Acurácia de 76%

### 8. Amostragem Balanceada

Criação de dataset balanceado com 200 mil amostras por classe:

```python
df_small = (
    df
    .groupby("sentimento", group_keys=False)
    .apply(lambda x: x.sample(200_000, random_state=42))
)

x = df_small["review_text"]
y = df_small["sentimento"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Amostras usadas para treinamento: {len(x_train)}")
```

**Treinamento com Grid Search:**

```python
grid_search = GridSearchCV(
    pipeline, 
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=3,
    n_jobs=-1,
    verbose=2 
)

grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

print(f"Classification Report: {classification_report(y_test, y_pred)}")
```

**Resultado:** Acurácia de 67%

### 9. Modelo Binário

Devido à confusão entre classes (especialmente "neutro"), foi desenvolvido um modelo binário focado apenas em sentimentos positivos e negativos:

```python
# Remoção da classe "neutro"
df_small_2 = df_small[df_small["sentimento"] != "neutro"]

x_2 = df_small_2["review_text"]
y_2 = df_small_2["sentimento"]

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(
    x_2, y_2, test_size=0.2, random_state=42
)

# Treinamento com GridSearchCV
grid_search_2 = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=3,
    n_jobs=-1,
    verbose=3
)

grid_search_2.fit(x_train_2, y_train_2)

best_model = grid_search_2.best_estimator_
y_pred_2 = best_model.predict(x_test_2)

print(f"Classification Report: {classification_report(y_test_2, y_pred_2)}")
```

**Resultado:** Acurácia de **87%**

---

## Resultados

| Modelo | Configuração | Acurácia |
|--------|--------------|----------|
| Baseline Multiclasse | TF-IDF + LogReg | 69% |
| GridSearch Multiclasse | TF-IDF + LogReg otimizado | 76% |
| Amostragem Balanceada | Dataset reduzido | 67% |
| **Modelo Binário** | **Apenas Positivo/Negativo** | **87%** |

### Principais Insights

- A classe "neutro" introduz significativa confusão no modelo multiclasse
- O modelo binário apresenta melhor desempenho para casos de uso que necessitam apenas distinguir entre sentimentos positivos e negativos
- A otimização de hiperparâmetros com GridSearchCV melhora consistentemente o desempenho
- O pré-processamento contextual (ex: normalização de nomes de lojas) contribui para a acurácia

---
