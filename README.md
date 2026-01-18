# Classificador de Emails

Sistema de classificação automática de emails em "Produtivo" ou "Improdutivo" usando Machine Learning.

## Estrutura do Projeto

```
app/
├── api.py                          # API Flask principal
├── requirements.txt                # Dependências Python
├── classificadores/
│   ├── base_de_dados.py           # Pré-processamento do dataset
│   ├── pre_processamento.py       # Limpeza e normalização de texto
│   ├── treinamento_modelo.py      # Treinamento do modelo ML
│   └── modelo_respostas.py        # Geração de respostas contextuais
├── database/
│   ├── emails_produtivos_improdutivos.csv  # Dataset original
│   └── emails_processados.csv              # Dataset processado
└── interface/
    └── index.html                  # Interface web
```

## Funcionamento

### 1. Pré-processamento (`pre_processamento.py`)
Limpa e normaliza os textos dos emails:
- Remove emails, URLs e números
- Converte para minúsculas
- Remove acentos e caracteres especiais
- Remove stopwords (palavras irrelevantes)
- Aplica lemmatização com spaCy

### 2. Preparação da Base de Dados (`base_de_dados.py`)
Processa o dataset original:
- Lê o arquivo CSV com emails rotulados
- Aplica pré-processamento em cada texto
- Remove linhas com valores nulos
- Salva o resultado em `emails_processados.csv`

### 3. Treinamento do Modelo (`treinamento_modelo.py`)
Treina o classificador:
- Carrega os emails pré-processados
- Balanceia as classes (Produtivo/Improdutivo)
- Divide em treino (80%) e teste (20%)
- Usa TF-IDF para vetorização de texto
- Treina uma Regressão Logística
- Salva o modelo treinado como `modelo_classificacao.pkl`

### 4. Gerador de Respostas (`modelo_respostas.py`)
Analisa o contexto do email e gera respostas personalizadas:
- Detecta problemas conhecidos (acesso, erro, performance, etc.)
- Identifica sistemas mencionados
- Calcula nível de urgência
- Sugere respostas apropriadas

### 5. API Flask (`api.py`)
Disponibiliza endpoints HTTP:
- `GET /` - Interface web
- `POST /api/classificar` - Classifica um email e retorna respostas sugeridas

## Instalação

### 1. Instalar dependências
```bash
pip install -r app/requirements.txt
```

### 2. Baixar modelo do spaCy
```bash
python -m spacy download pt_core_news_sm
```

## Executar

### 1. Processar o dataset
```bash
cd app/classificadores
python base_de_dados.py
```

### 2. Treinar o modelo
```bash
python treinamento_modelo.py
```

### 3. Iniciar a API
```bash
cd ..
python api.py
```

A API estará disponível em `http://localhost:5000`

## Usar a API

### Via interface web
Acesse `http://localhost:5000` no navegador

### Via requisição HTTP
```bash
curl -X POST http://localhost:5000/api/classificar \
  -H "Content-Type: application/json" \
  -d '{"texto": "Preciso de acesso urgente ao sistema"}'
```

### Resposta exemplo
```json
{
  "classificacao": "Produtivo",
  "confianca": 0.85,
  "respostas_sugeridas": [
    {
      "titulo": "Resposta Padrão de Priorização",
      "texto": "Prezado(a),\n\nRecebemos sua mensagem..."
    }
  ]
}
```

## Tecnologias Utilizadas

- **Python 3.x** - Linguagem principal
- **Flask** - Framework web
- **scikit-learn** - Machine Learning
- **spaCy** - Processamento de linguagem natural
- **pandas** - Manipulação de dados
- **joblib** - Persistência do modelo
