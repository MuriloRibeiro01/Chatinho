# 🤖 Chatinho com Keras e NLTK

Projeto acadêmico de um chatbot baseado em classificação de intenções, utilizando uma rede neural treinada com o framework **Keras** e pré-processamento de linguagem natural com **NLTK**.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Funciona](#como-funciona)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Como Adicionar Novas Intenções](#como-adicionar-novas-intenções)
- [Detalhes Técnicos](#detalhes-técnicos)
- [Dependências](#dependências)

---

## Visão Geral

O chatbot funciona em duas fases bem separadas:

- **Fase de Treino:** o modelo aprende com os exemplos definidos manualmente e é salvo em disco.
- **Fase de Uso:** o modelo salvo é carregado e utilizado para responder ao usuário em tempo real.

A abordagem central é a **classificação de intenções**: dado um texto do usuário, a rede neural identifica *qual é a intenção* por trás da mensagem e retorna uma resposta adequada.

---

## Estrutura do Projeto

```
chatbot/
├── intencoes.json   # Dados de treino: padrões e respostas por intenção
├── treinar.py       # Pré-processamento + definição + treino da rede neural
├── chatbot.py       # Interface de conversa (usa o modelo já treinado)
├── modelo.keras     # Modelo salvo após o treino (gerado automaticamente)
├── palavras.pkl     # Vocabulário salvo após o treino (gerado automaticamente)
└── classes.pkl      # Lista de intenções salva após o treino (gerado automaticamente)
```

> Os arquivos `.keras` e `.pkl` são gerados automaticamente ao rodar `treinar.py`. Não é necessário criá-los manualmente.

---

## Como Funciona

### 1. Dados de Treino (`intencoes.json`)

Cada **intenção** representa um tipo de mensagem que o usuário pode enviar. Para cada intenção, são definidos:

- **`tag`**: nome identificador da intenção (ex: `"saudacao"`)
- **`padroes`**: exemplos de frases que o usuário poderia digitar
- **`respostas`**: possíveis respostas que o bot pode dar

Quanto mais variações de padrões forem fornecidas, melhor o modelo aprende a reconhecer aquela intenção.

### 2. Pré-processamento (`treinar.py`)

Antes de treinar a rede, o texto precisa ser convertido em números. Isso é feito em etapas:

**a) Tokenização:** cada frase é quebrada em palavras individuais.

```
"Bom dia!" → ["Bom", "dia", "!"]
```

**b) Stemming:** palavras são reduzidas ao seu radical, para que variações sejam tratadas como a mesma coisa.

```
"correndo", "correu", "correr" → "corr"
```

**c) Bag of Words:** cada frase é convertida em um vetor binário. O vetor tem o tamanho do vocabulário total, com `1` nas posições das palavras que aparecem na frase e `0` nas demais.

```
Vocabulário: ["bom", "dia", "oi", "tchau"]
"Bom dia"  → [1, 1, 0, 0]
"Oi"       → [0, 0, 1, 0]
```

Esse vetor é a **entrada (X)** da rede neural. A **saída (y)** é um vetor *one-hot* indicando a intenção correta.

### 3. Arquitetura da Rede Neural (`treinar.py`)

A rede é do tipo **Sequential** com as seguintes camadas:

| Camada | Tipo | Função |
|---|---|---|
| Entrada | `Dense` + ReLU | Aprende padrões nos vetores de entrada |
| `Dropout` | Regularização | Evita que o modelo decore os dados de treino |
| Oculta | `Dense` + ReLU | Aprofunda o aprendizado de padrões |
| `Dropout` | Regularização | Mesma função da anterior |
| Saída | `Dense` + Softmax | Retorna a probabilidade de cada intenção |

A função de perda usada é `categorical_crossentropy`, adequada para classificação com múltiplas classes. O otimizador é `adam`.

### 4. Inferência (`chatbot.py`)

Quando o usuário digita uma mensagem, o chatbot:

1. Aplica o mesmo pré-processamento do treino (tokenização + stemming + bag of words)
2. Passa o vetor pelo `modelo.predict()`
3. Identifica a intenção com maior probabilidade
4. Verifica se a confiança é suficiente (threshold de 0.7)
5. Retorna uma resposta aleatória daquela intenção

> ⚠️ O pré-processamento no `chatbot.py` deve ser **idêntico** ao do `treinar.py`. Qualquer diferença gera vetores incompatíveis e respostas erradas.

---

## Instalação

**1. Clone o repositório e entre na pasta:**
```bash
cd chatbot
```

**2. Crie e ative um ambiente virtual:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

**3. Instale as dependências:**
```bash
pip install tensorflow nltk
```

---

## Como Usar

**Passo 1 — Treinar o modelo:**
```bash
python treinar.py
```

Você verá a acurácia evoluindo a cada epoch. Ao final, os arquivos `modelo.keras`, `palavras.pkl` e `classes.pkl` serão gerados.

**Passo 2 — Iniciar o chatbot:**
```bash
python chatbot.py
```

Digite mensagens e o bot responderá. Para encerrar, digite `sair`.

---

## Como Adicionar Novas Intenções

1. Abra o `intencoes.json`
2. Adicione um novo bloco seguindo a estrutura existente, com `tag`, `padroes` e `respostas`
3. Forneça **pelo menos 5 variações** de padrões para que o modelo aprenda bem
4. Rode `treinar.py` novamente para atualizar o modelo
5. Rode `chatbot.py` normalmente

> Não é necessário alterar nenhum outro arquivo. O treino reconstrói o vocabulário e as classes automaticamente a partir do JSON.

### Depurando respostas incorretas

Se o bot não reconhecer bem uma intenção nova, adicione este print temporário no `chatbot.py` para inspecionar o que o modelo está prevendo:

```python
intencao, confianca = prever_intencao(texto)
print(f"[DEBUG] Intenção: {intencao} | Confiança: {confianca:.2f}")
```

- **Confiança baixa:** adicione mais variações de padrão no JSON e treine novamente.
- **Intenção errada:** os padrões estão se confundindo com outra intenção — torne-os mais específicos e distintos.

---

## Detalhes Técnicos

| Item | Detalhe |
|---|---|
| Framework ML | Keras (via TensorFlow) |
| Linguagem | Python 3.10+ |
| Pré-processamento | NLTK (tokenização + stemming RSLP para português) |
| Representação do texto | Bag of Words (vetor binário) |
| Tipo de rede | Feedforward (MLP) |
| Classificação | Multiclasse com Softmax |
| Serialização | `modelo.save()` para a rede; `pickle` para vocabulário e classes |

---

## Dependências

| Pacote | Versão recomendada | Função |
|---|---|---|
| `tensorflow` | 2.15+ | Framework de deep learning (inclui Keras) |
| `nltk` | 3.8+ | Tokenização e stemming em português |

**Recursos do NLTK necessários** (baixados automaticamente ao rodar `treinar.py`):

```python
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('rslp')
```
