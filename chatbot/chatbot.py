import json, pickle, os, random
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import RSLPStemmer
from embeddings import gerar_embedding
from semantic_search import buscar_tarefas_semelhantes
from tasks import carregar_tarefas, salvar_tarefas

contexto = {
    "ultima_intencao": None,
    "ultimas_tarefas": None,
    "ultima_busca": []
  }

stemmer   = RSLPStemmer()
palavras  = pickle.load(open("palavras.pkl", "rb"))
classes   = pickle.load(open("classes.pkl", "rb"))
modelo    = load_model("modelo.keras")

with open("intencoes.json", encoding="utf-8") as f:
    dados = json.load(f)

TASKS_FILE = "tarefas.json"

# ---------- PREDIÇÃO ----------
def prever_intencao(texto):
    tokens = nltk.word_tokenize(texto.lower(), language="portuguese")
    stems  = [stemmer.stem(t) for t in tokens]
    bag    = np.array([[1 if p in stems else 0 for p in palavras]])
    pred   = modelo.predict(bag, verbose=0)[0]
    return classes[np.argmax(pred)], max(pred)

# ---------- TAREFAS ----------
def carregar_tarefas():
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def salvar_tarefas(tarefas):
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tarefas, f, ensure_ascii=False, indent=2)

def extrair_nome_tarefa(texto):
    keywords = ["lembrar de", "anotar", "adicionar", "criar", "incluir", "colocar", "adiciona", "cria", "anota"]
    texto_lower = texto.lower()
    for kw in sorted(keywords, key=len, reverse=True):
        if kw in texto_lower:
            idx   = texto_lower.index(kw) + len(kw)
            resto = texto[idx:].strip(" :.-")
            if resto:
                return resto
    return None

def cmd_criar_tarefa(texto):

    tarefas = carregar_tarefas()

    nome    = extrair_nome_tarefa(texto)    

    if not nome:
        print("Bot: Qual é o nome da tarefa?")
        nome = input("Você: ").strip()

    if not nome:
        print("Bot: Nenhum nome informado, tarefa não criada.")
        return
    
    embedding = gerar_embedding(nome).tolist()

    tarefas.append({
        "nome": nome, 
        "feita": False,
        "embedding": embedding
    })

    salvar_tarefas(tarefas)

    contexto["ultimas_tarefas"] = nome
    contexto["ultima_intencao"] = "criar_tarefa"

    print(f'Bot: Tarefa "{nome}" adicionada!')

def cmd_busca_semantica(texto):

    resultados = buscar_tarefas_semelhantes(texto)

    contexto["ultima_busca"] = resultados

    if not resultados:
        print("Bot: Não achei nenhuma tarefa relacionada.")
        return
    
    print("Bot: Achei essas tarefas relacionadas:")

    for i, r in enumerate(resultados, 1):
        print(f"{i}, {r['tarefa']['nome']}")


def cmd_listar_tarefas():
    tarefas = carregar_tarefas()
    if not tarefas:
        print("Bot: Nenhuma tarefa cadastrada ainda. Adicione uma!")
        return
    print("Bot: Suas tarefas:")
    for i, t in enumerate(tarefas, 1):
        status = "✓" if t["feita"] else "○"
        print(f"  {i}. [{status}] {t['nome']}")
    contexto["ultimas_tarefas"]

def cmd_concluir_tarefa():
    tarefas   = carregar_tarefas()
    pendentes = [(i, t) for i, t in enumerate(tarefas) if not t["feita"]]
    if not pendentes:
        print("Bot: Parabéns! Não há tarefas pendentes.")
        return
    print("Bot: Qual tarefa você concluiu? (digite o número)")
    for n, (_, t) in enumerate(pendentes, 1):
        print(f"  {n}. {t['nome']}")
    try:
        escolha = int(input("Você: ")) - 1
        idx     = pendentes[escolha][0]
        tarefas[idx]["feita"] = True
        salvar_tarefas(tarefas)
        print(f'Bot: Ótimo! "{tarefas[idx]["nome"]}" marcada como concluída!')
    except (ValueError, IndexError):
        print("Bot: Número inválido, tente novamente.")

def cmd_deletar_tarefa():
    tarefas = carregar_tarefas()
    if not tarefas:
        print("Bot: Não há tarefas para remover.")
        return
    print("Bot: Qual tarefa deseja remover? (digite o número)")
    for i, t in enumerate(tarefas, 1):
        status = "✓" if t["feita"] else "○"
        print(f"  {i}. [{status}] {t['nome']}")
    try:
        escolha  = int(input("Você: ")) - 1
        removida = tarefas.pop(escolha)
        salvar_tarefas(tarefas)
        print(f'Bot: Tarefa "{removida["nome"]}" removida!')
    except (ValueError, IndexError):
        print("Bot: Número inválido, tente novamente.")

TASK_HANDLERS = {
    "criar_tarefa":    cmd_criar_tarefa,
    "listar_tarefas":  lambda _: cmd_listar_tarefas(),
    "concluir_tarefa": lambda _: cmd_concluir_tarefa(),
    "deletar_tarefa":  lambda _: cmd_deletar_tarefa(),
    "buscar_tarefas_semelhantes": cmd_busca_semantica,
}

# ---------- LOOP PRINCIPAL ----------
print("Chatinho iniciado! (Digite 'sair' para encerrar)")

while True:
    msg = input("Você: ").strip()
    if not msg:
        continue
    if msg.lower() == "sair":
        print("Bot: Até mais!")
        break

    intencao, confianca = prever_intencao(msg)

    if confianca < 0.7:
        print("Bot: Não entendi. Pode explicar novamente?")
        continue

    if intencao in TASK_HANDLERS:
        TASK_HANDLERS[intencao](msg)
    else:
        for i in dados["intencoes"]:
            if i["tag"] == intencao:
                print("Bot:", random.choice(i["respostas"]))
                break
