import json
import os

TASKS_FILE = "tarefas.json"

def carregar_tarefas():

    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
        
    return []

def salvar_tarefas(tarefas):

    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            tarefas,
            f,
            ensure_ascii=False,
            indent=2
        )