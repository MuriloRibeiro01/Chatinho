import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from embeddings import gerar_embedding
from tasks import carregar_tarefas
from chatbot import contexto

def buscar_tarefas_semelhantes(texto, limite=3, treshold=0.55):
    
    tarefas = carregar_tarefas()

    if not tarefas:
        return []
    
    consulta_embedding = gerar_embedding(texto)

    resultados = []

    for tarefa in tarefas:

        if "embedding" not in tarefa:
            continue

        # Acha similaridade com base nos vetores da tarefa
        similaridade = cosine_similarity(
            [consulta_embedding],
            [tarefa["embedding"]]
        )[0][0]

        if similaridade >= treshold:

            # Adiciona contexto para a tarefa com base na similaridade
            resultados.append({
                "tarefa": tarefa,
                "similaridade": float(similaridade)
            })
        
    contexto["ultima_busca"] = resultados
    
    # Ele organiza pela similaridade em ordem decrescente.
    resultados.sort(
        key=lambda x: x["similaridade"],
        reverse=True
    )

    return resultados[:limite]    