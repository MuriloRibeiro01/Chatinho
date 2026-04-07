import json, pickle, numpy as np, nltk, random
from keras.models import load_model
from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()
palavras = pickle.load(open("palavras.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
modelo = load_model("modelo.keras")

with open("intencoes.json", encoding="utf-8") as f:
    dados = json.load(f)

def prever_intencao(texto):
    tokens = nltk.word_tokenize(texto.lower(), language="portuguese")
    stems = [stemmer.stem(t) for t in tokens]
    bag = np.array([[1 if p in stems else 0 for p in palavras]])
    pred = modelo.predict(bag, verbose=0)[0]
    return classes[np.argmax(pred)], max(pred)

def responder(texto):
    intencao, confianca = prever_intencao(texto)
    if confianca < 0.7:
        return "Não entendi. Pode explicar novamente?"
    for i in dados["intencoes"]:
        if i["tag"] == intencao:
            return random.choice(i["respostas"])
        
print("Chatbot iniciado! (Digite 'sair' para encerrar)")
while True:
    msg = input("Você: ")
    if msg.lower() == "sair":
        break
    print("Bot: ", responder(msg))