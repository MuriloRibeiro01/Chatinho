import json
import numpy as np
import pickle
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import nltk
from nltk.stem import RSLPStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('rslp')

stemmer = RSLPStemmer()

with open("intencoes.json", encoding="utf-8") as f:
    dados = json.load(f)

palavras = []
classes = []
documentos = []

for intencao in dados["intencoes"]:
    for padrao in intencao["padroes"]:
        tokens = nltk.word_tokenize(padrao.lower(), language="portuguese")
        palavras.extend(tokens)
        documentos.append((tokens, intencao["tag"]))
    if intencao["tag"] not in classes:
        classes.append(intencao["tag"])

palavras = sorted(set(stemmer.stem(p) for p in palavras if p.isalpha()))
classes = sorted(classes)

pickle.dump(palavras, open("palavras.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

X_treino = []
Y_treino = []

for tokens, tag in documentos:
    stems = [stemmer.stem(t) for t in tokens]
    bag = [1 if p in stems else 0 for p in palavras]
    label = [1 if c == tag else 0 for c in classes]
    X_treino.append(bag)
    Y_treino.append(label)

X_treino = np.array(X_treino)
Y_treino = np.array(Y_treino)

modelo = Sequential([
    Dense(64, input_shape=(len(palavras),), activation="relu"),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

modelo.summary()

historico = modelo.fit(X_treino, Y_treino, epochs=200, batch_size=8, verbose=1)

modelo.save("modelo.keras")
print("Modelo treinado e salvo.")