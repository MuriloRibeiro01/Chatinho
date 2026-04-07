import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import numpy as np
import random
import nltk
import pickle
from nltk.stem import PorterStemmer

# 1. SETUP INICIAL
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
stemmer = PorterStemmer()

# 2. CARREGAR DADOS
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documentos = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documentos.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set([stemmer.stem(w.lower()) for w in words if w not in ignore_words])))
classes = sorted(list(set(classes)))

# 3. VETORIZAÇÃO (PREPARAÇÃO PARA TREINO)
training = []
output_empty = [0] * len(classes)

for doc in documentos:
    bag = []
    pattern_words = [stemmer.stem(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# 4. CONSTRUÇÃO E TREINO DO MODELO
model = keras.Sequential([
    layers.Input(shape=(len(train_x[0]),)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- Iniciando Treinamento ---")
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print("Modelo treinado e salvo com sucesso!\n")

# 5. FUNÇÕES PARA O CHAT
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [stemmer.stem(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: bag[i] = 1
    return np.array([bag])

# 6. LOOP DE CONVERSA (MODO CHAT)
print("--- BOT PRONTO PARA CONVERSAR (digite 'sair' para parar) ---")

while True:
    pergunta = input("Você: ")
    if pergunta.lower() == "sair":
        break

    input_vector = bow(pergunta, words)
    res = model.predict(input_vector, verbose=0)[0]
    
    # Pega o melhor palpite
    idx = np.argmax(res)
    probabilidade = res[idx]
    
    if probabilidade > 0.5: # Só responde se tiver mais de 50% de certeza
        tag = classes[idx]
        for i in intents['intents']:
            if i['tag'] == tag:
                print(f"Bot: {random.choice(i['responses'])} (Confiança: {probabilidade:.2f})")
    else:
        print("Bot: Não entendi muito bem, pode repetir?")