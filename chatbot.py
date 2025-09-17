#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install keras


# In[14]:


pip install tensorflow


# In[15]:


pip install tensorflow


# In[6]:


{"intents": [
    {"tag": "greeting",
     "patterns": ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello", "Good day"],
     "responses": ["Hello", "Good to see you again", "Hi there, how can I help?"],
     "context": [""]
    },
    {"tag": "goodbye",
     "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
     "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
     "context": [""]
    },
    {"tag": "thanks",
     "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
     "responses": ["My pleasure", "You're Welcome"],
     "context": [""]
    },
    {"tag": "query",
     "patterns": ["What is Simplilearn?"],
     "responses": ["Simplilearn is the popular online Bootcamp & online courses learning platform "],
     "context": [""]
    } 
]}


# In[8]:


import random 
import json 
import pickle
import numpy as np
import tensorflow as tf

import nltk 
from nltk.stem import WordNetLemmatizer

lemmatizer = WorldNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
class = []
document = []
ignorLetter = ['?','!','.',',']

for intent in intents['intents']
    for pattern in['patterns']:
        wordList = nltk.world_tokenize(pattern)
        words.extend(worldList)
        douctment.append((wordList, intent['tag']))
        if intent['tag'] not in classe:
           class.append(intent['tag'])
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetter
words = sorted(set(classes))

classes = sorted(set(classes))

picle.dump(words,open('words.plk','web'))
picle.dump(classes,open('classes.plk','web'))

training = []
outputEmpty = [0] * len(classes)

for docucument in docucments:
    bag = []
    worldpatterns = document[0]
    wordpatterns = [lemmatizer.lemmatize(world.lower()) for word in wordpatterns]
    for word in words: bag.append(1) if word in  wordpatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.indexdoucment[1]] = 1
    trainning.append(bag + outputRow)

radom.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.sequential()

model.add(tf.keras.layer.Dense(128,input_shape = (len(trainx[0],),),activation = 'relu'))
model.add(tf.keras.layer.Dropout(0.5))
model.add(tf.keras.layer.Dense(64, activation = 'relu'))
model.add(tf.keras.layer.Dense(len(trainY[0]),activation = "softmax"))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum = 0.9, nesterov=True)

model.compile(loss= 'categorical_crossentropy', optimizer = sgd, metrics= ['accuracy'])
hist = model.fit(np.array(trainX),np.array(trainY),epochs = 200, batch_size = 5,
model.save('chatboat_simplilearnmodel.hs', hist)
print("Executed")
        


# In[20]:


import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:\Simplilearn\Python\Python projects\chatbot using python\chatbot\intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class (message)
    res = get_response (ints, intents)
    print (res)
    


# In[18]:


import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:\Simplilearn\Python\Python projects\chatbot using python\chatbot\intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class (message)
    res = get_response (ints, intents)
    print (res)
    


# In[16]:


import random
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Intents data (embedded instead of reading from intents.json)
intents = {
    "intents": [
        {"tag": "greeting",
         "patterns": ["Hi there", "How are you", "Is anyone there?", "Hey", "Hola", "Hello", "Good day"],
         "responses": ["Hello", "Good to see you again", "Hi there, how can I help?"],
         "context": [""]},

        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
         "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
         "context": [""]},

        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
         "responses": ["My pleasure", "You're Welcome"],
         "context": [""]},

        {"tag": "query",
         "patterns": ["What is Simplilearn?"],
         "responses": ["Simplilearn is the popular online Bootcamp & online courses learning platform"],
         "context": [""]}
    ]
}

# Load pre-trained model, words, and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence to bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get bot response
def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "I'm not sure I understand. Can you rephrase?"
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return "Sorry, I don't know about that."

print("GO! Bot is running!")

# Chat loop
while True:
    message = input("You: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)


# In[28]:


import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download nltk resources (only the first time)
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Intents data
intents = {
    "intents": [
        {"tag": "greeting",
         "patterns": ["Hi there", "How are you", "Is anyone there?", "Hey", "Hola", "Hello", "Good day"],
         "responses": ["Hello", "Good to see you again", "Hi there, how can I help?"],
         "context": [""]},

        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
         "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
         "context": [""]},

        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
         "responses": ["My pleasure", "You're Welcome"],
         "context": [""]},

        {"tag": "query",
         "patterns": ["What is Simplilearn?"],
         "responses": ["Simplilearn is the popular online Bootcamp & online courses learning platform"],
         "context": [""]}
    ]
}

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize each pattern
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print("Words:", words)
print("Classes:", classes)

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save model
model.save('chatbot_model.h5')
print("Model trained and saved as chatbot_model.h5")


# In[24]:


import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')


# In[25]:


pip install nltk_data


# In[30]:


import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize lemmatizer and load resources
lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r'C:\Simplilearn\Python\Python projects\chatbot using python\chatbot\intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# 🔍 Preprocess input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# 🧠 Convert sentence to bag-of-words format
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# 🔮 Predict intent class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# 💬 Generate response based on predicted intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understand. Can you rephrase?"
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# 🚀 Start chatbot
print("🤖 GO! Bot is running. Type something to begin chatting...")

while True:
    message = input("You: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)


# In[ ]:


import random
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Download NLTK data (only needed first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Intents data embedded directly (no need for intents.json file)
intents = {
    "intents": [
        {"tag": "greeting",
         "patterns": ["Hi there", "How are you", "Is anyone there?", "Hey", "Hola", "Hello", "Good day"],
         "responses": ["Hello", "Good to see you again", "Hi there, how can I help?"],
         "context": [""]},

        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
         "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
         "context": [""]},

        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
         "responses": ["My pleasure", "You're Welcome"],
         "context": [""]},

        {"tag": "query",
         "patterns": ["What is Simplilearn?"],
         "responses": ["Simplilearn is the popular online Bootcamp & online courses learning platform"],
         "context": [""]},

        {"tag": "love",
         "patterns": ["i love you"],
         "responses": [" ilove you too"],
         "context": [""]}
    ]
}

# Load pre-trained model, words, and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


# Preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Convert sentence to bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Predict intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# Get bot response
def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "I'm not sure I understand. Can you rephrase?"
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return "Sorry, I don't know about that."


# Run chatbot
print("GO! Bot is running!")

while True:
    message = input("You: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)


# In[1]:


#  Import Libraries
import random
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Download NLTK Data (only needed once)
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Embedded Intents Data
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi there", "How are you", "Is anyone there?", "Hey", "Hola", "Hello", "Good day"],
            "responses": ["Hello!", "Good to see you again.", "Hi there, how can I help?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
            "responses": ["See you!", "Have a nice day!", "Bye! Come back again soon."]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
            "responses": ["My pleasure!", "You're welcome!"]
        },
        {
            "tag": "query",
            "patterns": ["What is Simplilearn?"],
            "responses": ["Simplilearn is a popular online Bootcamp and learning platform."]
        },
        {
            "tag": "love",
            "patterns": ["I love you"],
            "responses": ["I love you too!"]
        }
    ]
}

#  Load Pre-trained Model and Data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

#  Preprocess User Input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#  Convert Sentence to Bag of Words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#  Predict Intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

#  Generate Bot Response
def get_response(intents_list, intents_json):
    if not intents_list:
        return random.choice([
            "Hmm, I didn't catch that. Could you rephrase?",
            "I'm still learning. Can you try asking differently?",
            "Sorry, I don't understand that yet."
        ])
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

#  Run Chatbot
print("===================================")
print(" Welcome to SunnyBot!")
print("Type your message below. Type 'exit' to quit.")
print("===================================\n")

while True:
    message = input("You: ").strip()
    if message.lower() in ["exit", "quit", "bye"]:
        print("Bot: It was great chatting with you. Goodbye!")
        break
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    print(" Bot:", response)


# In[ ]:





# In[ ]:




