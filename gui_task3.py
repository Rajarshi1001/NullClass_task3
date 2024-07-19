# import libraries
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from keras.layers import TextVectorization
import re
import requests
# import tensorflow.strings as tf_strings
import json
import string
from keras.models import load_model
from tensorflow import argmax
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import tensorflow as tf

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

## loading the tokenizers

model_hi = load_model('english_to_hindi_lstm_model')
model_fr = load_model('english_to_french_lstm_model')

#load Tokenizers

with open('english_tokenizer_hindi.json') as f:
    data = json.load(f)
    english_tokenizer_hindi = tokenizer_from_json(data)

with open('hindi_tokenizer.json') as f:
    data = json.load(f)
    hindi_tokenizer = tokenizer_from_json(data)
    
with open('english_tokenizer.json') as f:
    data = json.load(f)
    english_tokenizer = tokenizer_from_json(data)

with open('french_tokenizer.json') as f:
    data = json.load(f)
    french_tokenizer = tokenizer_from_json(data)
    
    
max_decoded_sentence_length = 20

with open('sequence_length_hindi.json') as f:
    max_length_hindi = json.load(f)
    
with open('sequence_length.json') as f:
    max_length = json.load(f)
    
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def translate_to_french(english_sentence):
    english_sentence = english_sentence.lower()
    
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length)
    
    english_sentence = english_sentence.reshape((-1,max_length))
    
    french_sentence = model_fr.predict(english_sentence)[0]
    
    french_sentence = [np.argmax(word) for word in french_sentence]

    french_sentence = french_tokenizer.sequences_to_texts([french_sentence])[0]
    
    # print("French translation: ", french_sentence)
    
    return french_sentence

def translate_to_hindi(english_sentence):
    english_sentence = english_sentence.lower()
    
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length_hindi)
    
    english_sentence = english_sentence.reshape((-1,max_length_hindi))
    
    hindi_sentence = model_hi.predict(english_sentence)[0]
    
    hindi_sentence = [np.argmax(word) for word in hindi_sentence]

    hindi_sentence = hindi_tokenizer.sequences_to_texts([hindi_sentence])[0]
    
    # print("hindi translation: ", hindi_sentence)
    
    return hindi_sentence

def solve():
    input_text = input_entry.get()
    url = "https://translate.googleapis.com/translate_a/single"
    params1 = {
        'client': 'gtx',
        'sl': 'en',  
        'tl': 'fr',  
        'dt': 't',
        'q': input_text  
    }
    params2 = {
        'client': 'gtx',
        'sl': 'en',  
        'tl': 'hi',  
        'dt': 't',
        'q': input_text  
    }
    
    if len(input_text) == 10:
        try:
            response_fr = requests.get(url, params=params1)
            response_hi = requests.get(url, params=params2)
            response_fr.raise_for_status()
            response_hi.raise_for_status()
            translation_fr = response_fr.json()[0][0][0]
            translation_hi = response_hi.json()[0][0][0]
            translated_sent = f"French: {translation_fr} \n Hindi : {translation_hi}"
        except Exception as e:
            translated_sent = "Error"
    else:
        translated_sent = "Please enter 10 letter english words only"
    
    result_label.config(text=translated_sent)
    

root = tk.Tk()
root.title("Language Translator")
root.geometry("500x300")

font = ('Helvetica', 14)

input_entry = tk.Entry(root, width=80, font=font)
input_entry.pack(pady=10)

instruction_label = tk.Label(root, text="Enter sentence for dual translation", wraplength=400, font=font)
instruction_label.pack(pady=10)

translate_button = tk.Button(root, text="Translate", command=solve, font=font)
translate_button.pack(pady=10)

result_label = tk.Label(root, text="", wraplength=400, font=font)
result_label.pack(pady=20)

root.mainloop()