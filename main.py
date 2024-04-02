import random
import json
import torch
import pyttsx3
import speech_recognition as sr
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents data
with open('Resources/data/data.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model
MODEL_PATH = "Resources/model/data.pth"
data = torch.load(MODEL_PATH)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize and load model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "o(*￣︶￣*)o"

def set_realistic_voice(engine):
    voices = engine.getProperty('voices')
    # Selecting a human-like voice, usually the first one
    engine.setProperty('voice', voices[0].id)
    # Adjusting speech rate (words per minute)
    engine.setProperty('rate', 150)  # You can adjust this value for your preference
    # Adjusting volume
    engine.setProperty('volume', 1.0)  # Set the volume to maximum (1.0)

def speak(text):
    engine = pyttsx3.init()
    # Setting up realistic voice options
    set_realistic_voice(engine)
    # Adding the text to the engine's queue
    engine.say(text)
    # Playing the audio in real-time
    engine.runAndWait()

def recognition_voice():    
    print("Let's chat! (say 'stop' to exit)")
    engine = pyttsx3.init()
    set_realistic_voice(engine)  # Setting up realistic voice options
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    while True:
        with microphone as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            sentence = recognizer.recognize_google(audio)
            print("You:", sentence)
        except sr.UnknownValueError:
            print("(⊙_⊙)？: Sorry, I didn't catch that.")
            continue
        except sr.RequestError:
            print("(⊙_⊙)？: Sorry, my speech recognition service is down.")
            continue
        if sentence.lower() == "stop":
            speak("bye master see you later....!")
            print("(づ￣ 3￣)づ.....!")
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents:
                if tag == intent["instruction"]:
                    response = intent['output']
                    print(f"{bot_name}: {response}")
                    speak(response)
        else:
            print(f"(⊙_⊙)？: I do not understand...")

if __name__ == '__main__':
    # Start the conversation
    recognition_voice()
