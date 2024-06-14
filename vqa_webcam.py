import cv2
import os
import torch
from flask import Flask, render_template
from transformers import BlipProcessor, BlipForQuestionAnswering
from gtts import gTTS
import pygame
import tempfile
import speech_recognition as sr

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture image")
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
    cv2.imwrite(img_path, frame)
    cap.release()
    return img_path

def get_question():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please ask a question about the captured image:")
        audio = recognizer.listen(source)
    try:
        question = recognizer.recognize_google(audio)
        print(f"You asked: {question}")
        return question
    except sr.UnknownValueError:
        print("Sorry, I could not understand your question.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def generate_answer(image_path, question):
    image = cv2.imread(image_path)
    inputs = processor(text=question, images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

def speak(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        audio_file = tmpfile.name
    tts.save(audio_file)

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pygame.mixer.quit()
        os.remove(audio_file)

@app.route('/')
def index():
    # Capture image
    try:
        img_path = capture_image()
    except Exception as e:
        return f"Error: {e}"
    
    # Get question from user through voice
    question = get_question()
    if not question:
        return "Sorry, I could not understand your question."
    
    # Generate answer
    answer = generate_answer(img_path, question)
    
    # Convert answer to audio and play
    speak(answer)
    
    # Render template with question and answer
    return render_template('index.html', question=question, answer=answer, image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
