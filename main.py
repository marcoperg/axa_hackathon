#!/usr/bin/env python
# coding: utf-8

import speech_recognition as sr
from IPython.display import Audio as AudioPlayer
from dotenv import load_dotenv
import base64
import wave
from pydub import AudioSegment
from gtts import gTTS
load_dotenv()

# Audio Configuration
SAMPLE_RATE = 24000  # Hz (24kHz)
CHANNELS = 1  # Mono
SAMPLE_WIDTH = 2  # Bytes (16 bits)

from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clip_predict(image):
    # Function to read class labels from a text file
    def read_class_labels(filepath):
        with open(filepath, 'r') as file:
            labels = [line.strip() for line in file.readlines()]
        return labels
    
    class_labels_file = "class_labels.txt"  # Replace with the actual path
    class_labels = read_class_labels(class_labels_file)
    
    inputs = processor(text=class_labels, images=image, return_tensors="pt", padding=True)
    
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # ENSEÑAR EL PERUAN
    return class_labels[np.argmax(probs.detach().numpy())]

import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("user_data/marco.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("user_data/paula.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a second sample picture and learn how to recognize it.
sanchez_image = face_recognition.load_image_file("user_data/david.jpg")
sanchez_face_encoding = face_recognition.face_encodings(sanchez_image)[0]

# Load a second sample picture and learn how to recognize it.
macron_image = face_recognition.load_image_file("user_data/estela.jpg")
macron_face_encoding = face_recognition.face_encodings(macron_image)[0]

# Load a second sample picture and learn how to recognize it.
rajoy_image = face_recognition.load_image_file("user_data/jorge.jpg")
rajoy_face_encoding = face_recognition.face_encodings(rajoy_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    sanchez_face_encoding,
    macron_face_encoding,
    rajoy_face_encoding
]
known_face_names = [
    "Marco",
    "Paula",
    "David",
    "Estela",
    "Jorge",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
def process_frame(process_this_frame=True):
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        clip_predct = clip_predict(small_frame)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame#small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(np.array(known_face_encodings), np.array(face_encoding))
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(np.array(known_face_encodings), np.array(face_encoding))
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        return frame, face_names, face_locations, clip_predct
    return frame, [], [], None

while True:
    # Grab a single frame of video
    frame, new_faces_names, new_face_locations, _ = process_frame(process_this_frame)
    if len(new_face_locations) > 0:
        face_locations = new_face_locations
        face_names = new_faces_names
    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    #cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    break

# Release handle to the webcam
#video_capture.release()
#cv2.destroyAllWindows()


def get_near_specialists(specialist_index: int = 0) -> str:
    """
    Use this function to get the url of different specialists near the user. Call it with the appropiate specialist when the user asks to recommend near specialists.
    
    Args:
        index_specialist (int): Number of specialist to return. 0 corresponds to "aparato digestivo" specialist. 
        1 corresponds to "urgencias" specialist. 
    """
    
    import webbrowser
    import time
    time.sleep(1.5)
    
    aparato_digestivo = "https://www.google.es/maps/place/Hospital+San+Rafael+Urgencias+Pediatricas/@40.4527341,-3.6827777,18.81z/data=!4m15!1m8!3m7!1s0xd4228dfde5b1a47:0xea6bdaadcd846541!2sC.+de+Serrano,+199,+Chamart%C3%ADn,+28016+Madrid!3b1!8m2!3d40.4527672!4d-3.6825297!16s%2Fg%2F11c0yqq6sb!3m5!1s0xd4228dfb8eb8cf1:0x8a99b45e3c909c61!8m2!3d40.4530137!4d-3.6822513!16s%2Fg%2F11gg974k41?hl=es&entry=ttu&g_ep=EgoyMDI1MDMyNS4xIKXMDSoASAFQAw%3D%3D"
    urgencias = "https://www.google.es/maps/place/Hospital+Universitario+Nuestra+Se%C3%B1ora+del+Rosario/@40.4315659,-3.681709,18.59z/data=!3m1!5s0xd4228c096294685:0x2aa961482ac1db98!4m15!1m8!3m7!1s0xd42290941204e97:0x4391523b45748d68!2sCalle+del+Pr%C3%ADncipe+de+Vergara,+53,+Salamanca,+28006+Madrid!3b1!8m2!3d40.4320241!4d-3.6798435!16s%2Fg%2F11tbwkwhmn!3m5!1s0xd4228c0e5bdf009:0xf1933265615a8312!8m2!3d40.4320666!4d-3.6799659!16s%2Fg%2F1q5bnb0cr?hl=es&entry=ttu&g_ep=EgoyMDI1MDMyNS4xIKXMDSoASAFQAw%3D%3D"
    
    
    url = None
    if specialist_index == 0:
        url = aparato_digestivo
    if specialist_index == 1:
        url = urgencias 
    if url == None:
        url = urgencias
    
    webbrowser.open(url)

def get_user_info() -> str:
    """
    Use this function to get general information about current user. 
    Returns a string with name, age, address, and other details.
    """
    users = {
        'Marco': {
            "name": "Marco",
            "age": 21,
            "address": "Calle de Serrano, 199, Chamartín, 28016 Madrid",
            "phone": "+34 912 345 678",
            "email": "marco.example@mail.com",
            "blood_type": "A+"
        },
        'Paula': {
            "name": "Paula",
            "age": 28,
            "address": "Calle del Príncipe de Vergara, 53, Salamanca, 28006 Madrid",
            "phone": "+34 987 654 321",
            "email": "sofia.example@mail.com",
            "blood_type": "O-"
        }
    }
    
    _, user_name, _, y_pred = process_frame()

    
    if len(user_name) == 0:
        return f"Emotion right now: {y_pred}"
    user = users[user_name[0]]
    info = f"""
    User Information:
    Name: {user['name']}
    Age: {user['age']}
    Address: {user['address']}
    Phone: {user['phone']}
    Email: {user['email']}
    Blood Type: {user['blood_type']}
    Emotion right now: {y_pred}
    """
    
    return info.strip()


from agno.agent import Agent
from agno.media import Audio
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.tools.pubmed import PubmedTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.models.google import Gemini


def create_agent():
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp", seed=1),
        markdown=True,
        show_tool_calls=True,
        tools=[get_user_info, PubmedTools(), get_near_specialists],
        add_history_to_messages=True,
        description="You are Axa's health insurance chatbot. You should recommend a specialist depending on the symptoms.",
        instructions=["If you receive an audio containing your prompt listen to it and answer",
                    "The person in speaking in spanish, answer in spanish",
                    "Be brief, helpful and friendly", "You should try to diagnose", 
                    "When asked about near specialists, use the tool get_near_specialists with the corresponding specialist_index",
                    "If you need information about the user, access it via get_user_info",
                    "Maybe the user asks about information about himself, ask if needed"],
    )


agent = create_agent() 

"""
r = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something!")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source, timeout=2) 


output_stream = agent.run(
    "Here is your audio",
    audio=[Audio(content=audio.get_wav_data())],
    stream=True,
    read_chat_history=True
)
msg = []
for chunk in output_stream:
    msg.append(chunk.content)

print("done generating")
tts = gTTS(''.join(msg), lang='es')
tts.save('output.mp3')
print("done tts")
audio = AudioSegment.from_mp3("output.mp3")
sped_up = audio.speedup(playback_speed=1.5)  # 1.5x speed
sped_up.export("output.mp3", format="mp3")
print("done speed up")
display(AudioPlayer("output.mp3"))


"""