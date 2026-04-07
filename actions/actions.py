from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os


class ActionAnalyzeSentiment(Action):

    def name(self) -> Text:
        return "action_analyze_sentiment"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get('text')
        print(f"Pesan User: {user_message}")

        if not os.path.exists('sentiment_model.h5') or not os.path.exists('tokenizer.pickle'):
            dispatcher.utter_message(
                text="Maaf, sistem analisis emosi sedang tidak aktif (File model tidak ditemukan).")
            return []

        try:
            model = tf.keras.models.load_model('sentiment_model.h5')
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
        except Exception as e:
            print(f"Error loading model: {e}")
            dispatcher.utter_message(text="Terjadi kesalahan sistem saat memuat model AI.")
            return []

        seq = tokenizer.texts_to_sequences([user_message])
        padded = pad_sequences(seq, maxlen=20, padding='post')
        prediction = model.predict(padded)

        class_idx = np.argmax(prediction)

        # labels = [0, 0, 1, 1, 2, 2] -> 0: Sedih, 1: Senang, 2: Cemas
        emotions = {0: "Sedih/Putus Asa", 1: "Senang/Bahagia", 2: "Cemas/Takut"}
        detected_emotion = emotions.get(class_idx, "Netral")

        print(f"Terdeteksi Emosi: {detected_emotion}")

        response_text = ""
        if class_idx == 0:  # Sedih
            response_text = f"Saya merasakan kesedihan dalam kata-katamu ({detected_emotion}). Tarik napas pelan-pelan. Saya di sini untuk mendengarkan. Apa yang membuatmu merasa berat hari ini?"
        elif class_idx == 1:  # Senang
            response_text = f"Wah, energi positifmu terasa sampai sini! ({detected_emotion}). Senang mendengarnya. Apa yang bikin harimu menyenangkan?"
        elif class_idx == 2:  # Cemas
            response_text = f"Saya menangkap rasa kekhawatiran ({detected_emotion}). Tidak apa-apa merasa cemas, itu manusiawi. Coba ceritakan pelan-pelan, apa yang sedang kamu takutkan?"
        else:
            response_text = "Saya mendengarkan cerita Anda. Silakan lanjutkan..."

        dispatcher.utter_message(text=response_text)

        return []