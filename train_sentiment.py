import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D
import pickle

sentences = [
    # --- LABEL 0: SEDIH ---
    "Saya merasa sangat sedih dan putus asa",
    "Hidup ini berat sekali",
    "Saya ingin menangis seharian",
    "Hati saya hancur berkeping-keping",
    "Saya kecewa dengan diri sendiri",
    "Rasanya tidak ada harapan lagi",
    "Hari ini sangat buruk bagi saya",
    "Saya lelah dengan semua masalah ini",

    # --- LABEL 1: SENANG ---
    "Saya bahagia hari ini",
    "Luar biasa senangnya",
    "Akhirnya saya lulus ujian",
    "Hati saya berbunga-bunga",
    "Saya sangat bersyukur atas hidup ini",
    "Hari ini adalah hari terbaik",
    "Semangat saya sedang membara",
    "Kabar gembira datang hari ini",

    # --- LABEL 2: CEMAS ---
    "Saya takut akan masa depan",
    "Cemas sekali rasanya",
    "Jantung saya berdebar kencang karena takut",
    "Saya khawatir tidak bisa menyelesaikannya",
    "Pikiran saya penuh dengan ketakutan",
    "Saya panik dan tidak tahu harus berbuat apa",
    "Saya gugup menghadapi besok",
    "Rasanya tidak tenang sama sekali"
]

labels = [0] * 8 + [1] * 8 + [2] * 8

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=20, padding='post')

model = Sequential([
    Embedding(1000, 16, input_length=20),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Sedang melatih ulang otak bot...")
model.fit(padded, np.array(labels), epochs=100)

model.save('sentiment_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model berhasil di-upgrade!")