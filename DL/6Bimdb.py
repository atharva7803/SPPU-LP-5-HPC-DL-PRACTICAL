
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences

# Load IMDB dataset
vocab_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to the same length
maxlen = 200
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Optional: Define a callback (like EarlyStopping or ModelCheckpoint)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

# Train the model with smaller batch size
results = model.fit(
    X_train, y_train,
    epochs=2,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[callback]
)

# Evaluate the model
score = model.evaluate(X_test, y_test, batch_size=500)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Mean Validation Accuracy:", np.mean(results.history['val_accuracy']))

# Print a sample review and its label
print("Sample review (encoded):", X_train[0])
print("Sample label:", y_train[0])

# Vocabulary lookup
vocab = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in vocab.items()])
decoded_review = " ".join([reverse_index.get(i - 3, "#") for i in X_train[0]])
print("Decoded review:", decoded_review)

# Plot training accuracy
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()