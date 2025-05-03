from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# Normalize the data and reshape for CNN
train_x = train_x.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_x = test_x.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

# Train the model
model.fit(train_x, train_y, epochs=5, validation_split=0.2)

# Evaluate the model
loss, acc = model.evaluate(test_x, test_y)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# Label map for predictions
label_names = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']

# Predict and visualize results
def predict(id_):
    prediction = model.predict(test_x[id_:id_+1])
    predicted_label = label_names[np.argmax(prediction)]
    print("Predicted:", predicted_label)
    plt.imshow(test_x[id_].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.show()

# Test prediction
for i in range(10, 20):
    predict(i)
