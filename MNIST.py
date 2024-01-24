import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test = np.array(x_test)
x_train = np.array(x_train)
y_test = np.array(y_test)
y_train = np.array(y_train)



x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Convert labels to one-hot encoding for categorical crossentropy
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=10)########

#but if we are using SparseCategoricalCrossentropy() 
#then we dont need to one -hot encode this
n = tf.keras.layers.Normalization(axis=-1)

n.adapt(x_train)
x_train = n(x_train)

n.adapt(x_test)
x_test = n(x_test)  

model = Sequential([      

       #BEFORE, WHEN I WAS "NOT" USING FROM_LOGITS IT WASNT AS ACCURATE because of roundoff error 
    Dense(50, activation='relu', name='layer1'),
    Dense(35, activation='relu', name='layer2'),
    Dense(10, activation='linear', name='layer3')  # Use softmax for multi-class classification
])                          #linear is used as we are using from_logits=True 

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),########
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)

model.fit(x_train, y_train, epochs=10)



ycap_probs = model.predict(x_test[:20])
p=tf.nn.softmax(ycap_probs).numpy() ############
ycap_classes = np.argmax(p, axis=1)

print("Predicted Probabilities:")
print(ycap_probs)
print("\nPredicted Classes:")
print(ycap_classes)
print("\nTrue Classes:")
print(y_test[:20])




