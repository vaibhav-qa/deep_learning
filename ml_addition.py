import tensorflow as tf
from tensorflow import keras
import numpy as np
import data_creation as dc 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(20, activation=tf.nn.relu),
	keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(1)
])
#optimizer = tf.train.AdamOptimizer(.001)
model.summary()
model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae'])

#model.compile(optimizer=optimizer, 
#              loss='mse',
#              metrics=['mae'])

model.fit(dc.train_data, dc.train_targets, epochs=10, batch_size=1)

test_loss, test_acc = model.evaluate(dc.test_data, dc.test_targets)
print('Test accuracy:', test_acc)
a= np.array([[2000,3000],[10,-5]])
print(model.predict(a))