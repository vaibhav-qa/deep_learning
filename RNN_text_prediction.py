# RNN to predict based on paradise lost
import numpy as np
from tensorflow import keras
# covert to lowercase
filename = "D:\\book\\paradise_lost.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers and reverse
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = keras.utils.to_categorical(dataY)
# define the LSTM model
model = keras.Sequential([
    keras.layers.LSTM((256), return_sequences=False,  input_shape=(X.shape[1], X.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(y.shape[1], activation='softmax')
    ])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy')

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=30, batch_size=128, callbacks=callbacks_list)

# pick a random seed
print("\nDone.")