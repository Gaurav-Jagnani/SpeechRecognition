# Dataset downloaded from
# https://www.kaggle.com/divyanshu99/spoken-digit-dataset

import tensorflow as tf
import numpy as np
import glob
import os
import librosa
from sklearn.model_selection import train_test_split


RECORDINGS_DIR = "Recordings"


def define_model(input_shape, n_classes):
	inp = tf.keras.layers.Input(input_shape)
	x = tf.keras.layers.LSTM(256, return_sequences=True)(inp)
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	out = tf.keras.layers.Dense(n_classes)(x)
	model = tf.keras.Model(inp, out)

	return model


def load_data():
	max_length = 80
	wavFiles = glob.glob(os.path.join(RECORDINGS_DIR, "*.wav"))
	X = list()
	Y = list()
	for wav in wavFiles:
		wave, sr = librosa.load(wav, mono=True)
		mfcc = librosa.feature.mfcc(wave, sr)
		mfcc = np.pad(
			mfcc,
			((0, 0), (0, max_length - len(mfcc[0]))),
			mode='constant',
			constant_values=0
		)

		X.append(mfcc)
		Y.append(
			wav.split(os.sep)[-1].split("_")[0]
		)

	X = np.array(X)
	Y = np.array(Y)
	Y = Y.astype(np.int32)

	return train_test_split(X, Y, test_size=0.2)


# Training
X_train, X_test, Y_train, Y_test = load_data()

model = define_model(X_train[0].shape, 10)

loss = tf.keras.losses.SparseCategoricalCrossentropy(
										from_logits=True)
accuracy = tf.metrics.Accuracy()
optimizer = tf.keras.optimizers.Adam(0.001)
step = tf.Variable(1, name="global_step")

@tf.function
def train_step(inputs, labels):
	with tf.GradientTape() as tape:
		logits = model(inputs)
		loss_value = loss(labels, logits)

	gradients = tape.gradient(loss_value,
							model.trainable_variables)
	optimizer.apply_gradients(
		zip(gradients, model.trainable_variables))
	step.assign_add(1)
	accuracy_value = accuracy(labels, tf.argmax(logits, axis=-1))

	return loss_value, accuracy_value

ckpt = tf.train.Checkpoint(
	model=model,
	optimizer=optimizer,
	step=step)
manager = tf.train.CheckpointManager(ckpt,
	".checkpoints",	max_to_keep=2)

epochs = 10
batch_size = 32
num_batches = X_train.shape[0] // batch_size

for epoch in range(epochs):
	for batch in range(num_batches):
		start_idx = batch * batch_size
		end_idx = (batch + 1) * batch_size
		X_batch = X_train[start_idx:end_idx]
		Y_batch = Y_train[start_idx:end_idx]

		loss_value, accuracy_value = train_step(
			X_batch, Y_batch)

		if (batch % 10 == 0):
			print(loss_value.numpy(), accuracy_value.numpy())

	print("Epoch: ", epoch + 1)

	save_path = manager.save()
	print("Checkpoint saved at: ", save_path)

	# Log validation accuracy and loss after epoch
	start_idx = 0
	end_idx = 128
	X_batch = X_test[start_idx:end_idx]
	loss_value, accuracy_value = train_step(
			X_test[start_idx:end_idx],
			Y_test[start_idx:end_idx])

	print("Validation loss: {0} \tValidation accuracy: {1}"
		.format(loss_value.numpy(), accuracy_value.numpy()))
