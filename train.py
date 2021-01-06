from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD

def load_dataset():
	(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)

	return train_images, train_labels, test_images, test_labels

def normalize(train, test):
	train_normalized = train.astype('float32')/255.0
	test_normalized = test.astype('float32')/255.0

	return train_normalized, test_normalized

def create_model():
	model = Sequential([
		Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
		Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
		MaxPooling2D((2, 2)),
		Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
		Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
		MaxPooling2D((2, 2)),
		Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
		Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
		MaxPooling2D((2, 2)),
		Flatten(),
		Dense(128, activation='relu', kernel_initializer='he_uniform'),
		Dense(10, activation='softmax')
	])

	optimizer = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	
	return model

def train():
	train_images, train_labels, test_images,test_labels = load_dataset()
	train_images, test_images = normalize(train_images, test_images)

	model = create_model()
	model.fit(train_images, train_labels, epochs=50, batch_size=500, verbose=0)
	model.save('models/current-model.h5')

	print('Training complete.')

def test():
	train_images, train_labels, test_images,test_labels = load_dataset()
	train_images, test_images = normalize(train_images, test_images)

	model = load_model('models/current.h5')
	metrics = model.evaluate(test_images, test_labels, verbose=0, return_dict=True)
	print('Accuracy: {accuracy}\nLoss: {loss}'.format(**metrics))

train()