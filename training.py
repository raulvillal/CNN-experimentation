import red_neuronal as rn
import numpy as np
import cv2
import os
from zipfile import ZipFile
import urllib
import urllib.request

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

FOLDER = 'fashion_mnist_images'

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'

FILE = 'fashion_mnist_images.zip'

FOLDER = 'fashion_mnist_images'

#Descarga de data set MNIST
''' 
if not os.path.isfile(FILE):
    print(f'Donwloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)



print('Unzipping images...')
with ZipFile(FILE, 'r') as zip_images:
    zip_images.extractall(FOLDER)

print('Done')
'''

def load_mnist_dataset(dataset, path):

    labels = os.listdir('fashion_mnist_images/train')

    X = []
    y = []

    for label in labels:

        for file in os.listdir(os.path.join(path, dataset, label)):

            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):

    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test


# Cargar los datos
print("Cargando datos de Fashion MNIST...")
X, y, X_test, y_test = create_data_mnist(FOLDER)

# Normalizar los datos
print("Normalizando datos...")
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

# Aplanar las imágenes
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(f"Forma de X: {X.shape}")
print(f"Forma de X_test: {X_test.shape}")

# Barajar los datos de entrenamiento
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Crear el modelo
print("\nConstruyendo arquitectura de la red neuronal...")
model = rn.Model()

# Agregar capas a la red
model.add(rn.Layer_Dense(X.shape[1], 128))
model.add(rn.Activation_ReLU())
model.add(rn.Layer_Dense(128, 128))
model.add(rn.Activation_ReLU())
model.add(rn.Layer_Dense(128, 10))
model.add(rn.Activation_Softmax())

# Configurar la pérdida, optimizador y métrica de precisión
model.set(
    loss=rn.Loss_CategorialCrossentropy(),
    optimizer=rn.Optimizer_Adam(decay=1e-5),
    accuracy=rn.Accuracy_Categorical()
)

# Finalizar el modelo
model.finalize()

# Entrenar el modelo
print("\nEntrenando el modelo...")
model.train(
    X, 
    y, 
    validation_data=(X_test, y_test), 
    epochs=10, 
    batch_size=128, 
    print_every=100
)

model.evaluate(X_test, y_test)

# Guardar el modelo completo
print("\nGuardando el modelo...")
model.save('fashion_mnist_model.model')
print("Modelo guardado en 'fashion_mnist_model.model'")

# Guardar los parámetros (pesos y sesgos)
print("Guardando los parámetros del modelo...")
model.save_parameters('fashion_mnist_model_params.parms')
print("Parámetros guardados en 'fashion_mnist_model_params.parms'")

print("\n¡Entrenamiento completado exitosamente!")
