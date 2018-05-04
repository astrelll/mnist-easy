# coding=utf-8
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from keras import backend as b, optimizers


def save(hist, val, name):
    v = hist[val]
    os.system("mkdir -p " + val)
    f = open(val + "/" + name, "w")
    for i in range(0, len(v)):
        f.write(str(i))
        f.write(",")
    f.write("\n")
    for a in v:
        f.write(str(a))
        f.write(",")
    f.write("\n")
    f.close()
    return 0


def go_gpu(GPU):
    num_cores = 2

    if GPU:
        num_GPU = 1
        num_CPU = 1
    else:
        num_CPU = 1
        num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True, \
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})
    session = tf.Session(config=config)
    b.set_session(session)


numpy.random.seed(42)
go_gpu(True)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование размерности изображений
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Нормализация данных
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Преобразуем метки в категории
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()

rand_init = RandomNormal(mean=0.0, stddev=1, seed=None)

"""
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

"""
print("Train data: " + str(len(X_train)))
print("Test data: " + str(len(X_test)))

"""_________Input data_________"""

Layers = [784, 350, 100, 10]
# Regularization
lmbda = 0.0005
# Optimization
opt = 'RMSprop'  # Adagrad RMSprop Adam SGD
opt_param = {'lr': 0.001,
             'eps':'def'}
# Barch size
batch_size = 100
# Dropout
dropout = False
"""____________________________"""

opt_str = str(opt_param)
name = str(Layers) + ' ' + opt + opt_str + ' lmbda=' + str(lmbda) + " init2 batch=" + str(batch_size)

nin = Layers[0]
model.add(Dense(nin, input_dim=784, activation="sigmoid"))

for n in range(1, len(Layers)-1):
    stddev = 1 / numpy.sqrt(nin)
    nin = Layers[n]
    model.add(Dense(nin, activation="sigmoid"
                    , kernel_initializer=RandomNormal(mean=0.0, stddev=stddev, seed=None)
                    , bias_initializer=rand_init
                    , kernel_regularizer=tf.keras.regularizers.l1(lmbda)
                    # , kernel_regularizer=tf.keras.regularizers.l2(lmbda)
                    ))
    if (dropout):
        model.add(Dropout(0.5))
model.add(Dense(Layers[-1], activation="sigmoid"))

# Компилируем модель
optimizer = 0
if (opt == 'SGD'):
    optimizer = optimizers.SGD(lr=opt_param['lr'])

elif (opt == 'Adagrad'):
    optimizer = optimizers.Adagrad(lr=opt_param['lr'])

elif (opt == 'Adam'):
    optimizer = optimizers.Adam(lr=opt_param['lr'])

elif (opt == 'RMSprop'):
    optimizer = optimizers.RMSprop(lr=opt_param['lr'])

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# model.compile(loss="mean_squared_error", optimizer="SGD", metrics=["accuracy"])

print(model.summary())
print("Filename: " + name)

# Обучаем сеть
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=30, validation_split=0.2, verbose=2)

# Оцениваем качество обучения сети на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))

save(history.history, 'acc', name + '.csv')
save(history.history, 'val_acc', name + '.csv')
save(history.history, 'loss', name + '.csv')
save(history.history, 'val_loss', name + '.csv')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
