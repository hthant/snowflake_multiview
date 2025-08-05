import xbatcher
import optuna
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from toKeras3 import *

#default variables defined here
numTrials = 60
NUMPERCLASS = 1200
TRAINSPLIT = 0.8
EPOCHS = 7
num_classes = 6 #, 5 or 2
count = 0

# Creating the Model layers
def getML(af, dr, hl, num_classes):
    input_shape1 = layers.Input(shape=(300, 300, 3))
    input_shape2 = layers.Input(shape=(300, 300, 3))
    input_shape3 = layers.Input(shape=(300, 300, 3))

    ceppy = InceptionV3(include_top = False, weights="imagenet")

    ceppy._name = "featureExtractor1"
    base_model1 = ceppy(input_shape1)
    base_model2 = ceppy(input_shape2)
    base_model3 = ceppy(input_shape3)
    base_model1 = layers.GlobalAveragePooling2D()(base_model1)
    base_model2 = layers.GlobalAveragePooling2D()(base_model2)
    base_model3 = layers.GlobalAveragePooling2D()(base_model3)

    base_model = layers.concatenate([base_model1, base_model2, base_model3], axis=1)
    base_model = layers.Flatten()(base_model)

    for h in hl:
        add_model = (layers.Dense(int(h), use_bias=False))(base_model)
        add_model = (layers.BatchNormalization(center=True, scale=False))(add_model)
        add_model = (layers.Activation(af))(add_model)
        add_model = (layers.Dropout(dr))(add_model)

    add_model = (layers.Dense(int(num_classes), activation="softmax"))(add_model)

    add_model = Model(inputs=[input_shape1, input_shape2, input_shape3], outputs=add_model)
    return add_model

# Fully define the model
def getModel(activation_func,learning_rate,dropout_rate,layer,num_classes,optimzer_func):
    model = getML(activation_func, dropout_rate, layer, num_classes)
    if optimzer_func == "rmsprop":
        model.compile(optimizer = RMSprop(learning_rate=learning_rate),
                        loss = 'categorical_crossentropy',
                        metrics = ['acc'])
    elif optimzer_func == "adam":
        model.compile(optimizer = Adam(learning_rate=learning_rate),
                        loss = 'categorical_crossentropy',
                        metrics = ['acc'])
    elif optimzer_func == "sgd":
        model.compile(optimizer = SGD(learning_rate=learning_rate),
                        loss = 'categorical_crossentropy',
                        metrics = ['acc'])
    return model

# Training code
def train(model, train_generator, validation_generator, ind, num_classes, batch_size):
    filepath = 'weights/p1_' + str(ind) + '-ep-{epoch:05d}.keras'
    callbacks = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = model.fit(
            train_generator,
            validation_data = validation_generator,
            epochs = EPOCHS,
            steps_per_epoch= int(((NUMPERCLASS*num_classes)*TRAINSPLIT)/batch_size),
            validation_steps = int(((NUMPERCLASS*num_classes)*(1-TRAINSPLIT))/batch_size),
            verbose = 1,
            callbacks=[callbacks],
            class_weight=class_weight
        )
    return model, history

# import data and compute weights
train_images, train_labels, validation_images, validation_labels = [], [], [], []
labels   = np.array(train_labels)      # 1‑D int array
classes  = np.unique(labels)
weights  = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
class_weight = dict(zip(classes, weights))

# Objective function for Optuna
def objective(trial):
    global count
    count = count + 1
    count = count
    activation = trial.suggest_categorical("activation",["relu","leaky_relu","gelu","swish"])
    optimizer = trial.suggest_categorical("optimizer",["adam","rmsprop","sgd"])
    learning_rate = trial.suggest_float("lr",1e-4,1e-1,log=True)
    batch_size = trial.suggest_int("batch_size",16,64)
    dropout = trial.suggest_float("dropout",0.1,0.5)
    steps_list = [
        [2048],
        [256,16],
        [128,8],
        [512,128,8],
        [256,128,16],
        [256,64,8],
        [128,16,8],
        [512,256,64,16]
    ]
    steps = trial.suggest_categorical("hidden_layers",steps_list)

    #Pre-procceses images for TensorFlow compatability
    def transformTensor(arr):
        arr = tf.cast(arr, dtype=tf.float32)
        arr = tf.math.subtract(arr, tf.math.reduce_mean(arr))
        arr = tf.math.divide(arr, tf.math.reduce_std(arr))
        arr = tf.reshape(arr, [3, 300, 300, batch_size])
        out = tf.zeros([0, 300, 300, 3])
        for i in range(batch_size):
            temp = arr[:, :, :, i]
            temp = tf.stack([temp, temp, temp])
            temp = tf.reshape(temp, [3, 300, 300, 3])
            out = tf.concat([out, temp], axis=0)
        out = tf.reshape(out, [batch_size, 3, 300, 300, 3])
        out1 = tf.reshape(out[:, 0, :, :, :], [batch_size, 300, 300, 3])
        out2 = tf.reshape(out[:, 1, :, :, :], [batch_size, 300, 300, 3])
        out3 = tf.reshape(out[:, 2, :, :, :], [batch_size, 300, 300, 3])
        return out1, out2, out3

    #One-hots labels for TensorFlow compatability
    def transformLabel(l):
        l = tf.cast(tf.math.subtract(l, 1), dtype=tf.uint8)
        l = tf.one_hot(l, num_classes)
        return l
    print("########################################################")
    print("Trial :", count)
    print(" ")
    print("Activation Function :",activation)
    print("Optimizer :",optimizer)
    print("Learning Rate :", learning_rate)
    print("Batch Size :", batch_size)
    print("Dropout Rate :", dropout)
    print("Layer Config :", steps)
    print("--------------------------------------------------------")

    print(" -- Splitting training and validation images")
    train_images_generator = xbatcher.BatchGenerator(train_images,input_dims={'flake_id':batch_size})
    validation_images_generator = xbatcher.BatchGenerator(validation_images,input_dims={'flake_id':batch_size})
    train_labels_generator = xbatcher.BatchGenerator(train_labels,input_dims={'l':batch_size})
    validation_labels_generator = xbatcher.BatchGenerator(validation_labels,input_dims={'l':batch_size})

    train_generator = CustomTFDataset(train_images_generator, train_labels_generator, transform=transformTensor, target_transform=transformLabel)
    validation_generator = CustomTFDataset(validation_images_generator, validation_labels_generator, transform=transformTensor, target_transform=transformLabel)
    print(" -- Splitting complete, now training model")

    model = getModel(activation, learning_rate, dropout, batch_size, steps, num_classes, optimizer)
    model, history = train(model, train_generator, validation_generator, count, num_classes, batch_size)
    print("Training complete! Check the results of the training above.")

    print("--------------------------------------------------------")
    print("Results :")
    print(history.history)

    print("########################################################")

    return max(history.history["val_acc"])

# Trials begin
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=numTrials)

print("\nfin\n")
