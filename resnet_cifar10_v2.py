import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer

bnEps=2e-5
bnMom=0.9

def residual_module(x, out_channels, strides=[1, 1], name=None):
    
    with tf.name_scope(name, "residual_module"):
        bn1 = tf.keras.layers.BatchNormalization(epsilon=bnEps, momentum=bnMom)(x)
        relu1 = tf.keras.layers.Activation('relu')(bn1)
        conv1x1_1 = tf.keras.layers.Conv2D(filters=out_channels//4 , use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005),  kernel_size=[1, 1],)(relu1)
        bn2 = tf.keras.layers.BatchNormalization(epsilon=bnEps, momentum=bnMom)(conv1x1_1)
        relu2 = tf.keras.layers.Activation('relu')(bn2)
        conv3x3_2 = tf.keras.layers.Conv2D(filters=out_channels//4 , use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005),  strides=strides, kernel_size=[3, 3], padding='SAME')(relu2)
        bn3 = tf.keras.layers.BatchNormalization(epsilon=bnEps, momentum=bnMom)(conv3x3_2)
        relu3 = tf.keras.layers.Activation('relu')(bn3)
        conv1x1_3 = tf.keras.layers.Conv2D(filters=out_channels , use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005), kernel_size=[1, 1])(relu3)

        shortcut = x
        if x.get_shape().as_list()[-1] != out_channels or strides != [1, 1]:
            shortcut = tf.keras.layers.Conv2D(filters=out_channels, kernel_regularizer=tf.keras.regularizers.l2(0.0005), kernel_size=[1, 1], use_bias=False, strides=strides)(relu1)
        
        residual_module_out = tf.keras.layers.add([shortcut, conv1x1_3])
        
        return residual_module_out
        return residual_module_out


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    num_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, num_batches):
        X_batch = X[batch_idx]
        y_batch = y[batch_idx].squeeze()
        yield X_batch, y_batch


NUM_EPOCHS = 100
INIT_LR = 0.1
def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # return the new learning rate
    return alpha

X = tf.keras.Input(shape=(32, 32, 3), name="X")

with tf.name_scope('conv1'):
    bn1 = tf.keras.layers.BatchNormalization(epsilon=bnEps, momentum=bnMom)(X)
    conv1 = tf.keras.layers.Conv2D(filters=64, padding="same", name="conv1", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005), kernel_size=[3, 3])(bn1)
    print(conv1.get_shape())
with tf.name_scope('conv2_x'):
    out = conv1
    # out = residual_module(out, 64, strides=[2, 2], name="conv2_0")
    for i in range(0, 9):
        out = residual_module(out, 64, name="conv2_{}".format(i))
    # out = residual_module(out, 64, strides=[2, 2], name="conv2_9")
    print(out.get_shape())
with tf.name_scope("conv3_x"):
    out = residual_module(out, 128, strides=[2, 2], name="conv3_0")
    for i in range(1, 9):
        out = residual_module(out, 128, name="conv3_{}".format(i))
    # out = residual_module(out, 128, strides=[2, 2], name="conv3_9")
    print(out.get_shape())
with tf.name_scope("conv4_x"):
    out = residual_module(out, 256, strides=[2, 2], name="conv4_0")
    for i in range(1, 9):
        out = residual_module(out, 256, name="conv4_{}".format(i))
    # out = residual_module(out, 256, strides=[2, 2], name="conv4_9")
    print(out.get_shape())

out = tf.keras.layers.Activation('relu')(out)
out = tf.keras.layers.BatchNormalization(epsilon=bnEps, momentum=bnMom)(out)
avg_pool = tf.keras.layers.AveragePooling2D(pool_size=[8, 8])(out)
flatten = tf.keras.layers.Flatten()(avg_pool)
# dropout = tf.keras.layers.Dropout(0.5)(flatten)
predictions = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(flatten)
# print(predictions)
# predictions = tf.keras.layers.Softmax()(logits)


model = tf.keras.Model(inputs=X, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, amsgrad=True), loss='categorical_crossentropy', metrics=['accuracy'])

((trainX, trainY), (testX, testY)) = tf.keras.datasets.cifar10.load_data()
# trainX, testX = trainX.astype('float'), testX.astype('float')
# testX, testY = testX.astype("int"), testY.astype("int")
# mean = np.mean(trainX, axis=0)
# trainX -= mean
# testX -= mean

# ((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1, 
                         horizontal_flip=True,
                         fill_mode="nearest")

batch_size = 128

model.fit_generator(aug.flow(trainX, trainY, batch_size=128), 
    validation_data=(testX, testY), 
    # validation_steps=len(testX)//batch_size,
    steps_per_epoch=len(trainX)//batch_size, epochs=NUM_EPOCHS, verbose=1 , callbacks=[tf.keras.callbacks.LearningRateScheduler(poly_decay)])
model.save('cifar10_resnet.hdf5')