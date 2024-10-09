import os
import random
import glob
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping


classes = ['spring','summer','fall','winter']
nb_classes = len(classes)

#train val dir
train_dir = './data/train'
val_dir = './data/val'
#model saved dir
model_dir = './model'

#num samples
train_samples = glob.glob(train_dir + '/*/*.jpg')
val_samples = glob.glob(val_dir + '/*/*.jpg')
train_samples = len(train_samples)
val_samples = len(val_samples)
print(train_samples)
print(val_samples)

#img size
img_w, img_h = 224,224

#keras image data generator
train_datagen = ImageDataGenerator(rescale = 1.0/255, zoom_range=0.2,horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_h,img_w),
    color_mode = 'rgb',
    classes = classes,
    class_mode = 'categorical',
    batch_size = 5
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_h,img_w),
    color_mode = 'rgb',
    classes = classes,
    class_mode = 'categorical',
    batch_size = 5
)

#TO save model
checkpoint = ModelCheckpoint(
    filepath = os.path.join(
        model_dir,
        'model_{epoch:02d}.hdf5'
    ),
    save_best_only=True
)

#TO early stopping
early_stopping = EarlyStopping(monitor='val_loss',patience=3,verbose=0,mode='auto')

### model ###
#ResNet50
input_tensor = Input(shape=(img_w,img_h,3))
## train skratch ==>> weights=None ##
ResNet50 = ResNet50(include_top=False, weights=None ,input_tensor=input_tensor)
#def softmax
top_model = Sequential()
top_model.add(Flatten(input_shape=ResNet50.output_shape[1:]))
top_model.add(Dense(nb_classes, activation='softmax'))
model = Model(input=ResNet50.input, output=top_model(ResNet50.output))

#hyper param
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

#train
epoch = 1
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples,
    epochs=epoch,
    validation_data=val_generator,
    callbacks=[checkpoint,early_stopping]
)
print(history.history)

#plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = epoch

plt.figure()
plt.plot(range(1,epochs+1),acc,'b',label = 'traning accracy')
plt.plot(range(1,epochs+1),val_acc,'r',label='validation accracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('result_acc')


plt.figure()
plt.plot(range(1,epochs+1), loss, 'bo', label='Training loss')
plt.plot(range(1,epochs+1), val_loss, 'ro', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('result_loss')

plt.show()
