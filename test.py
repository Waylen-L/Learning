from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
"""
To load dataset from directories, use
tensorflow.keras.utils.image_dataset_from_directory

For more details: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing
"""
import numpy as np
import glob
import matplotlib.pyplot as plt

#test and model dir
test_dir = './data/test'
model_dir = './model'

#num samples
test_samples = glob.glob(test_dir + '/*/*.jpg')
print(test_samples)

#img size
img_w, img_h = 224,224

###model###
model = load_model(model_dir + '/model_07.hdf5')

for i in enumerate(test_samples):
    print(i[1])
    x = load_img(i[1],color_mode='rgb',target_size=(img_h,img_w))
    x = img_to_array(x)/255
    x = x[None,...]
    predict = model.predict(x,batch_size=1,verbose=1)
    print(predict)
