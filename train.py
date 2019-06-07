import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import cv2
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from model import residual_unet
np.random.seed(42)

data_location = './data/DRIVE'
training_images_loc = data_location + '/training/images/'
training_label_loc = data_location + '/training/label/'
testing_images_loc = data_location + '/test/images/'
testing_label_loc = data_location + '/test/label2/'

train_files = os.listdir(training_images_loc)
train_data = []
train_label = []


for i in train_files:
    train_data.append(cv2.resize((cv2.imread(training_images_loc + i)), (512, 512)))
    # Change '_manual1.tiff' to the label name
    temp = cv2.resize(cv2.imread(training_label_loc + i.split('_')[0] + '_manual1.tiff'),
                      (512, 512))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)
train_data = np.array(train_data)
train_label = np.array(train_label)

test_files = os.listdir(testing_images_loc)
test_data = []
test_label = []

for i in test_files:
    test_data.append(cv2.resize((cv2.imread(testing_images_loc + i)), (512, 512)))
    # Change '_manual1.tiff' to the label name
    temp = cv2.resize(cv2.imread(testing_label_loc + i.split('_')[0] + '_manual1.tiff'),
                      (512, 512))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_label.append(temp)
test_data = np.array(test_data)
test_label = np.array(test_label)

x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 512, 512, 3))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), 512, 512, 1))  # adapt this if using `channels_first` im

x_test = test_data.astype('float32') / 255.
y_test = test_label.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), 512, 512, 3))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), 512, 512, 1))  # adapt this if using `channels_first` im

TensorBoard(log_dir='./autoencoder', histogram_freq=0,
            write_graph=True, write_images=True)


model = residual_unet((512, 512, 3))

if os.path.isfile('resnet_retinal_vessel.hdf5'): model.load_weights('resnet3_retinal_vessel.hdf5')

model_checkpoint = ModelCheckpoint('resnet_retinal_vessel.hdf5', monitor='val_acc', verbose=1, save_best_only=True)

plot_model(model, to_file='unet_resnet.png', show_shapes=False, show_layer_names=False)

model.fit(x_train, y_train,
                epochs=300,
                batch_size=2,
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks= [TensorBoard(log_dir='./autoencoder'), model_checkpoint])

