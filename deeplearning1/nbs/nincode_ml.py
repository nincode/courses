from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
import time
import random
import bcolz
import pickle
import copy
from operator import itemgetter
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from tensorflow import keras as K
# from keras import backend as K
# from keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, Lambda, Dropout, BatchNormalization, ZeroPadding2D, Convolution2D, GlobalAveragePooling2D
# from keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam, Nadam
# from keras.preprocessing import image
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras._impl.keras import callbacks as cbks

# In case we are going to use the TensorFlow backend we need to explicitly set the Theano image ordering
# from keras import backend as K
# K.backend.set_image_dim_ordering('th')
K.backend.set_image_data_format('channels_last')
default_batch_size = 16

class NincodeUtils():
    def __init__(self):
        if os.path.exists("c:/"):
            self.rootdir='d:/temp/ml/nincode/'
        else:
            self.rootdir = "/home/relja/data/nincode/"
        os.makedirs(self.get_filename('.'), exist_ok=True)
        print("NincodeUtils configured to {0}".format(self.rootdir))

    def get_filename(self, fname):
        return os.path.join(self.rootdir, fname)

    def save_array(fname, arr): 
        print("Saving",fname)
        c=bcolz.carray(arr, rootdir='.bcolz/'+fname, mode='w')
        c.flush()

    def load_array(fname): 
        print("Loading",fname)
        return bcolz.open('.bcolz/'+fname)[:]
        
    def save_object(self, fname, obj):
        try:
            fname = self.get_filename(fname)
#            print("Saving object to", fname)
            temp_name = fname+'.tmp'
            backup_name = fname+'.bak'
            with open(temp_name,'wb') as file:
                pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
            if (os.path.exists(backup_name)):
                os.remove(backup_name)
            if (os.path.exists(fname)):
                os.rename(fname, backup_name)
            os.rename(temp_name, fname)
        except IOError:
            return None

    def load_object(self, fname):
        try:
            fname = self.get_filename(fname)
#            print("Loading object from", fname)
            with open(fname,'rb') as file:
                x = pickle.load(file)
            return x
        except IOError:
            return None

NU = NincodeUtils();

class DataBatch():
    """ Abstracts loading batches from disk """
    def __init__(self, path, batch_size=default_batch_size):
        self.load(path,batch_size=batch_size)

    def load(self, path,batch_size=default_batch_size):
        self.image_data_generator = image.ImageDataGenerator()
        x = self.image_data_generator.flow_from_directory(directory=path, target_size=(224,224), class_mode='categorical', shuffle=True, batch_size=batch_size)
        self.iter = x
        self.batch_size=batch_size
        self.file_count=len(self.iter.filenames)
        return x

    def step_count(self):
        return 32
#        return int(self.file_count/self.batch_size + 1)
        
class TrainCallback(cbks.Callback):
    # Internal only.
    def __init__(self, epoch_history, filename_history, parent):
        self.totals = {}
        self.seen = 0
        self.filename_history = filename_history
        self.parent = parent
        self.glogs = epoch_history

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size
#        print(self.totals)

    def on_epoch_end(self, epoch, logs=None):
        logs = copy.deepcopy(logs)
        if 'val_loss' not in logs:
            # This sure looks like a Keras bug. The callback arrived before the end of the epoch.
            return
        logs['timestamp'] = time.strftime("%y%m%d-%H%M%S")
        logs['epoch'] = epoch
        save_epoch = False

        val_loss = logs['val_loss']
        best_val_loss = 1000
        if len(self.glogs) > 0:
            best_val_loss = float(self.glogs[0]['val_loss'])
            if val_loss < best_val_loss:
                save_epoch = True
        else:
            save_epoch = True

        if save_epoch:
            self.glogs.append(logs)
            self.glogs.sort(key=itemgetter('val_loss'), reverse=False)           

            print('')
            print("Best epoch, needs to be saved. Current:{0:.3f}  old_best:{1:.3f}".format(val_loss, best_val_loss))
            self.parent.checkpoint_save(logs['timestamp'], val_loss)

        self.parent._save_state()

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1, 3))
class Model1():
    """ 
    model - Sequential model. Feel free to inspect if needed.
    """
    @staticmethod
    def vgg_preprocess(x):
        # Preprocess images (RGB->BGR, subract means)
        x = x - vgg_mean
        return x[..., ::-1] # reverse axis rgb->bgr

    def __init__(self, name):
        self.name = name
        self.folder_model = name
        self.checkpoint_data = []               
        os.makedirs(NU.get_filename(self.folder_model), exist_ok=True)
        self.file_history = os.path.join(self.folder_model, "history.pkl")
        self.history = NU.load_object(self.file_history) or []
        if len(self.history) > 0:
            print("Loaded history", self.file_history)
        self.callback = TrainCallback(epoch_history=self.history, filename_history=self.file_history, parent=self)

    def clear(self):
        """ Reset history, delete disk checkpoints """
        self.history.clear()
        self.cleanup_checkpoints()
        self._save_state()

    def _save_state(self):
        NU.save_object(self.file_history, self.history)

    def restore_checkpoint(self):
        self.cleanup_checkpoints()
        if len(self.checkpoint_data) == 0:
            print("No checkpoints found to restore")
            return
        print("Restoring {0}, with loss {1:.3f}".format(self.checkpoint_data[0]['filename'], self.checkpoint_data[0]['val_loss']))
        self.model.load_weights(self.checkpoint_data[0]['filename'], by_name=True)
#        self.model.load_weights(self.checkpoint_data[0]['filename'])

    def cleanup_checkpoints(self):
        checkpoints_to_keep = 3
        # Build index
        timestamp_index = {}
        for k in self.history:
            timestamp_index[k['timestamp']] = k

        # Delete checkpoint files that are not tracked
        # also build a list of files on disk (used for sorting later on)
        self.checkpoint_data = []
        file_names = {}
        path = NU.get_filename(self.folder_model)
        for file in os.listdir(path):
            if file.endswith(".checkpoint"):
                fname = os.path.join(path, file)
                key = file.split('_')[0]
                if key in timestamp_index:
                    file_names[key] = fname
                    timestamp_index[key]['filename']=fname
                    self.checkpoint_data.append(timestamp_index[key])
                else:
                    print('Deleting unknown checkpoint', fname)
                    os.remove(fname)
                    print("timestamp_index")
                    for k,v in timestamp_index.items(): print(k,v)
                    print("history")
                    for k in self.history: print(k)
        self.checkpoint_data.sort(key=itemgetter('val_loss'), reverse=False)

        # Find items to delete based on val_loss
        if len(self.checkpoint_data) == 0:
            max_val_loss = 0
        else:
            max_val_loss = self.checkpoint_data[min(checkpoints_to_keep, len(self.checkpoint_data) - 1)]['val_loss']   

        for t in range(checkpoints_to_keep, len(self.checkpoint_data)):
            os.remove(file_names[self.checkpoint_data[t]['timestamp']])
            print('Deleting inferior checkpoint', file_names[self.checkpoint_data[t]['timestamp']])
        del self.checkpoint_data[checkpoints_to_keep:]

    def checkpoint_save(self, timestamp, val_loss):
        fname = os.path.join(self.folder_model,'{0}_{1:.3f}.checkpoint'.format(timestamp,val_loss))
        fname = NU.get_filename(fname)
        print("Saving model checkpoint to", fname)
        self.model.save_weights(fname, overwrite=True)
        self.cleanup_checkpoints()

    def test(self, batch_valid):
        return self.model.evaluate_generator(batch_valid.iter, batch_valid.step_count())

    def flatten(self, model):
        def proc(k, m2, override_trainable = True):
            cfg = k.get_config()
            trainable = True
            if type(cfg) == dict:
                name = cfg.get('name', 'None')
                trainable = cfg.get('trainable', False)
                
            if type(k)!=Model and type(k)!=Sequential:
                actual_trainable =  trainable & override_trainable
#                print("{0:30} - {1} - {2}".format(name, trainable, actual_trainable))
                k.trainable = actual_trainable
                m2.add(k)
            else:
                for layer in k.layers:
                    proc(layer, m2, trainable & override_trainable)
                    
        m2 = Sequential()
        proc(model, m2)
        return m2

    def create_model_VGG16(self):
        # Create a stock tf VGG16, prepended with an image processor
        vgg = VGG16(input_shape=(224,224,3), include_top=True, weights='imagenet')
        vgg.trainable = False
        layer_block5_conv3 = vgg.get_layer(name='block5_conv3')    # Last 14x14 block

        vgg2 = Model(inputs=vgg.input, outputs=layer_block5_conv3.output)

        top_model = Sequential()
        lbd = Lambda(self.vgg_preprocess,input_shape=(224,224,3),name="image_preprocess")
        lbd.trainable = False
        top_model.add(lbd)
        top_model.add(vgg2)
        top_model.layers[1].trainable=False
        return top_model

    def add_conv_block(self, model_small, block_id, convolutions=512):
        name="cblock_{0}".format(block_id)
        model_small.add(ZeroPadding2D((2,2), name=name+"_padding"))
        model_small.add(Convolution2D(512, kernel_size=(5, 5), activation='relu', name=name+"_conv"))
        model_small.add(BatchNormalization(name=name+"_batchnorm"))
        model_small.add(Dropout(0.5, name=name+"_dropout"))
        

    def create_model_top(self, model, class_num=2):
        # Fun bits. Sits on top of VGG16. This is where we play.
        for i in range(4):
            self.add_conv_block(model, block_id=i)
        
        model.add(GlobalAveragePooling2D(name="top_global_average"))
        model.add(Dropout(0.2, name="top_dropout"))
        model.add(Dense(class_num, activation='softmax', name="top_output"))

    def summary(self):
        trainable_count = int(np.sum([K.backend.count_params(p) for p in set(self.model.trainable_weights)]))
        non_trainable_count = int(np.sum([K.backend.count_params(p) for p in set(self.model.non_trainable_weights)]))

        print('Total params        : {:12,}'.format(trainable_count + non_trainable_count))
        print('Non-trainable params: {:12,}'.format(non_trainable_count))
        print('Trainable params    : {:12,}'.format(trainable_count))

    def create_model(self, class_num=2):
        """Stick preprocessing, VGG16, and a custom top model together.
        Args:
            class_num(int) - number of classes to differentiate
        Returns:
            Sequential model
        """
        base_model = self.create_model_VGG16()
        new_model = Sequential()
        new_model.add(base_model)
#        new_model.add(top_model)
        new_model = self.flatten(new_model)
        self.create_model_top(new_model, class_num=class_num)
        self.model = new_model
        self.restore_checkpoint()
        self.summary()

    def compile(self):
        self.model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])
#        self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model compiled")

    def train(self, batch_train, batch_valid, callbacks=[], epochs=1):
        callbacks = copy.copy(callbacks)
        callbacks.append(self.callback)
        self.cleanup_checkpoints()
        return self.model.fit_generator(generator=batch_train.iter, steps_per_epoch=batch_train.step_count(), 
                                 validation_data=batch_valid.iter, validation_steps=batch_valid.step_count(), 
                                 callbacks=callbacks,
                                 epochs=epochs)
 

def save_array(fname, arr): 
    print("Saving",fname)
    c=bcolz.carray(arr, rootdir='.bcolz/'+fname, mode='w')
    c.flush()

def load_array(fname): 
    print("Loading",fname)
    return bcolz.open('.bcolz/'+fname)[:]



class Vgg17():
    """
        The VGG 16 Imagenet model
    """


    def __init__(self):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.create()
        self.get_classes()


    def get_classes(self):
        """
            Downloads the Imagenet classes index file and loads it to self.classes.
            The file is downloaded only if it not already in the cache.
        """
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        """
            Predict the labels of a set of images using the VGG16 model.

            Args:
                imgs (ndarray)    : An array of N images (size: N x width x height x channels).
                details : ??
            
            Returns:
                preds (np.array) : Highest confidence value of the predictions for each image.
                idxs (np.ndarray): Class index of the predictions with the max confidence.
                classes (list)   : Class labels of the predictions with the max confidence.
        """
        # predict probability of each class for each image
        all_preds = self.model.predict(imgs)
        # for each image get the index of the class with max probability
        idxs = np.argmax(all_preds, axis=1)
        # get the values of the highest probability for each image
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        # get the label of the class with the highest probability for each image
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes


    def ConvBlock(self, layers, filters):
        """
            Adds a specified number of ZeroPadding and Covolution layers
            to the model, and a MaxPooling layer at the very end.

            Args:
                layers (int):   The number of zero padded convolution layers
                                to be added to the model.
                filters (int):  The number of convolution filters to be 
                                created for each layer.
        """
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(3, 3, filters, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))


    def FCBlock(self):
        """
            Adds a fully connected layer of 4096 neurons to the model with a
            Dropout of 0.5

            Args:   None
            Returns:   None
        """
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))


    def create(self):
        """
            Creates the VGG16 network achitecture and loads the pretrained weights.

            Args:   None
            Returns:   None
        """
        model = self.model = Sequential()
        preproc = Lambda(vgg_preprocess, input_shape=(224,224, 3))
        preproc.Trainable = False
        model.add(preproc)

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16.h5'
        fname = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))


    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        """
            Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.

            See Keras documentation: https://keras.io/preprocessing/image/
        """
        return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


    def ft(self, num):
        """
            Replace the last layer of the model with a Dense (fully connected) layer of num neurons.
            Will also lock the weights of all layers except the new layer so that we only learn
            weights for the last layer in subsequent training.

            Args:
                num (int) : Number of neurons in the Dense layer
            Returns:
                None
        """
        model = self.model
        model.pop()
        for layer in model.layers: layer.trainable=False
        model.add(Dense(num, activation='softmax'))
        self.compile()

    def finetune(self, batches):
        """
            Modifies the original VGG16 network architecture and updates self.classes for new training data.
            
            Args:
                batches : A keras.preprocessing.image.ImageDataGenerator object.
                          See definition for get_batches().
        """
        self.ft(batches.nb_class)
        classes = list(iter(batches.class_indices)) # get a list of all the class labels
        
        # batches.class_indices is a dict with the class name as key and an index as value
        # eg. {'cats': 0, 'dogs': 1}

        # sort the class labels by index according to batches.class_indices and update model.classes
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes


    def compile(self, lr=0.001):
        """
            Configures the model for training.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])


    def fit_data(self, trn, labels,  val, val_labels,  nb_epoch=1, batch_size=64):
        """
            Trains the model for a fixed number of epochs (iterations on a dataset).
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.fit(trn, labels, nb_epoch=nb_epoch,
                validation_data=(val, val_labels), batch_size=batch_size)


    def fit(self, batches, val_batches, nb_epoch=1):
        """
            Fits the model on data yielded batch-by-batch by a Python generator.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample)


    def test(self, path, batch_size=8):
        """
            Predicts the classes using the trained model on data yielded batch-by-batch.

            Args:
                path (string):  Path to the target directory. It should contain one subdirectory 
                                per class.
                batch_size (int): The number of images to be considered in each batch.
            
            Returns:
                test_batches, numpy array(s) of predictions for the test_batches.
    
        """
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)

