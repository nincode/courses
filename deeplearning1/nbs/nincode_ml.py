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
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras._impl.keras.preprocessing.image import Iterator
from tensorflow.python.keras.layers import Dense, Lambda, Dropout, BatchNormalization, ZeroPadding2D, Convolution2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras._impl.keras import callbacks as cbks

# Example of precaching
# cached_valid = m1.precache_batch(batches_valid)
# cached_train = m1.precache_batch(batches_train)

K.backend.set_image_data_format('channels_last')
default_batch_size = 32

class NincodeUtils():
    """ Common load/save, dealing with paths on Linux, Windows etc.  """
    def __init__(self):
        if os.path.exists("c:/"):
            self.rootdir='d:/temp/ml/nincode/'
        else:
            self.rootdir = "/home/relja/data/nincode/"
        os.makedirs(self.get_filename('.'), exist_ok=True)
        print("NincodeUtils configured to {0}".format(self.rootdir))

    def get_filename(self, fname):
        return os.path.join(self.rootdir, fname)

    def save_array(self, fname, arr): 
        fname = self.get_filename(fname)
        print("Saving",fname)
        c=bcolz.carray(arr, rootdir=fname, mode='w')
        c.flush()

    def load_array(self, fname): 
        fname = self.get_filename(fname)
        print("Loading",fname)
        return bcolz.open(fname)[:]
        
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
                x = pickle._load(file)
            return x
        except IOError:
            return None

NU = NincodeUtils();

class _CachedIterator(Iterator):
    def __init__(self, cached_data, cached_labels, batch_size=default_batch_size):
        self.data = cached_data
        self.labels = cached_labels
        self.samples = len(self.data)
        print("  Data shape",self.data.shape)
        print("  Labels shape",self.labels.shape)
        super(_CachedIterator, self).__init__(self.samples, batch_size, True, 1)
    
    def next(self):
        with self.lock:
              index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = []
        batch_y = []
        for i, j in enumerate(index_array):
            batch_x.append([self.data[j]])
            batch_y.append([self.labels[j]])
        npx = np.concatenate(batch_x)
        npy = np.concatenate(batch_y)
        return np.concatenate(batch_x), npy

class DataBatch():
    """ Abstracts loading batches from disk, handles cached and regular data """
    def __init__(self, path, name, batch_size=default_batch_size, always_use_full_batch=False, ignore_cached=False):
        """ Params:
            path                  - d:\temp\dogscats 
            name                  - train, valid, etc.
            batch_size            - number of data points / images to return in a single next() call
            always_use_full_batch - validation batches should probably always be fully used. for training, shorter epochs may be better.
        """
        self.name = name
        self.batch_size = batch_size
        self.path = path
        self.always_use_full_batch = always_use_full_batch
        self.is_cached = False
        if name == 'valid' and self.always_use_full_batch == False:
            print("You probably want always_use_full_batch=True on validation batches")
        self._load(ignore_cached)

    def _load_noncached(self):
        self.is_cached = False
        self.image_data_generator = image.ImageDataGenerator()
        location = os.path.join(self.path, self.name)
        x = self.image_data_generator.flow_from_directory(directory=location, 
            target_size=(224,224), 
            class_mode='categorical', 
            shuffle=True, 
            batch_size=self.batch_size)
        self.iter = x
        self.file_count=len(self.iter.filenames)

    def _load_cached(self, path_data, path_labels):
        self.is_cached = True
        self.cached_data = NU.load_array(path_data)
        self.cached_labels = NU.load_array(path_labels)
        if (len(self.cached_data) != len(self.cached_labels)):
            raise ValueError("Cached data and labels are of different lengths.")
        self.file_count=len(self.cached_data)
        self.iter = _CachedIterator(self.cached_data, self.cached_labels)

    def _save_cached(self, data, labels):
        NU.save_array(os.path.join(self.path, "cached_"+self.name+"_data"), data)
        NU.save_array(os.path.join(self.path, "cached_"+self.name+"_labels"), labels)

    def _load(self,ignore_cached):
        if ignore_cached == False:
            cached_name_data = os.path.join(self.path, "cached_"+self.name+"_data")
            cached_name_labels = os.path.join(self.path, "cached_"+self.name+"_labels")
            if os.path.exists(cached_name_data) and os.path.exists(cached_name_labels):
                print("Found cached data",cached_name_data)
                self._load_cached(cached_name_data, cached_name_labels)
                return
        self._load_noncached()

    def full_step_count(self):
        return int((self.iter.samples+self.iter.batch_size - 1)/self.batch_size)

    def step_count(self):
        if self.always_use_full_batch:
            return self.full_step_count()
        else:
            return self.full_step_count()/4

class _TrainCallback(cbks.Callback):
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
            self.parent._checkpoint_save(logs['timestamp'], val_loss)

        self.parent._save_state()

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1, 3))
class Model1():
    """ 
    model - Sequential model. Feel free to inspect if needed.
    """

    def __init__(self, name):
        self.name = name
        self.folder_model = name
        self.checkpoint_data = []
        self.model_trainable = None         # Used for running cached data
        self.model_nontrainable = None      # Used for generating cached data
        os.makedirs(NU.get_filename(self.folder_model), exist_ok=True)
        self.file_history = os.path.join(self.folder_model, "history.pkl")
        self.history = NU.load_object(self.file_history) or []
        if len(self.history) > 0:
            print("Loaded history", self.file_history)
        self.callback = _TrainCallback(epoch_history=self.history, filename_history=self.file_history, parent=self)

    def clear(self):
        """ Reset history, delete disk checkpoints """
        self.history.clear()
        self._cleanup_checkpoints()
        self._save_state()


    def test(self, batch_valid):
        if batch_valid.is_cached == True:
            self._go_cached()
            model = self.model_trainable
        else:
            model = self.model
        ret = model.evaluate_generator(batch_valid.iter, batch_valid.step_count())
        print("val_loss:{0:.4f}  val_acc:{1:.4f}".format(ret[0], ret[1]))
        print(ret)

    def compile(self):
        self.model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model compiled")

    def train(self, batch_train, batch_valid, callbacks=[], epochs=1):
        if batch_valid.is_cached == True:
            self._go_cached()
            model = self.model_trainable
        else:
            model = self.model

        callbacks = copy.copy(callbacks)
        callbacks.append(self.callback)
        self._cleanup_checkpoints()
        ret = model.fit_generator(generator=batch_train.iter, steps_per_epoch=batch_train.step_count(), 
                                 validation_data=batch_valid.iter, validation_steps=batch_valid.step_count(), 
                                 callbacks=callbacks,
                                 epochs=epochs)
        return ret

    def print_layers(self, model):
        def proc(k):
            cfg = k.get_config()
            trainable = True
            if type(cfg) == dict:
                name = cfg.get('name', 'None')
                trainable = cfg.get('trainable', False)
                
            if type(k)!=Model and type(k)!=Sequential:
                print("{0:30} - {1:6} - {2:20} - {3:30}".format(name, trainable, str(k.input.get_shape()), str(k.output.get_shape())))
            else:
                for layer in k.layers:
                    proc(layer)
                    
        print("{0:30} - {1:6} - {2:20} - {3:30}".format("Name", "Trainable", "Input", "Output"))
        proc(model)

    def precache_batch(self,  batch):
        """ Process data for caching 
            batch = Batch to be precached
        """ 
        print("Caching", batch.name)
        processed = self._generate_precache_model(batch)
    
        batch._save_cached(data=processed['data'], labels=processed['labels'])       
        return (processed['data'], processed['labels'])

    ##########################################################
    ## Checkpoints
    def _restore_best_checkpoint(self):
        """ Find best checkpoint, and restore to it """
        self._cleanup_checkpoints()
        if len(self.checkpoint_data) == 0:
            print("No checkpoints found to restore")
            return
        print("Restoring {0}, with loss {1:.3f}".format(self.checkpoint_data[0]['filename'], self.checkpoint_data[0]['val_loss']))
        self.model.load_weights(self.checkpoint_data[0]['filename'], by_name=True)

    def _cleanup_checkpoints(self):
        """ Deletes >3 checkpoints, deletes unknown checkpoints, updates the file cache """
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

    def _checkpoint_save(self, timestamp, val_loss):
        fname = os.path.join(self.folder_model,'{0}_{1:.3f}.checkpoint'.format(timestamp,val_loss))
        fname = NU.get_filename(fname)
        print("Saving model checkpoint to", fname)
        self.model.save_weights(fname, overwrite=True)
        self._cleanup_checkpoints()


    ##########################################################
    ## Model creation
    @staticmethod
    def _vgg_preprocess(x):
        # Preprocess images (RGB->BGR, subract means)
        x = x - vgg_mean
        return x[..., ::-1] # reverse axis rgb->bgr

    def create_model(self, class_num=2):
        """Stick preprocessing, VGG16, and a custom top model together.
        Args:
            class_num(int) - number of classes to differentiate
        Returns:
            Sequential model
        """
        base_model = self._create_model_VGG16()
        new_model = Sequential()
        new_model.add(base_model)
        new_model = self._flatten(new_model)
        self._create_model_top(new_model, class_num=class_num)
        self.model = new_model
        self._restore_best_checkpoint()
        self.summary()

    def _create_model_VGG16(self):
        # Create a stock tf VGG16, prepended with an image processor
        vgg = VGG16(input_shape=(224,224,3), include_top=True, weights='imagenet')
        vgg.trainable = False
        layer_block5_conv3 = vgg.get_layer(name='block5_conv3')    # Last 14x14 block

        vgg2 = Model(inputs=vgg.input, outputs=layer_block5_conv3.output)

        top_model = Sequential()
        lbd = Lambda(self._vgg_preprocess,input_shape=(224,224,3),name="image_preprocess")
        lbd.trainable = False
        top_model.add(lbd)
        top_model.add(vgg2)
        top_model.layers[1].trainable=False
        return top_model

    def _add_conv_block(self, model_small, block_id, convolutions=512):
        name="cblock_{0}".format(block_id)
        model_small.add(ZeroPadding2D((2,2), name=name+"_padding"))
        model_small.add(Convolution2D(convolutions, kernel_size=(5, 5), activation='relu', name=name+"_conv"))
        model_small.add(BatchNormalization(name=name+"_batchnorm"))
        if convolutions > 2:
            # Don't add dropout to a single or binary convolutional layer
            model_small.add(Dropout(0.3, name=name+"_dropout"))
        

    def _create_model_top(self, model, class_num=2):
        # Fun bits. Sits on top of VGG16. This is where we play.

        for i in range(16):
            self._add_conv_block(model, block_id=i, convolutions=16)

        self._add_conv_block(model, block_id=1001, convolutions=2)
        
        model.add(GlobalAveragePooling2D(name="top_global_avg"))
#        model.add(Dropout(0.2, name="top_dropout"))
        model.add(Dense(class_num, activation='softmax', name="top_output"))

    def summary(self):
        trainable_count = int(np.sum([K.backend.count_params(p) for p in set(self.model.trainable_weights)]))
        non_trainable_count = int(np.sum([K.backend.count_params(p) for p in set(self.model.non_trainable_weights)]))

        print('Total params        : {:12,}'.format(trainable_count + non_trainable_count))
        print('Non-trainable params: {:12,}'.format(non_trainable_count))
        print('Trainable params    : {:12,}'.format(trainable_count))

    ##########################################################
    ## Precaching. Start with precache_batch
    def _generate_precache_model(self, batches):
        if self.model_nontrainable == None:      
            last_non_trainable = None
            mdl = Sequential()
            for i in range(len(self.model.layers)):
                if self.model.layers[i].trainable == True:
                    break
                last_non_trainable = self.model.layers[i]
                mdl.add(self.model.layers[i])

            if last_non_trainable == None:
                print("All layers trainable. Don't know what to cache. Exiting")
                return None

            print("Cached model - using output of",last_non_trainable.name)
            mdl.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])
            self.model_nontrainable = mdl
            
        cnt = batches.full_step_count()
        batch_data = batches.iter.next()
        data=[]
        label=[]
        for i in range(cnt):
            batch_data = batches.iter.next()
            if (i%10 == 0):
                print("Loaded {0} datapoints\r".format(i*batches.batch_size),end='',flush=True)
            data.append(batch_data[0])
            label.append(batch_data[1])
        imgs = np.concatenate([data[i] for i in range(cnt)])
        labels = np.concatenate([label[i] for i in range(cnt)])
        y = self.model_nontrainable.predict(imgs, verbose=1)
        print(y.shape)
        return {'data':y, 'labels':labels}

    def _go_cached(self):
        """ Generate the model needed to use cached batches (i.e. drop the non-trainable steps) """
        print("Using cached mode (skipping non-trainable steps)")
        first_trainable = None
        if self.model_trainable == None:
            print("Trainable model not found. Building.")
            mdl = Sequential()
            for i in range(len(self.model.layers)):
                if self.model.layers[i].trainable == True:
                    if first_trainable is None:
                        first_trainable = self.model.layers[i]
                        mdl.add(Dropout(0, input_shape=(14,14,512)))
                        mdl.add(self.model.layers[i])
                    else:
                        mdl.add(self.model.layers[i])

            if first_trainable == None:
                print("All layers non-trainable. Don't know what to cut. Exiting")
                return

            self.model_trainable = mdl
            self.model_trainable.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])
            self.model_trainable.summary()

    ##########################################################
    ## Utilities
    def _save_state(self):
        NU.save_object(self.file_history, self.history)

    def _flatten(self, model):
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


 

def save_array(fname, arr): 
    print("Saving",fname)
    c=bcolz.carray(arr, rootdir='.bcolz/'+fname, mode='w')
    c.flush()

def load_array(fname): 
    print("Loading",fname)
    return bcolz.open('.bcolz/'+fname)[:]


