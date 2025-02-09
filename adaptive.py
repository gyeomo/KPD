import warnings
warnings.filterwarnings(action='ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

#tf.disable_v2_behavior()

import tensorflow.keras as keras
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from classification_models.models.resnet import ResNet34, preprocess_input
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras import regularizers
import numpy as np
import wide_residual_network as wrn
import random

from tensorflow.compat.v1.keras.datasets import cifar10, cifar100
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
import time
start=time.time()
tf.set_random_seed(777)
random.seed(777)
np.random.seed(777)

import argparse
parser = argparse.ArgumentParser(description='KPD with Tensorflow')
parser.add_argument('--model',default="cifar10_13",type=str,
                        help='check')
parser.add_argument('--eps', default=16,type=int,
                        help='check number')
args = parser.parse_args()
eps = args.eps 

class CifarModel:
    def __init__(self, path, classes):
        self.num_classes = classes
        self.model_flag = path.split("_")[1]
        self.weight_decay = 0.0005
        if self.model_flag == "re":
            self.model = self.build_model_res()
        elif self.model_flag == "wide":
            self.model = wrn.create_wide_residual_network((32, 32, 3), nb_classes=self.num_classes, N=4, k=8, dropout=0.00)
        else:
            self.model = self.build_model_vgg()
        self.model(np.zeros((1,32,32,3),dtype=np.float32))
        
        self.model.load_weights("./weights_cifar/{}.h5".format(path))

    def build_model_res(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        base_model = ResNet34(input_shape=(32,32,3), include_top=False)
        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        output = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = keras.models.Model(inputs=[base_model.input], outputs=[output])
        return model
        
    def build_model_vgg(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        if self.model_flag == "19":
            model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        if self.model_flag == "19":
            model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        if self.model_flag == "19":
            model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


import tensorflow_probability as tfp
tfd = tfp.distributions

class Detector:
    def __init__(self,class_len=10,layer_ratio=1.0,neuron_num=20,q=0.1,batch_size=100):
        self.class_len=class_len
        self.layer_ratio=layer_ratio
        self.neuron_num=neuron_num
        self.q=q
        self.batch_size=batch_size
        
    def get_key(self,model, m_name):
        key = []
        for i in model.layers:
            if m_name.endswith("13"):
                if type(i)==type(Activation('relu')):
                    if len(i.output.get_shape())==4: 
                        key.append(i.name)
            else:
                if i.name.startswith("add"):
                    key.append(i.name)
        for i in model.layers:
            if len(i.output.get_shape())==2: 
                key.append(i.name)
        return key
    
    def k_fuction(self,model,key):
        outputs = [model.get_layer(name).output for name in key]
        functor = tf.keras.models.Model(model.inputs, outputs)
        #functor = K.function(model.input,outputs)
        return functor
    
    def get_act(self,model,in_data,functor):
        total_len=len(in_data)
        first=True
        acts=[]
        for i in range(0,total_len,self.batch_size):
            start=i
            end=i+self.batch_size
            if end>total_len:
                end=total_len
            ins=in_data[start:end]
            ins=np.array(ins)
            if len(ins.shape) != 4:
                continue
            layer_outs=functor([ins])
            if first:
                first=False
                for a in layer_outs:
                    if len(a.shape) == 4:
                        acts.append(tf.reduce_mean(a, axis=(1,2)))
                    else:
                        acts.append(a)
            else:
                for k,a in enumerate(layer_outs):
                    if len(a.shape) == 4:
                        aa=tf.reduce_mean(a, axis=(1,2))
                    else:
                        aa = a
                    acts[k]=np.append(acts[k],aa,axis=0)
        return acts
    
    def select_neurons(self,model,in_data,functor):
        act_dict = []
        idx_dict = []
        for i in range(len(in_data)):
            act_dict_=[]
            idx_dict_=[]
            acts = self.get_act(model,in_data[i],functor)
            for act in acts:
                max_idx=tf.argsort(tf.math.reduce_std(act,axis=0)/tf.math.reduce_mean(act,axis=0))
                if len(max_idx) != self.class_len:
                    max_idx = max_idx[:4]
                if self.class_len == 10:
                    max_act=act[:,max_idx]
                else:
                    max_act=act.numpy()[:,max_idx]
                act_dict_.append(max_act)
                idx_dict_.append(max_idx)
            act_dict.append(act_dict_)
            idx_dict.append(idx_dict_)
        #indices_list=tf.constant(idx_dict)
        return act_dict,idx_dict
    
    def model_prediction(self,model,in_data,in_label):
        total_len=len(in_data)
        preds=[]
        for i in range(0,total_len,self.batch_size):
            start=i
            end=i+self.batch_size
            if end>total_len:
                end=total_len
            ins=in_data[start:end]
            preds.extend(model.predict(ins).argmax(1))
        return np.where((in_label==preds)==True)[0]
    
    def normalize(self,x_train,x_test):
        mean=np.mean(x_train,axis=(0,1,2,3))
        std=np.std(x_train, axis=(0,1,2,3))
        x_train=(x_train-mean)/(std+1e-7)
        x_test=(x_test-mean)/(std+1e-7)
        return x_train, x_test   
    
    def make_pdf(self,activations,iter_num):
        means = []
        covs = []
        for i in range(iter_num):
            train_act=np.concatenate(activations[i],axis=1).T
            means.append(train_act.mean(1))
            cc=np.cov(train_act)
            cc=cc+np.eye(len(cc))*1e-03
            covs.append(cc)
        means = np.array(means,dtype=np.float32)
        covs = np.array(covs,dtype=np.float32)
        pdf =tfd.MultivariateNormalFullCovariance(loc=means,covariance_matrix=covs)
        return pdf     
    
    def get_act_instance(self,in_data,idx,functor):
        layer_outs = functor([in_data])
        actss  = []
        for k in range(self.class_len):
            acts  = []
            for i,a in enumerate(layer_outs):
                if len(a.shape) == 4:
                    aa = tf.reduce_mean(a, axis=(1,2))
                else:
                    aa = a
                for n ,j in enumerate(idx[k][i]):
                    if i == 0 and n==0:
                        acts.append(aa[:,j])
                    else:
                        acts.append(aa[:,j])
            actss.append(acts)
        actss = tf.stack(actss)
        actss = tf.transpose(actss, perm=[2,0,1])
        return actss
    
    def make_inference_form(self,in_data,idx,functor):
        instance_act =self.get_act_instance(in_data,idx,functor)
        return instance_act
    def calculate_prob(self,in_data,functor,indices,pdf):
        instance_act=self.make_inference_form(in_data,indices,functor)
        prob=pdf.log_prob(instance_act)
        return prob  


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
    
model_name = args.model

class_len = int(model_name.split("_")[0][5:])
model_flag = model_name.split("_")[1]
batch_size = 100
print(model_name, args.eps)
neuron_num = 4#10
layer_ratio = 10 * 0.1#0.5
quantile = 5*0.01

if class_len == 10:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_length = 1000
else:
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    train_length = 250
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_test=y_test.flatten()
K.set_learning_phase(False)
model = CifarModel(model_name, class_len).model
model(np.zeros((1,32,32,3),dtype=np.float32))
inputs = tf.keras.Input(shape=(32,32,3))
model.call(inputs);
model = keras.models.Model(inputs=[model.input], outputs=[model.output])
model.trainable = False
D=Detector(class_len=class_len, 
           layer_ratio=layer_ratio, 
           neuron_num=neuron_num, 
           q=quantile,
           batch_size=batch_size)
x_train, x_test = D.normalize(x_train, x_test)
train_dict = {}
for i in range(y_train.max()+1):
    train_dict[i] = []
for i,l in enumerate(y_train):
    train_dict[l[0]].append(x_train[i])
for i in range(class_len):
    train_dict[i] = np.array(random.sample(train_dict[i],  train_length))
key = D.get_key(model,model_name)
functor = D.k_fuction(model, key)
activations, indices = D.select_neurons(model, train_dict, functor)
pdf = D.make_pdf(activations, class_len)

def attack_ours(image, D, functor, indices, pdf,model, eps):
    ori_image = image.copy()
    mi=ori_image.min()
    ma=ori_image.max()
    pred=model(image)
    pred = pred.numpy().argmax(1)
    pm = pred.max()+1
    pp = pred + 1
    pp[pm==pp] = 0
    backup = ori_image
    ss = 0
    for i in range(100):
        with tf.GradientTape() as tape:
            if i == 0:
                image = np.array(image)
                image = tf.Variable(image,dtype=tf.float32)
            tape.watch(image)
            loss = -D.calculate_prob(image, functor, indices, pdf)
            losses = []
            for n,j in enumerate(pp):
                losses.append(loss[n,j])
            loss = tf.stack(losses)
        gradient = tape.gradient(loss, image)
        perturbations = tf.sign(gradient)
        adv_x = image + (eps/255.)*perturbations
        eta = tf.clip_by_value(adv_x-ori_image,-0.3,0.3)
        image = tf.clip_by_value(ori_image + eta,mi,ma)
        pred_v=model(image).numpy().argmax(1)
        ss_ = np.sum(pred_v!=pred)
        if ss < ss_:
            ss = ss_
            backup = image
    return backup
    
    
from tqdm import tqdm
from PIL import Image
import os

    
total = 0
correct = 0
adv_index = []
adv_example = []
total_len = len(x_test)
batch_size = 100
with tqdm(range(0,total_len,batch_size)) as pbar:
    for i in pbar:
        start=i
        end=i+batch_size
        if end>total_len:
            end=total_len
        label = y_test[start:end]
        adv = attack_ours(x_test[start:end],D,functor,indices,pdf,model, eps)
        pred = model(adv)
        pred = tf.argmax(pred,1).numpy()
        correct += np.array(pred == label).sum()
        total +=batch_size
        acc = 100 * float(correct) / total
        
        image = np.array(x_test[start:end])
        image = tf.Variable(image,dtype=tf.float32)
        pred_t = model(image)
        pred_t = tf.argmax(pred_t,1).numpy()
        adv_index.extend(np.array(pred_t != pred))
        pbar.set_postfix(loss=np.mean(adv_index), acc=acc)
        adv_example.extend(adv.numpy())
        
adv_example = np.array(adv_example)
adv_index = np.array(adv_index)
os.makedirs("./our/{}_{}".format(model_name,str(eps)),exist_ok=True)
np.save("./our/{}_{}/OUR1_example".format(model_name,str(eps)),adv_example)
np.save("./our/{}_{}/OUR1_idx".format(model_name,str(eps)),adv_index)

print('Accuracy of test text: %f %%' % (100 * float(correct) / total))





