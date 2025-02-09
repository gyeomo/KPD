import warnings
warnings.filterwarnings(action='ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras as keras
from tensorflow.compat.v1.keras.layers import Activation, Flatten
from tensorflow.compat.v1.keras import optimizers
from tensorflow.compat.v1.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import scipy
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import *
import sklearn.metrics as metrics
import random

random.seed(777)
tf.set_random_seed(777)
np.random.seed(777)
class Detector:
    def __init__(self,class_len=10,layer_ratio=1.0,neuron_num=20,q=0.1,batch_size=100):
        self.class_len=class_len
        self.layer_ratio=layer_ratio
        self.neuron_num=neuron_num
        self.q=q
        self.batch_size=batch_size
    
    # Find out ReLU layers with the pre-defined layer ratio
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
        print(key)
        return key
    
    # Kernels can be directly accessed through the 'key'
    def k_fuction(self,model,key):
        outputs = [model.get_layer(name).output for name in key]
        functor = K.function([model.layers[0].input,K.set_learning_phase(0)],outputs)
        return functor
    
    # GAP
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
                    acts.append(a.mean(axis=(1,2)))
            else:
                for k,a in enumerate(layer_outs):
                    aa=a.mean(axis=(1,2))
                    acts[k]=np.append(acts[k],aa,axis=0)
        return acts
    
    # Remove outliers
    def quantile(self,activation,lower,upper):
        if (lower+upper)!=1.0:
            print("boundary error")
            return -1 
        q=np.array([lower,upper])
        size=int(activation.shape[0]*(1-lower*2))+1
        iqr=np.quantile(activation,q,axis=0)
        act=[]
        for n in range(activation.shape[1]):
            act_n=activation[:,n]
            act_n = np.clip(act_n, a_min=iqr[:,n][0], a_max=iqr[:,n][1])
            while(True):
                if len(act_n)>=size:
                    break
                a=np.expand_dims(np.mean(act_n),axis=0)
                act_n=np.concatenate([act_n,a],axis=0)
            act.append(act_n)
        act = np.array(act)
        act = act.transpose(1,0)
        return act
    
    # Finding kernels with low CV
    def select_neurons(self,model,in_data,functor):
        act_dict = {}
        idx_dict = {}
        for i in range(len(in_data)):
            act_dict[i]=[]
            idx_dict[i]=[]
            acts = self.get_act(model,in_data[i],functor)
            for act in acts:
                max_idx=np.argsort(act.std(0)/act.mean(0))
                if len(max_idx) != self.class_len:
                    max_idx = max_idx[:self.neuron_num]
                max_act=act[:,max_idx]
                if self.q>0:
                    max_act=self.quantile(max_act,self.q,1.0-self.q)
                act_dict[i].append(max_act)
                idx_dict[i].append(max_idx)
        indices_list=[]
        for i in range(len(in_data)):
            indices_list.append(idx_dict[i])
        indices_list=np.array(indices_list)
        return act_dict,indices_list
    
    # Constructing path distribution
    def make_pdf(self,activations,iter_num):
        means = []
        covs = []
        for i in range(iter_num):
            train_act=np.concatenate(activations[i],axis=1).T
            means.append(train_act.mean(1))
            cc=np.cov(train_act)
            cc=cc+np.eye(len(cc))*1e-03
            covs.append(cc)
        # mean vector
        means = np.array(means,dtype=np.float32)
        # covariance matrix
        covs = np.array(covs,dtype=np.float32)
        pdf =tfd.MultivariateNormalFullCovariance(loc=means,covariance_matrix=covs)
        return pdf      
    
    # Collect outputs of a test sample on the paths.
    def get_act_instance(self,in_data,idx,functor):
        layer_outs = functor([in_data])
        actss = []
        for j in range(self.class_len):
            for i,a in enumerate(layer_outs):
                if len(a.shape) == 4:
                    aa = a.mean(axis=(1,2))
                else:
                    aa = a
                if i == 0:
                    acts=aa[:,idx[j][i]]
                else:
                    acts=np.append(acts,aa[:,idx[j][i]],axis=1)
            actss.append(acts)
        actss = np.array(actss).transpose(1,0,2)
        return actss
    
    
    def make_inference_form(self,sample,idx,functor):
        instance_act=self.get_act_instance(sample,idx,functor)
        return instance_act
    
    
    def calculate_prob(self,model,in_data,functor,indices,pdf):
        total_len=len(in_data)
        out_data=[]
        for i in range(0,total_len,self.batch_size):
            start=i
            end=i+self.batch_size
            if end>total_len:
                end=total_len
            ins=in_data[start:end]
            instance_act=self.make_inference_form(ins,indices,functor)
            prob=pdf.log_prob(instance_act).eval()
            pred = model.predict(ins)
            prob = prob[np.arange(len(pred)),pred.argmax(1)].reshape(-1,1)
            prob = np.nan_to_num(prob, copy=True, nan=0.0, posinf=0, neginf=0)
            out_data.extend(prob)
        out_data = np.array(out_data)
        return out_data


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
    
    
    def model_prediction_adv(self,model,in_data,in_label):
        total_len=len(in_data)
        preds=[]
        for i in range(0,total_len,self.batch_size):
            start=i
            end=i+self.batch_size
            if end>total_len:
                end=total_len
            ins=in_data[start:end]
            preds.extend(model.predict(ins).argmax(1))
        return np.where((in_label!=preds)==True)[0]
        
    
    def report(self, label, value):
        pr_auc=metrics.average_precision_score(label,value)
        roc_auc=metrics.roc_auc_score(label,value)
        fpr,tpr,_=metrics.roc_curve(label,value)
        tpr90_pos=np.abs(tpr-0.90).argmin()
        tnr_at_tpr90=1.0-fpr[tpr90_pos]
        fpr10_pos=np.abs(fpr - 0.10).argmin()
        tpr_at_fpr10=tpr[fpr10_pos]
        return roc_auc,pr_auc,tnr_at_tpr90,tpr_at_fpr10
    
    # load adversarial examples
    def get_adv_data(self,model_name,attack_type):
        file_name1="example.npy"
        path_example="./attack_example/{}/{}_{}".format(model_name,attack_type,file_name1) 
        examples=np.load(path_example,allow_pickle=True)
        return examples
    
    
    def normalize(self,x_train,x_test):
        mean=np.mean(x_train,axis=(0,1,2,3))
        std=np.std(x_train, axis=(0,1,2,3))
        x_train=(x_train-mean)/(std+1e-7)
        x_test=(x_test-mean)/(std+1e-7)
        return x_train, x_test
    
        
