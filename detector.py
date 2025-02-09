import warnings
warnings.filterwarnings(action='ignore')
import argparse
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.keras.datasets import cifar10, cifar100
from tensorflow.compat.v1.keras import backend as K
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import *
from functions import Detector
from cifar_model import CifarModel
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

tf.set_random_seed(777)
random.seed(777)
np.random.seed(777)
parser = argparse.ArgumentParser(description='KPD: Kernel Path Distribution for abnormal data detection')
parser.add_argument('--ratio', default=10,type=int,
                    help='check number')
parser.add_argument('--num', default=20,type=int,
                    help='check number')
parser.add_argument('--q', default=5,type=int,
                    help='check number')
parser.add_argument('--l', default=5000,type=int,
                    help='check number')
parser.add_argument('--model', default="cifar10_13",type=str,
                    help='check number')
args = parser.parse_args()

model_name = args.model
class_len = int(model_name.split("_")[0][5:])
model_flag = model_name.split("_")[1]

batch_size = 400
    
neuron_num = args.num
layer_ratio = args.ratio * 0.1
quantile = args.q*0.01
print("layer_ratio: %3.2f neuron_num: %3.d quantile: %3.3f train length: %d"%(layer_ratio,neuron_num,quantile, args.l))
print("selected model name:",model_name)

# ./attack_example/{model name}/{attack name}
# Please make sure that the data regarding the adversarial attacks exists on the path.
attack_list = [
               "L2CW", 
              ]
# load the dataset.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_test=y_test.flatten()
with K.get_session():
    K.set_learning_phase(0)
    # load the model.
    model = CifarModel(model_name).model
    model(np.zeros((1,32,32,3),dtype=np.float32))
    inputs = tf.keras.Input(shape=(32,32,3))
    model.call(inputs);
    # make the detector
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
    # Find ReLU layer
    key = D.get_key(model, model_name)
    # Directly access to kernels.
    functor = D.k_fuction(model, key)
    # Load kernel paths and their indices.
    activations, indices = D.select_neurons(model, train_dict, functor)
    # constructing kernel path distribution
    pdf = D.make_pdf(activations, class_len)
    # what is the class label of the clean data?
    c_idx=D.model_prediction(model, x_test, y_test)

    for att in attack_list:
        # load adversarial examples
        example = D.get_adv_data(model_name, att)
        # sampling
        x_select, y_select, x_adv_test = x_test[c_idx], y_test[c_idx], example[c_idx]
        e_idx=D.model_prediction_adv(model, x_adv_test, y_select)
        x_select, y_select, x_adv_test = x_select[e_idx], y_select[e_idx], x_adv_test[e_idx]
        x_select = np.array(random.choices(x_select, k=1000))#x_select[idx]#np.array(random.choices(x_select, k=1000))
        x_adv_test = np.array(random.choices(x_adv_test, k=1000))
        # log probabilities
        prob_clean = D.calculate_prob(model, x_select, functor, indices, pdf)
        prob_adv = D.calculate_prob(model, x_adv_test, functor, indices, pdf)
        
        # data composition for training the linear classifier 
        label_clean = np.zeros(len(prob_clean), np.int)
        label_adv = np.ones(len(prob_adv), np.int)
        split = round(len(prob_clean)/2)
        cls_train_label = np.concatenate([label_clean[:split],label_adv[:split]])
        cls_train_data = np.concatenate([prob_clean[:split],prob_adv[:split]])
        cls_test_label = np.concatenate([label_clean[split:],label_adv[split:]])
        cls_test_data = np.concatenate([prob_clean[split:],prob_adv[split:]])
        
        # Training the classifier
        cls = LogisticRegression(random_state=777)
        cls.fit(cls_train_data, cls_train_label)
        
        # run test
        prob = cls.predict_proba(cls_test_data)[:,1]
        roc_auc, pr_auc, tnr_at_tpr90, tpr_at_fpr10=D.report(cls_test_label,prob)
        print("Attack: %9s ROC %.3f, PR %.3f TNR at TPR90 %.3f TPR at FPR10 %.3f" % (att, roc_auc, pr_auc, tnr_at_tpr90, tpr_at_fpr10))


