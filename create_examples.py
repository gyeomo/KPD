import warnings
warnings.filterwarnings(action='ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.compat.v1.keras.losses import categorical_crossentropy
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
from art.attacks.evasion import GeoDA
from art.attacks.evasion import GRAPHITEBlackbox
from art.attacks.evasion import PixelAttack 
from art.attacks.evasion import SignOPTAttack
from art.attacks.evasion import SquareAttack
from art.attacks.evasion import SimBA
from art.attacks.evasion import DeepFool
from art.attacks.evasion import BasicIterativeMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import CarliniLInfMethod
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import AutoProjectedGradientDescent

from art.estimators.classification import KerasClassifier
from art.utils import load_cifar10
from cifar_model import CifarModel
import os
import random
# Step 1: Load the MNIST dataset
        
import argparse
parser = argparse.ArgumentParser(description='KPD with Tensorflow')
parser.add_argument('--att', default="OPT",type=str,
                    help='check number')
parser.add_argument('--model', default="cifar10_re",type=str,
                    help='check number')
parser.add_argument('--eps', default=8,type=int,
                    help='check number')
args = parser.parse_args()
eps = args.eps
model_name = args.model
class_len = int(model_name.split("_")[0][5:])

def normalize(X_train,X_test):
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test
if class_len==10:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
else:
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_test = keras.utils.to_categorical(y_test, class_len)
x_train, x_test = normalize(x_train, x_test)
min_pixel_value = x_test.min()
max_pixel_value = x_test.max()
#x_test = x_test[27:].copy()
#y_test = y_test[27:].copy()
model = CifarModel(model_name).model
model(np.zeros((1,32,32,3),dtype=np.float32))
inputs = tf.keras.Input(shape=(32,32,3))
if model_name == "cifar10_13" or model_name == "cifar100_13":
    model = keras.models.Model(inputs=[inputs],outputs=[model(inputs)])
else:
    model.call(inputs);
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

#predictions = classifier.predict(x_test)
#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print("Accuracy on benign test examples: {}%".format(accuracy * 100))

if args.att == "FGSM":
    attack = FastGradientMethod(estimator=classifier, eps = 8./255., batch_size=100)
    x_test_adv = attack.generate(x=x_test)
elif args.att =="PGD":
    attack = ProjectedGradientDescent(estimator=classifier, eps = 8./255., batch_size=100)
    x_test_adv = attack.generate(x=x_test)
elif args.att =="BIM":
    attack = BasicIterativeMethod(estimator=classifier, eps = 8./255., batch_size=100)
    x_test_adv = attack.generate(x=x_test)
elif args.att =="APGD":
    attack = AutoProjectedGradientDescent(estimator=classifier, eps = 8./255., batch_size=100)
    x_test_adv = attack.generate(x=x_test)
elif args.att =="DF":
    attack = DeepFool(classifier=classifier, batch_size=100)
    x_test_adv = attack.generate(x=x_test)
elif args.att =="CW":
    attack = CarliniLInfMethod(classifier=classifier, batch_size=100)
    x_test_adv = attack.generate(x=x_test)
elif args.att =="GEO":
    attack = GeoDA(estimator=classifier, sub_dim = 3, max_iter = 300)
    x_test_adv = attack.generate(x=x_test)
elif args.att =="PIX":
    attack = PixelAttack(classifier=classifier,max_iter=10,verbose=True)
    x_test_adv = attack.generate(x=x_test)
elif args.att =="OPT":
    attack = SignOPTAttack(estimator=classifier,num_trial=3,max_iter=10,targeted=False,query_limit=1000,verbose=True)
    x_test_adv = attack.generate(x=x_test)
elif args.att =="SBA":
    attack = SimBA(classifier=classifier, max_iter = 50)
    x_test_adv = attack.generate(x=x_test)
elif args.att =="SQA":
    attack = SquareAttack(estimator=classifier,eps=8./255.)
    x_test_adv = attack.generate(x=x_test)

predictions = classifier.predict(x_test_adv)
indices = np.array(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)).flatten()
accuracy = np.sum(indices) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

os.makedirs("./attack_example/{}_{}/".format(model_name,str(eps)),exist_ok=True)
np.save("./attack_example/{}_{}/{}_example".format(model_name,str(eps),args.att), x_test_adv)
np.save("./attack_example/{}_{}/{}_idx".format(model_name,str(eps),args.att), indices)