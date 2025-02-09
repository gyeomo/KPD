# KPD
KPD: An Efficient and Effective Detector Against Adversarial Attacks and Out-of-Distribution based on Kernel Path Distribution (PAKDD 2025)

# Requirements
- CUDA == 11.1
- cudnn == 8.1.0
- foolbox == 3.3.3
- keras == 2.4.0
- numpy == 1.19.5
- scikit-learn == 0.24.2
- scipy == 1.4.1
- silence-tensorflow == 1.2.0
- tensorflow == 2.5.2
- tensorflow-gpu == 2.5.0


# Implementation
* defauts:
  * Adversarial attacks: CW. 
  * Model: VGG13 trained on CIFAR10.

```sh
create_examples.py  # it creates adversarial examples
```

```sh
detector.py # it detects the adversarial examples.
```

```sh
adaptive.py # it creates the adaptive attackk examples.
```

Modify "cifar_model.py" to run different pre-trained model.

Modify "create_examples.py" to examine against more diverse attacks.



