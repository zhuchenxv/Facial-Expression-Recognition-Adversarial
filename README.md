# Facial-Expression-Recognition-Adversarial

## About

This is the course project of CS420 Machine Learning, SJTU. We have tried diﬀerent methods to deal with the project, including CNN+Facial_landmarks+HOG, ResNet18 and VGG19. The highest accuracy can reach to 73.11% in private test. Afterwards, we use Fast Gradient Sign Method to attack VGG19, resulting in the accuracy dropping to 21.12%. Then, we conduct the adversarial training, raising the accuracy to 33.60%.

## Getting started

### CNN+Facial_landmarks+HOG

#### Prerequisites

* Download the dataset (fer2013.csv)
* Download the Face Landmarks model (Dlib Shape Predictor model)

#### Convert the dataset to extract Face Landmarks and HOG Features

  ```
  python convert_fer2013_to_images_and_landmarks.py
  ```

#### Train the model

* Choose the parameters in 'parameters.py'

* Optimize training hyperparameters.

  ```
  python optimize_hyperparams.py --max_evals=20
  ```

* Launch training

  ```
  python train.py --train=yes
  ```
### ResNet18 and VGG19

#### Prerequisites

* Download the dataset (fer2013.csv)

* Preprocess the data

  ```
  python preprocess_fer2013.py
  ```

#### Train the model

* Launch training

  ```
  python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01
  ```

  You can also use different parameters incluing model (VGG19, ResNet18), batch size and learning rate.
