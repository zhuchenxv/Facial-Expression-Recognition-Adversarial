# Facial-Expression-Recognition-Adversarial

## Main idea

We have tried diﬀerent methods to deal with the project, including CNN+Facial landmarks+HOG, ResNet18 and VGG19. The highest accuracy can reach to 73.11% in private test. Afterwards, we use Fast Gradient Sign Method to attack VGG19, resulting in the accuracy dropping to 21.12%. Then, we conduct the adversarial training, raising the accuracy to 33.60%.

###              fer2013 Accurary             ###

- Model：    VGG19 ;       PublicTest_acc：  71.496% ;     PrivateTest_acc：73.112%     <Br/>
- Model：   Resnet18 ;     PublicTest_acc：  71.190% ;    PrivateTest_acc：72.973%     
