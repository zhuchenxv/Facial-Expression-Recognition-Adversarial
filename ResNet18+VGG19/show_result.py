import torch
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	result = torch.load('FER2013_VGG19_adv/Result.t7', map_location=lambda storage, loc: storage)
	PublicTest_acc_list = result['PublicTest_acc_list']
	PrivateTest_acc_list= result['PrivateTest_acc_list']
	PublicTest_adv_acc_list=result['PublicTest_adv_acc_list']
	PrivateTest_adv_acc_list=result['PrivateTest_adv_acc_list']
	for i in range(len(PublicTest_adv_acc_list)):
		PublicTest_adv_acc_list[i]=PublicTest_adv_acc_list[i]*100
		PrivateTest_adv_acc_list[i]=PrivateTest_adv_acc_list[i]*100
	Loss_list=result['Loss_list']

	print(max(PublicTest_acc_list))
	print(max(PrivateTest_acc_list))
	print(max(PublicTest_adv_acc_list))
	print(max(PrivateTest_adv_acc_list))

	plt.plot(np.arange(len(PublicTest_acc_list)), PublicTest_acc_list,label='PublicTest Accuracy')
	plt.plot(np.arange(len(PrivateTest_acc_list)), PrivateTest_acc_list,label='PrivateTest Accuracy')
	plt.plot(np.arange(len(PublicTest_adv_acc_list)), PublicTest_adv_acc_list,label='PublicTest Adversarial Accuracy')
	plt.plot(np.arange(len(PrivateTest_adv_acc_list)), PrivateTest_adv_acc_list,label='PrivateTest Adversarial Accuracy')
	plt.xlim([0,250])
	plt.ylim([0,100])
	plt.xlabel('Trainging Step')
	plt.ylabel('Accuracy(%)')
	plt.legend()
	plt.show()
	plt.plot(np.arange(len(Loss_list)), Loss_list,label='Loss_list')
	plt.show()

