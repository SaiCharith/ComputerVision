import os

import torch
import torchfile

sample_dir='./myfiles/'
my_dir='./output/'

sample_sufix='_sample_2'
sample_list=['output','gradB','gradW']
my_list=['output','gradB','gradW']


for l in range(len(my_list)):
    print(my_list[l])
    print(sample_list[l])
    sample_out=torchfile.load(sample_dir+sample_list[l]+sample_sufix+'.bin')
    my_out=torch.load(my_dir+my_list[l]+sample_sufix+'.bin')
    # print(sample_out)
    # print(my_out)
    if l==0:
        sample_out=torch.tensor(sample_out)
        print(torch.sum((sample_out-my_out)**2))
    else:
        # if(l==1):   
        #     print(sample_out)
        for m in range(len(my_out)):
            
            sample_1=torch.tensor(sample_out[m])
            my_1=my_out[m]
            # if m==0:
            #     print(sample_1)
            #     print(my_1)
            print(torch.sum((sample_1-my_1)**2))




