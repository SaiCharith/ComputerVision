import os

import torch
import torchfile


# run scripts
file_folder='./myfiles/'
output_folder='./output/'
ext='.bin'
list=['1','2']
for l in list:
    #check_Models
    my_pref='my_'
    string_base ='python3 checkModel.py '
    string_conf = ' -config '+file_folder+'modelConfig_'+l+'.txt'
    string_inp = ' -i '+file_folder+'input_sample_'+l+ext
    string_grad_Out=' -og '+file_folder+'gradOutput_sample_'+l+ext
    string_out=' -o '+output_folder+my_pref+'output_sample_'+l+ext
    string_gradW=' -ow '+output_folder+my_pref+'gradW_sample_'+l+ext
    string_gradB=' -ob '+output_folder+my_pref+'gradB_sample_'+l+ext
    string_gradInp=' -ig '+output_folder+my_pref+'gradInput_sample_'+l+ext
    string=string_base+string_conf+string_inp+string_grad_Out+string_out+string_gradW+string_gradB+string_gradInp
    os.system(string)

    #compare with existing
    list_to_check=['output','gradB','gradW']
    for i in range(len(list_to_check)):
        print(list_to_check[i])
        sample_out=torchfile.load(file_folder+list_to_check[i]+'_sample_'+l+'.bin')
        my_out=torch.load(output_folder+my_pref+list_to_check[i]+'_sample_'+l+'.bin')
        if i==0:
            sample_out=torch.tensor(sample_out)
            print(torch.sum((sample_out-my_out)**2))
        else:
            for m in range(len(my_out)):
                print(m)
                sample_1=torch.tensor(sample_out[m])
                my_1=my_out[m]
                print(torch.sum((sample_1-my_1)**2))

