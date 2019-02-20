import sys
sys.path.insert(0, './src')

import Linear
import ReLU
import Model
import Criterion

import argparse
import torch
import torchfile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


USAGE=""" 
(a) -i /path/to/input.bin
(b) -t /path/to/target.bin
(c) -ig /path/to/gradInput.bin
"""

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Give input.bin path',dest ="path_i",default='./input/input.bin')
    parser.add_argument('-t', help='give target.bin path',dest ="path_t",default='./input/target.bin')
    parser.add_argument('-ig', help='give gradInput.bin',dest ="path_ig",default='./output/gradInput.bin')
    args = parser.parse_args()

    my_inp=torch.tensor(torchfile.load(args.path_i),device=device,dtype=torch.double)
    my_target=torch.tensor(torchfile.load(args.path_t),dtype=torch.long,device=device).reshape(my_inp.size()[0])
    my_target-=1
    print(my_target.size())
    mycrit=Criterion.Criterion()

    avgLoss=mycrit.forward(my_inp,my_target)
    print("Average Loss is = "+str(avgLoss))

    Input_Grad=mycrit.backward(my_inp,my_target)
    torch.save(Input_Grad,args.path_ig)

    