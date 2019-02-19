import sys
sys.path.insert(0, './src')

import Linear
import ReLU
import Model

import argparse
import torch
import torchfile
USAGE="""
(a) -config which is the /path/to/modelConfig.txt
(b) -i which is the /path/to/input.bin
(c) -og which is the /path/to/gradOutput.bin
(d) -o which is the /path/to/output.bin
(e) -ow which is the /path/to/gradWeight.bin
(f) -ob which is the /path/to/gradB.bin and
(g) -ig which is the /path/to/gradInput.bin """

if __name__=='__main__':
    # my_input=OptionParser(USAGE)
    # my_input.add_option('-c','--config', type="string",dest ="path_to_config" ,default=".")

    # (options, args) = my_input.parse_args()
    # print("here")
    # print options,args	
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='Give Config Path',dest ="path_config",default='./bestModel/modelConfig.txt')
    parser.add_argument('-i', help='Give input.bin path',dest ="path_i",default='./input/input.bin')
    parser.add_argument('-og', help='give gradOutput.bin path',dest ="path_og",default='./input/gradOutput.bin')
    parser.add_argument('-o', help='give output.bin path',dest ="path_o",default='./output/output.bin')
    parser.add_argument('-ow', help='give gradWeight.bin path',dest ="path_ow",default='./output/gradWeight.bin')
    parser.add_argument('-ob', help='give gradB.bin path',dest ="path_ob",default='./output/gradB.bin')
    parser.add_argument('-ig', help='give gradInput.bin',dest ="path_ig",default='./output/gradInput.bin')
    args = parser.parse_args()
    
    network=Model.Model()
    print(args.path_config)
    network.loadModel(args.path_config)
    my_inp=torch.tensor(torchfile.load(args.path_i))
    # print(sz)
    grad_Out=torch.tensor(torchfile.load(args.path_og))
    # print(grad_Out.size())
    sz=my_inp.size()
    t=1
    for i in range(1,len(sz)):
        t*=sz[i]
    my_inp=my_inp.reshape(sz[0],t)
    out=network.forward(my_inp)
    torch.save(out,args.path_o)

    Input_grad=network.backward(my_inp,grad_Out)
    torch.save(Input_grad,args.path_ig)

    network.save_Grads(args.path_ow,args.path_ob)
    # print args.path