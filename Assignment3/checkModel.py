import sys
sys.path.insert(0, './src')

import argparse
import Model
import Linear
import ReLU
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


def read_from_file_and_create_nn(path_config):
    with open(path_config) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
    print (content)
    no_layers=int(content[0])
    print (no_layers)
    indices=[]
    network=Model.Model()
    for i in range(1,len(content)-2):
        words=content[i].split()
        # print(words)
        if(words[0]=='linear'):
            in_nodes=int(words[1])
            out_nodes=int(words[2])
            print("creating linear layer with " + str(in_nodes) +" "+str(out_nodes))
            network.addLayer(Linear.Linear(in_nodes,out_nodes))
            indices.append(i-1)
        elif(words[0]=='relu'):
            print("creating relu layer")
            network.addLayer(ReLU.ReLU())
    print(indices)
    layer_w_path=content[-2]
    layer_bias_path=content[-1]
    print(layer_w_path)
    print(layer_bias_path)
    weights=torchfile.load(layer_w_path)
    bias=torchfile.load(layer_bias_path)
    j=0
    for i in indices:
        network.setParams(weights[j],bias[j],i)
        j+=1
    return network


if __name__=='__main__':
    # my_input=OptionParser(USAGE)
    # my_input.add_option('-c','--config', type="string",dest ="path_to_config" ,default=".")

    # (options, args) = my_input.parse_args()
    # print("here")
    # print options,args	
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='Give Config Path',dest ="path_config",default='./myfiles/modelConfig.txt')
    parser.add_argument('-i', help='Give input.bin path',dest ="path_i",default='./myfiles/input.bin')
    parser.add_argument('-og', help='give gradOutput.bin path',dest ="path_og",default='./myfiles/gradOutput.bin')
    parser.add_argument('-o', help='give output.bin path',dest ="path_o",default='./myfiles/output.bin')
    parser.add_argument('-ow', help='give gradWeight.bin path',dest ="path_ow",default='./myfiles/gradWeight.bin')
    parser.add_argument('-ob', help='give gradB.bin path',dest ="path_ob",default='./myfiles/gradB.bin')
    parser.add_argument('-ig', help='give gradInput.bin',dest ="path_ig",default='./myfiles/gradInput.bin')
    args = parser.parse_args()
    
    # network=Model.Model()
    network=read_from_file_and_create_nn(args.path_config)
    my_inp=torch.tensor(torchfile.load(args.path_i))
    print(my_inp.size())
    # print(len(my_inp[1]))
    # print(len(my_inp[1][2]))
    # print(len(my_inp[1][2][5]))
    # print(len(my_inp[1][2][5][2]))

    grad_Out=torchfile.load(args.path_og)
    
    
    # print args.path