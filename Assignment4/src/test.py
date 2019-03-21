import RNN
import torch
import Model

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

r = RNN.RNN(2,5,3)
# inp = torch.randn(10,100,2, dtype=dtype, device=device)
inp = [torch.randn(10,2, dtype=dtype, device=device) for _ in range(100)]
import RNN
fwd = r.forward(inp)
grads = [torch.zeros(10,3,dtype=dtype, device=device) for _ in range(100)]
r.clearGradParam()
r.backward(inp,grads)
