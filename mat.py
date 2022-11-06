import torch
while True:
    a = torch.rand(7680,768).to("cuda:0")
    a = a.matmul(a.transpose(0,1))
    a = torch.rand(7680,768).to("cuda:1")
    a = a.matmul(a.transpose(0,1))
    a = torch.rand(7680,768).to("cuda:2")
    a = a.matmul(a.transpose(0,1))
    a = torch.rand(7680,768).to("cuda:3")
    a = a.matmul(a.transpose(0,1))
    a = torch.rand(7680,768).to("cuda:4")
    a = a.matmul(a.transpose(0,1))
    a = torch.rand(7680,768).to("cuda:5")
    a = a.matmul(a.transpose(0,1))
    a = torch.rand(7680,768).to("cuda:6")
    a = a.matmul(a.transpose(0,1))
    a = torch.rand(7680,768).to("cuda:7")
    a = a.matmul(a.transpose(0,1))

