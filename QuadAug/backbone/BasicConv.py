import torch
import torch.nn as nn
from collections import OrderedDict
from utils.compute_conv import compute_conv_output_size

class BasicConv(nn.Module):
    def __init__(self,input_size: int, output_size: int):
        super(BasicConv,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conv2_feature = None
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []

        self.map.append(28)
        self.ksize.append(2)
        self.in_channel.append(1)

        s = compute_conv_output_size(28, 2)
        s = s // 2
        self.map.append(s)
        self.ksize.append(2)
        self.in_channel.append(64)

        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.map.append(s)
        self.ksize.append(2)
        self.in_channel.append(128)

        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.map.append(s)
        self.ksize.append(2)
        self.in_channel.append(128)

        self.smid = s
        self.map.append(256 * self.smid * self.smid)
        self.map.extend([1000])

        self.conv1 = nn.Conv2d(in_channels=self.input_size,out_channels=64,kernel_size=(2,2),stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=1)
        self.relu = nn.ReLU()
        self.pool2d =  nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(in_features=1024, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=self.output_size)


    def forward(self, x: torch.Tensor):
        self.act['conv1'] = x
        x = self.conv1(x)
        x = self.pool2d(self.relu(x))

        self.act['conv2'] = x
        x = self.conv2(x)
        self.conv2_feature = x
        x = self.pool2d(self.relu(x))

        self.act['conv3'] = x
        x = self.conv3(x)
        x = self.pool2d(self.relu(x))

        x = x.view(x.size(0), -1)
        self.act['fc1'] = x
        x = self.fc1(x)
        x = self.relu(x)

        self.act['fc2'] = x
        x = self.fc2(x)
        x = self.relu(x)

        out = self.fc3(x)
        return out

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                                               torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)