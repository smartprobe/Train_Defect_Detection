import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class _External(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self,output_dim):
        super(_External, self).__init__()
        self.output_dim =output_dim
        self.Linear1 = nn.Linear(2, 10)
        self.Linear2 = nn.Linear(10, self.output_dim)
        self.Linear3 = nn.Linear(self.output_dim*2, self.output_dim)

    def forward(self, pooled,rois):
        embedding = self.Linear1(rois)
        embedding = F.relu(embedding)
        embedding = self.Linear2(embedding)
        embedding = F.relu(embedding)
        x = torch.cat([pooled,embedding],1)
        x = self.Linear3(x)
        x = F.tanh(x)




        return x




