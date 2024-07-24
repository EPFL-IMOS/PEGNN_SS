from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
from torch_geometric.nn import ChebConv, GATv2Conv, TransformerConv

class GNNModel(nn.Module):
    def __init__(self, input_dim = 8, hidden_dim = 16, out_dim = 8, num_heads = 5):
        super().__init__()
        self.GCN1 = GATv2Conv(input_dim, hidden_dim, heads = num_heads, add_self_loops = False)
        self.GCN2 = GATv2Conv(num_heads*hidden_dim, hidden_dim, heads = num_heads, add_self_loops = False)
        self.linear = nn.Linear(2* num_heads*hidden_dim, out_dim)
        self.linear1 = nn.Linear(38, 128)
        self.linear2 = nn.Linear(128 , 37)

    def forward(self, data):
      x, edge_index, edge_weight = data.x.float(), data.edge_index, data.edge_weight
      length = x.shape[1]
      x = F.selu(self.GCN1(x, edge_index, edge_weight))
      x_prev1 = x
      x = F.selu(self.GCN2(x, edge_index, edge_weight))
      x = torch.cat([x_prev1, x], dim = 1)
      x = F.selu(self.linear(x))
      x = F.selu(self.linear1(x.view(-1, 38, length).permute(0 , 2, 1)))
      x = self.linear2(x)
      return x