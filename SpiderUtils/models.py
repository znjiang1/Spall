import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class Spider(nn.Module):
    def __init__(self, input_num_features, hid_units, n_heads, nb_classes, edge_dim, dropout=0.0, activation=nn.ELU()):
        super(Spider, self).__init__()
        self.activation = activation
        self.conv1 = GATv2Conv(input_num_features, hid_units[0], heads=n_heads[0], dropout=dropout, concat=False, edge_dim=edge_dim, add_self_loops=False)
        self.conv2 = GATv2Conv(hid_units[0], nb_classes, heads=n_heads[1], dropout=dropout, concat=False, edge_dim=edge_dim, add_self_loops=False)
        self.res_fc1 = nn.Linear(input_num_features, hid_units[0], bias=False)

    def forward(self, data):
        x0, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1 = self.conv1(x0, edge_index, edge_attr=edge_attr)
        x1 = self.activation(x1)
        x1 = x1 + self.res_fc1(x0)

        x2 = self.conv2(x1, edge_index, edge_attr=edge_attr)
        return x2

