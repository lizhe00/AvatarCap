import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels = [512, 512, 512, 343, 512, 512],
                 res_layers = [], nlactv = 'relu', last_op='sigmoid', norm = None):
        super(MLP, self).__init__()

        if nlactv == 'leaky_relu':
            self.nlactv = nn.LeakyReLU(0.02, inplace = True)
        elif nlactv == 'soft_plus':
            self.nlactv = nn.Softplus()
        else:
            self.nlactv = nn.ReLU(inplace = True)

        self.fc_list = nn.ModuleList()
        self.res_layers = res_layers

        self.all_channels = [in_channels]
        self.all_channels.extend(inter_channels)
        for l in range(0, len(self.all_channels) - 1):
            if l in res_layers:
                if norm == 'weight':
                    self.fc_list.append(nn.Sequential(
                        nn.utils.weight_norm(nn.Conv1d(self.all_channels[l] + self.all_channels[0], self.all_channels[l + 1], 1)),
                        self.nlactv
                    ))
                else:
                    self.fc_list.append(nn.Sequential(
                        nn.Conv1d(self.all_channels[l] + self.all_channels[0], self.all_channels[l + 1], 1),
                        self.nlactv
                    ))
            else:
                if norm == 'weight':
                    self.fc_list.append(nn.Sequential(
                        nn.utils.weight_norm(nn.Conv1d(self.all_channels[l], self.all_channels[l + 1], 1)),
                        self.nlactv
                    ))
                else:
                    self.fc_list.append(nn.Sequential(
                        nn.Conv1d(self.all_channels[l], self.all_channels[l + 1], 1),
                        self.nlactv
                    ))

        self.fc_list.append(nn.Conv1d(self.all_channels[-1], out_channels, 1))
        self.all_channels.append(out_channels)

        if last_op == 'sigmoid':
            self.last_op = nn.Sigmoid()
        elif last_op == 'tanh':
            self.last_op = nn.Tanh()
        else:
            self.last_op = None

    def forward(self, x, return_inter_layer = []):
        tmpx = x
        inter_feat_list = []
        for i, fc in enumerate(self.fc_list):
            if i in self.res_layers:
                x = fc(torch.cat([x, tmpx], dim = 1))
            else:
                x = fc(x)
            if i == len(self.fc_list) - 1 and self.last_op is not None:  # last layer
                x = self.last_op(x)
            if i in return_inter_layer:
                inter_feat_list.append(x.clone())

        if len(return_inter_layer) > 0:
            return x, inter_feat_list
        else:
            return x


class OffsetDecoder(nn.Module):
    """
    Same architecture with ShapeDecoder in POP (https://github.com/qianlim/POP).
    """
    def __init__(self, in_size, hsize = 256, actv_fn='softplus'):
        self.hsize = hsize
        super(OffsetDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_size, self.hsize, 1)
        self.conv2 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv3 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv4 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv5 = torch.nn.Conv1d(self.hsize+in_size, self.hsize, 1)
        self.conv6 = torch.nn.Conv1d(self.hsize, self.hsize, 1)
        self.conv7 = torch.nn.Conv1d(self.hsize, self.hsize, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.hsize)
        self.bn2 = torch.nn.BatchNorm1d(self.hsize)
        self.bn3 = torch.nn.BatchNorm1d(self.hsize)
        self.bn4 = torch.nn.BatchNorm1d(self.hsize)

        self.bn5 = torch.nn.BatchNorm1d(self.hsize)
        self.bn6 = torch.nn.BatchNorm1d(self.hsize)
        self.bn7 = torch.nn.BatchNorm1d(self.hsize)

        self.actv_fn = nn.ReLU() if actv_fn=='relu' else nn.Softplus()

    def forward(self, x):
        x1 = self.actv_fn(self.bn1(self.conv1(x)))
        x2 = self.actv_fn(self.bn2(self.conv2(x1)))
        x3 = self.actv_fn(self.bn3(self.conv3(x2)))
        x4 = self.actv_fn(self.bn4(self.conv4(x3)))
        x5 = self.actv_fn(self.bn5(self.conv5(torch.cat([x,x4],dim=1))))

        # position pred
        x6 = self.actv_fn(self.bn6(self.conv6(x5)))
        x7 = self.actv_fn(self.bn7(self.conv7(x6)))

        return x7
