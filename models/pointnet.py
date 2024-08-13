import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class STNkd(nn.Module):

    def __init__(self, k=3):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        # print("stk input", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print("stk conv1", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print("stk conv2", x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print("stk conv3", x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        # print("stk max", x.shape)
        x = x.view(-1, 1024)
        # print("stk view", x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        # print("stk fc1", x.shape)
        x = F.relu(self.bn5(self.fc2(x)))
        # print("stk fc2", x.shape)
        x = self.fc3(x)
        # print("stk fc3", x.shape)

        iden = torch.eye(self.k, dtype=torch.float32).flatten().view(
            1, self.k * self.k).repeat(batchsize, 1)
        # print("stk iden", iden.shape)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        # print("stk iden+x", x.shape)
        x = x.view(-1, self.k, self.k)
        # print("stk out", x.shape)
        return x


class PointNetfeat(nn.Module):

    def __init__(self, feature_len, extra_feature_len=32):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=feature_len)
        # self.fstn = STNkd(k=32 + extra_feature_len)
        self.conv1 = torch.nn.Conv1d(feature_len, 64, 1)
        self.conv2 = torch.nn.Conv1d(64 + extra_feature_len, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # print("point net x: ", x.shape)
        # print("stn input: ", x[:, :3].shape)
        # trans pc only and layer 1
        trans = self.stn(x[:, :3])
        x_p = torch.bmm(x[:, :3].transpose(2, 1), trans)
        x_p = self.conv1(x_p.transpose(2, 1))
        x_p = F.relu(self.bn1(x_p))
        # print("point net trans, x_p: ", trans.shape, x_p.shape)
        # concat rgbd features
        x = torch.cat([x_p, x[:, 3:]], 1)
        # print("point net cat: ", x.shape)
        # feature trans and layer 2
        # trans = self.fstn(x)
        # x = torch.bmm(x.transpose(2, 1), trans)
        # x = x.transpose(2, 1)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        # feature layer 3
        # print("point net conv2: ", x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        # print("point net conv3: ", x.shape)
        x = torch.max(x, 2, keepdim=True)[0]
        # print("after maxpool: ", x.shape)
        x = x.view(-1, 1024)
        # print("point net: ", x.shape)
        return x
