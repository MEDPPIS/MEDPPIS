import pickle
import math
import dgl
import os
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.nn import init
from torch import nn, sum
from torch.nn.parameter import Parameter
from torch_geometric.utils import softmax
from typing import Optional
import torch_geometric.nn as gnn
from torch_geometric.utils import degree
import scipy.sparse as sp
from typing import Tuple, Optional, List
from torch.nn import init
import warnings


warnings.filterwarnings("ignore")
# Feature Path
Feature_Path = "./Feature/"
# Seed
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# model parameters
ADD_NODEFEATS = 'all'  # all/atom_feats/psepose_embedding/no
USE_EFEATS = True  # True/False
MAP_CUTOFF = 14
DIST_NORM = 15

# INPUT_DIM
if ADD_NODEFEATS == 'all':  # add atom features and psepose embedding
    INPUT_DIM = 54 + 7 + 1
elif ADD_NODEFEATS == 'atom_feats':  # only add atom features
    INPUT_DIM = 54 + 7
elif ADD_NODEFEATS == 'psepose_embedding':  # only add psepose embedding
    INPUT_DIM = 54 + 1
elif ADD_NODEFEATS == 'no':
    INPUT_DIM = 54
HIDDEN_DIM = 256  # hidden size of node features
LAYER = 3  # the number of MGU layers
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5

LEARNING_RATE = 1E-3
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2  # [not bind, bind]
NUMBER_EPOCHS = 70

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embedding(sequence_name):
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def cal_edges(sequence_name, radius=MAP_CUTOFF):  # to get the index of the edges
    dist_matrix = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(int)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list


def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix, pos = map(list, zip(*samples))
    label = torch.Tensor(label)
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix)
    pos = torch.cat(pos)
    pos = torch.Tensor(pos)
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix, pos


class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM,
                 psepos_path='./Feature/psepos/Train335_psepos_SC.pkl'):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        nodes_num = len(sequence)
        pos = self.residue_psepos[sequence_name]
        pos = torch.from_numpy(pos).type(torch.FloatTensor)
        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features], axis=1)
        node_features = torch.from_numpy(node_features)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'atom_feats':
            res_atom_features = get_res_atom_features(sequence_name)
            res_atom_features = torch.from_numpy(res_atom_features)
            node_features = torch.cat([node_features, res_atom_features], dim=-1)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'psepose_embedding':
            node_features = torch.cat([node_features, torch.sqrt(
                torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        radius_index_list = cal_edges(sequence_name, MAP_CUTOFF)
        edge_feat = self.cal_edge_attr(radius_index_list, pos)

        G = dgl.DGLGraph()
        G.add_nodes(nodes_num)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)

        self.add_edges_custom(G,
                              radius_index_list,
                              edge_feat
                              )
        adj_matrix = load_graph(sequence_name)
        node_features = node_features.detach().numpy()
        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

        return sequence_name, sequence, label, node_features, G, adj_matrix, pos

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)
        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list
    def add_edges_custom(self, G, radius_index_list, edge_features):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)


class GNNLayer(nn.Module):
    def __init__(self, nfeats_in_dim, nfeats_out_dim, edge_dim=2, use_efeats=True, K=3):
        super(GNNLayer, self).__init__()
        self.use_efeats = use_efeats
        self.K = K

        self.fc = nn.Linear(nfeats_in_dim, nfeats_out_dim, bias=False)
        if self.use_efeats:
            # 增加 K 阶矩维度
            self.attn_fc = nn.Linear(2 * nfeats_out_dim * K + edge_dim, 1, bias=False)
            self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=False)
            self.fc_eFeatsDim_to_nFeatsDim = nn.Linear(edge_dim, nfeats_out_dim, bias=False)
        else:
            self.attn_fc = nn.Linear(2 * nfeats_out_dim * K, 1, bias=False)

        # 矩变换矩阵，注意这里应该是 K - 1 个变换，因为一阶矩已在 moments 中
        self.moment_transforms = nn.ModuleList([
            nn.Linear(nfeats_in_dim, nfeats_out_dim, bias=False)
            for _ in range(self.K - 1)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if self.use_efeats:
            nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_eFeatsDim_to_nFeatsDim.weight, gain=gain)

        for transform in self.moment_transforms:
            nn.init.xavier_normal_(transform.weight, gain=gain)

    def calc_moments(self, z):
        # z形状: [num_edges, feature_dim]
        moments = [z]  # 一阶矩
        if self.K > 1:
            mu = z.mean(dim=0, keepdim=True)  # 计算均值，改为dim=0
            for k in range(2, self.K + 1):
                # 计算k阶中心矩
                centered = (z - mu)
                moment_k = torch.mean(torch.pow(centered, k), dim=0)  # 改为dim=0

                # 标准化
                sign = torch.sign(moment_k)
                moment_k = sign * torch.abs(moment_k).pow(1.0 / k)

                # 扩展维度以匹配batch size
                moment_k = moment_k.unsqueeze(0).expand(z.size(0), -1)
                moment_k = self.moment_transforms[k - 2](moment_k)
                moments.append(moment_k)

        return moments

    def edge_attention(self, edges):
        # 获取源节点和目标节点特征
        h_src = edges.src['z']  # 形状：[num_edges, feature_dim]
        h_dst = edges.dst['z']  # 形状：[num_edges, feature_dim]

        # 对每个边分别计算多阶矩
        src_moments = self.calc_moments(h_src)
        dst_moments = self.calc_moments(h_dst)

        # 拼接所有阶矩和边特征
        if self.use_efeats:
            z2 = torch.cat([*src_moments, *dst_moments, edges.data['ex']], dim=1)
            a = self.attn_fc(z2)
            ez = self.fc_eFeatsDim_to_nFeatsDim(edges.data['ex'])
            return {'e': F.leaky_relu(a), 'ez': ez}
        else:
            z2 = torch.cat([*src_moments, *dst_moments], dim=1)
            a = self.attn_fc(z2)
            return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        if self.use_efeats:
            return {'z': edges.src['z'], 'e': edges.data['e'], 'ez': edges.data['ez']}
        else:
            return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        attn_w = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(attn_w * nodes.mailbox['z'], dim=1)
        if self.use_efeats:
            h = h + torch.sum(attn_w * nodes.mailbox['ez'], dim=1)
        return {'h': h}

    def forward(self, g, h, e=None):
        z = self.fc(h)
        g.ndata['z'] = z
        if self.use_efeats and e is not None:
            ex = self.fc_edge_for_att_calc(e)
            g.edata['ex'] = ex
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')



class model_MED(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha):
        super(model_MED, self).__init__()
        self.layer1 = nlayers
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.mLSTM = mLSTM(256, 256, 1)

        # 定义特征维度列表
        self.dim_list = [256, 128, 64, 32]

        # 定义依赖于特征维度的层，使用 ModuleDict
        self.attn_norm_dict = nn.ModuleDict({
            str(dim): nn.LayerNorm(dim, eps=1e-6) for dim in self.dim_list
        })
        self.mLSTM = nn.ModuleDict({
            str(dim): mLSTM(dim, dim, 1) for dim in self.dim_list
        })
        self.embedding_lap_pos_enc_dict = nn.ModuleDict({
            str(dim): nn.Linear(8, dim) for dim in self.dim_list
        })

        self.fc_one_dimension_dict = nn.ModuleDict({
            str(dim): nn.Linear(dim, 1) for dim in self.dim_list
        })

        # 为不同维度的 MuliHeadAttention_one_dimension 定义实例
        self.MuliHeadAttention_one_dimension_dict = nn.ModuleDict({
            str(dim): MuliHeadAttention_one_dimension(hidden_size=dim, dropout_rate=0.1) for dim in self.dim_list
        })

        # 定义不同维度的 MultiAttention
        self.MultiAttention_dict = nn.ModuleDict({
            str(dim): MuliHeadAttention(dim, dropout, head_size=8) for dim in self.dim_list
        })

        # 定义不同维度的 GNNLayer
        self.GNN_dict = nn.ModuleDict({
            str(dim): GNNLayer(dim,dim) for dim in self.dim_list
        })

        # 定义不同维度的 GEANet
        self.GEANet_dict = nn.ModuleDict({
            str(dim): GEANet(dim, GEANetConfig(
                n_heads=8,
                shared_unit=True,
                edge_unit=True,
                unit_size=dim // 8  # 根据需要调整 unit_size
            )) for dim in self.dim_list
        })

        # 定义不同维度的归一化层
        self.gcn_norm_dict = nn.ModuleDict({
            str(dim): nn.LayerNorm(dim, eps=1e-6) for dim in self.dim_list
        })

        self.ffn_norm_dict = nn.ModuleDict({
            str(dim): nn.LayerNorm(dim, eps=1e-6) for dim in self.dim_list
        })

        # 为不同维度定义 edge 相关的线性层
        self.fc_edge_1_dict = nn.ModuleDict({
            str(dim): nn.Linear(2, dim) for dim in self.dim_list
        })
        self.fc_edge_2_dict = nn.ModuleDict({
            str(dim): nn.Linear(dim, 2) for dim in self.dim_list
        })

        # 定义其他层
        self.act_fn = nn.ReLU()
        self.relu = nn.ReLU()
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self.fcs_1 = nn.Linear(nhidden, nhidden // 2)
        self.fcs_2 = nn.Linear(nhidden // 2, nhidden // 4)
        self.fcs_3 = nn.Linear(nhidden // 4, nhidden // 8)

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden // 4, nclass))


    def generate_mask(self, adjacency_matrix):
        return (adjacency_matrix == 0)

    def create_edge_index(self, adjacency_matrix):
        edge_index = torch.nonzero(adjacency_matrix).t().contiguous()
        return edge_index.to(adjacency_matrix.device)

    def rebuild_adj_matrix(self, edge_index, num_nodes):
        device = edge_index.device
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        return adj_matrix

    def laplacian_positional_encoding(self, adj_matrix, pos_enc_dim=8):
        adj_matrix = adj_matrix.to(torch.float32)

        # 计算度矩阵
        degrees = adj_matrix.sum(dim=1).cpu().numpy().flatten()

        # 计算度矩阵的逆平方根，避免除以零
        degrees_inv_sqrt = np.power(degrees, -0.5, where=degrees!=0)
        D_inv_sqrt = sp.diags(degrees_inv_sqrt, 0)

        # 将邻接矩阵转换为 CPU 上的 NumPy 数组
        adj_matrix_np = adj_matrix.cpu().numpy()

        # 构建归一化的拉普拉斯矩阵
        L = np.eye(adj_matrix.shape[0]) - D_inv_sqrt @ adj_matrix_np @ D_inv_sqrt

        # 计算拉普拉斯矩阵的特征值和特征向量
        EigVal, EigVec = np.linalg.eigh(L)

        # 对特征值和特征向量进行排序
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], EigVec[:, idx]

        # 如果特征向量数量不足，进行零填充
        if EigVec.shape[1] < pos_enc_dim + 1:
            EigVec = np.pad(EigVec, ((0, 0), (0, pos_enc_dim + 1 - EigVec.shape[1])), mode='constant')

        # 选择特征向量作为位置编码
        pos_enc = EigVec[:, 1:pos_enc_dim + 1]

        # 将结果转换回 PyTorch 张量，并移动到原始设备
        pos_enc = torch.tensor(pos_enc, dtype=torch.float32, device=adj_matrix.device)
        return pos_enc

    def MainModule(self, h, adj, edge_index, laplacian, graph, efeats):
        x = h
        e = efeats
        dim = x.size(-1)
        dim_str = str(dim)

        # 根据当前维度选择对应的层
        attn_norm = self.attn_norm_dict[dim_str]
        embedding_lap_pos_enc = self.embedding_lap_pos_enc_dict[dim_str]
        fc_one_dimension = self.fc_one_dimension_dict[dim_str]
        MuliHeadAttention_one_dimension = self.MuliHeadAttention_one_dimension_dict[dim_str]
        MultiAttention = self.MultiAttention_dict[dim_str]
        GNN = self.GNN_dict[dim_str]
        GEANet = self.GEANet_dict[dim_str]
        gcn_norm = self.gcn_norm_dict[dim_str]
        ffn_norm = self.ffn_norm_dict[dim_str]
        fc_edge_1 = self.fc_edge_1_dict[dim_str]
        fc_edge_2 = self.fc_edge_2_dict[dim_str]

        y = attn_norm(x)
        y = self.relu(y)
        laplacian = embedding_lap_pos_enc(laplacian.float())
        y = y + laplacian

        y1 = y
        y1 = fc_one_dimension(y1)
        y1 = MuliHeadAttention_one_dimension(y1, y1, y1)

        y2 = y
        y2 = y2.transpose(0, 1)
        fc_y = nn.Linear(y2.shape[1], 1).to(device)
        y2 = fc_y(y2)
        y2 = y2.transpose(0, 1)
        y2 = MuliHeadAttention_one_dimension(y2, y2, y2)

        y3 = y1 * y2
        y = MultiAttention(y, y, y, adj)
        y = y3 + y
        y = self.attn_dropout(y)
        y = gcn_norm(y)
        y = self.relu(y)
        y = GNN(graph, y, efeats)
        y = self.attn_dropout(y)
        y = ffn_norm(y)
        y = self.relu(y)
        efeats = fc_edge_1(efeats)
        y, efeats = GEANet(y, efeats)
        efeats = fc_edge_2(efeats)
        y = self.ffn_dropout(y)
        y = y + h
        efeats = efeats + e
        return y, efeats

    def forward(self, x, adj_matrix=None, graph=None, efeats=None):
        h = self.act_fn(self.fcs[0](x))
        dim = h.size(-1)
        dim_str = str(dim)
        mLSTM = self.mLSTM[dim_str]
        h = h.unsqueeze(0)
        h = mLSTM(h)[0]
        h = h.squeeze(0)
        edge_index = self.create_edge_index(adj_matrix)
        laplacian = self.laplacian_positional_encoding(adj_matrix)
        for i in range(self.layer1):
            h, efeats = self.MainModule(h, adj_matrix, edge_index, laplacian, graph, efeats)

        h = self.act_fn(self.fcs_1(h))

        for i in range(self.layer1):
            h, efeats = self.MainModule(h, adj_matrix, edge_index, laplacian, graph, efeats)

        h = self.act_fn(self.fcs_2(h))

        for i in range(self.layer1):
            h, efeats = self.MainModule(h, adj_matrix, edge_index, laplacian, graph, efeats)

        h = F.dropout(h, self.dropout, training=self.training).to(device)
        output = self.fcs[-1](h)
        return output

class MEDPPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha):
        super(MEDPPIS, self).__init__()
        self.model_MED = model_MED(nlayers=nlayers, nfeat=nfeat, nhidden=nhidden, nclass=nclass,dropout=dropout, lamda=lamda, alpha=alpha)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=10,min_lr=1e-6)
    def forward(self, x, graph,adj_matrix,pos):
        x = x.float().to(device)
        x = x.view([x.shape[0] * x.shape[1], x.shape[2]])
        output = self.model_MED(x=x,adj_matrix=adj_matrix,graph=graph, efeats=graph.edata['ex'])
        return output

class MuliHeadAttention_one_dimension(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=6):
        super(MuliHeadAttention_one_dimension, self).__init__()
        self.head_size = head_size
        self.attn_size = attn_size = hidden_size // head_size
        self.scale = attn_size ** -0.5

        self.linear_q = nn.Linear(1, head_size * attn_size, bias=False)
        self.linear_k = nn.Linear(1, head_size * attn_size, bias=False)
        self.linear_v = nn.Linear(1, head_size * attn_size, bias=False)

        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.attn_dropout = nn.Dropout(dropout_rate)
        # 修改output_layer，输出维度为1
        self.output_layer = nn.Linear(head_size * attn_size, 1, bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v):
        # 保存输入的原始shape以便后续恢复
        original_shape = q.shape

        # 处理输入维度
        if q.dim() == 1:
            q = q.unsqueeze(0).unsqueeze(-1)
        elif q.dim() == 2:
            if q.size(1) == 1:  # m*1的情况
                q = q.unsqueeze(0)
            else:  # 1*n的情况
                q = q.transpose(0, 1).unsqueeze(0)

        # 对k和v进行相同的处理
        if k.dim() == 1:
            k = k.unsqueeze(0).unsqueeze(-1)
        elif k.dim() == 2:
            if k.size(1) == 1:
                k = k.unsqueeze(0)
            else:
                k = k.transpose(0, 1).unsqueeze(0)

        if v.dim() == 1:
            v = v.unsqueeze(0).unsqueeze(-1)
        elif v.dim() == 2:
            if v.size(1) == 1:
                v = v.unsqueeze(0)
            else:
                v = v.transpose(0, 1).unsqueeze(0)

        # 获取维度
        batch_size, seq_length, _ = q.size()

        # 线性变换和重塑
        q = self.linear_q(q).view(batch_size, seq_length, self.head_size, self.attn_size).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, seq_length, self.head_size, self.attn_size).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, seq_length, self.head_size, self.attn_size).transpose(1, 2)

        # 缩放点积注意力
        q = q * self.scale
        x = torch.matmul(q, k.transpose(-2, -1))
        x = torch.softmax(x, dim=-1)

        # dropout和注意力输出
        x = self.attn_dropout(x)
        x = torch.matmul(x, v)

        # 重塑和输出
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.head_size * self.attn_size)
        output = self.output_layer(x)

        # 恢复原始维度
        if len(original_shape) == 1:
            output = output.squeeze()
        else:
            if original_shape[0] == 1:  # 1*n的情况
                output = output.squeeze(0).transpose(0, 1)
            else:  # m*1的情况
                output = output.squeeze(0)

        return output
class MuliHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=6):
        super(MuliHeadAttention, self).__init__()
        self.head_size = head_size
        self.attn_size = attn_size = hidden_size // head_size
        self.scale = attn_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * attn_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * attn_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * attn_size, bias=False)

        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.attn_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * attn_size, hidden_size, bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, adj_matrix):

        if q.dim() == 2:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        batch_size, seq_length, _ = q.size()



        if adj_matrix.dim() == 2:
            adj_matrix = adj_matrix.unsqueeze(0)


        adj_matrix = adj_matrix.unsqueeze(1).expand(batch_size, self.head_size, seq_length, seq_length)


        q = self.linear_q(q).view(batch_size, seq_length, self.head_size, self.attn_size).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, seq_length, self.head_size, self.attn_size).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, seq_length, self.head_size, self.attn_size).transpose(1, 2)


        q = q * self.scale

        x = torch.matmul(q, k.transpose(-2, -1))
        x = torch.mul(adj_matrix, x)

        x = torch.softmax(x, dim=-1)

        x = self.attn_dropout(x)
        x = torch.matmul(x, v)
        # 重塑和输出
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.head_size * self.attn_size)
        output = self.output_layer(x)


        if q.size(0) == 1:
            output = output.squeeze(0)

        return output


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, fliter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, fliter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(fliter_size, hidden_size)
        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
def external_norm(attn):
    softmax = nn.Softmax(dim=0)  # N
    attn = softmax(attn)  # bs,n,S
    attn = attn /sum(attn, dim=2, keepdim=True)  # bs,n,S
    return attn


class DNorm(nn.Module):
    def __init__(
            self,
            dim1=0, dim2=2
    ):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.softmax = nn.Softmax(dim=self.dim1)

    def forward(self, attn: Tensor) -> Tensor:
        attn = self.softmax(attn)
        attn = attn / sum(attn, dim=self.dim2, keepdim=True)
        return attn


class GEANet(nn.Module):

    def __init__(
            self, dim, GEANet_cfg):
        super().__init__()


        self.dim = dim
        self.external_num_heads = GEANet_cfg.n_heads
        self.use_shared_unit = GEANet_cfg.shared_unit
        self.use_edge_unit = GEANet_cfg.edge_unit
        self.unit_size = GEANet_cfg.unit_size

        # self.q_Linear = nn.Linear(in_dim, gconv_dim - dim_pe)
        self.node_U1 = nn.Linear(self.unit_size, self.unit_size)
        self.node_U2 = nn.Linear(self.unit_size, self.unit_size)

        assert self.unit_size * self.external_num_heads == self.dim, "dim must be divisible by external_num_heads"

        # nn.init.xavier_normal_(self.node_m1.weight, gain=1)
        # nn.init.xavier_normal_(self.node_m2.weight, gain=1)
        if  self.use_edge_unit:
            self.edge_U1 = nn.Linear(self.unit_size, self.unit_size)
            self.edge_U2 = nn.Linear(self.unit_size, self.unit_size)
            if self.use_shared_unit:
                self.share_U = nn.Linear(dim, dim)

            # nn.init.xavier_normal_(self.edge_m1.weight, gain=1)
            # nn.init.xavier_normal_(self.edge_m2.weight, gain=1)
            # nn.init.xavier_normal_(self.share_m.weight, gain=1)
        self.norm = DNorm()

        # self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, node_x ,edge_attr = None) -> Tensor:
        if self.use_shared_unit:
            node_x = self.share_U(node_x)
            edge_attr = self.share_U(edge_attr)
        # x : N x 64
        # External attention
        N, d, head = node_x.size()[0], node_x.size()[1], self.external_num_heads
        node_out = node_x.reshape(N, head ,-1)  # Q * 4（head）  ：  N x 16 x 4(head)
        # node_out = node_out.transpose(1, 2)  # (N, 16, 4) -> (N, 4, 16)
        node_out = self.node_U1(node_out)
        attn = self.norm(node_out)  # 行列归一化  N x 16 x 4
        node_out = self.node_U2(attn)
        node_out = node_out.reshape(N, -1)

        if self.use_edge_unit:

            N, d, head = edge_attr.size()[0], edge_attr.size()[1], self.external_num_heads
            edge_out = edge_attr.reshape(N, -1, head)  # Q * 4（head）  ：  N x 16 x 4(head)
            edge_out = edge_out.transpose(1, 2)  # (N, 16, 4) -> (N, 4, 16)
            edge_out = self.edge_U1(edge_out)
            attn = self.norm(edge_out)  # 行列归一化  N x 16 x 4
            edge_out = self.edge_U2(attn)
            edge_out = edge_out.reshape(N, -1)
        else:
            edge_out = edge_attr

        return node_out ,edge_out
class GEANetConfig:
    def __init__(self, n_heads, shared_unit, edge_unit, unit_size):
        self.n_heads = n_heads
        self.shared_unit = shared_unit
        self.edge_unit = edge_unit
        self.unit_size = unit_size

class mLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Initialize weights and biases
        self.W_i = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_f = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_o = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_q = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_k = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_v = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )

        if self.bias:
            self.B_i = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_f = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_o = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_q = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_k = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_v = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(
        self,
        x: torch.Tensor,
        internal_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Get the internal state
        C, n, m = internal_state

        #  Calculate the input, forget, output, query, key and value gates
        i_tilda = (
            torch.matmul(x, self.W_i) + self.B_i
            if self.bias
            else torch.matmul(x, self.W_i)
        )
        f_tilda = (
            torch.matmul(x, self.W_f) + self.B_f
            if self.bias
            else torch.matmul(x, self.W_f)
        )
        o_tilda = (
            torch.matmul(x, self.W_o) + self.B_o
            if self.bias
            else torch.matmul(x, self.W_o)
        )
        q_t = (
            torch.matmul(x, self.W_q) + self.B_q
            if self.bias
            else torch.matmul(x, self.W_q)
        )
        k_t = (
            torch.matmul(x, self.W_k) / torch.sqrt(torch.tensor(self.hidden_size))
            + self.B_k
            if self.bias
            else torch.matmul(x, self.W_k) / torch.sqrt(torch.tensor(self.hidden_size))
        )
        v_t = (
            torch.matmul(x, self.W_v) + self.B_v
            if self.bias
            else torch.matmul(x, self.W_v)
        )

        # Exponential activation of the input gate
        i_t = torch.exp(i_tilda)
        f_t = torch.sigmoid(f_tilda)
        o_t = torch.sigmoid(o_tilda)

        # Stabilization state
        m_t = torch.max(torch.log(f_t) + m, torch.log(i_t))
        i_prime = torch.exp(i_tilda - m_t)

        # Covarieance matrix and normalization state
        C_t = f_t.unsqueeze(-1) * C + i_prime.unsqueeze(-1) * torch.einsum(
            "bi, bk -> bik", v_t, k_t
        )
        n_t = f_t * n + i_prime * k_t

        normalize_inner = torch.diagonal(torch.matmul(n_t, q_t.T))
        divisor = torch.max(
            torch.abs(normalize_inner), torch.ones_like(normalize_inner)
        )
        h_tilda = torch.einsum("bkj,bj -> bk", C_t, q_t) / divisor.view(-1, 1)
        h_t = o_t * h_tilda

        return h_t, (C_t, n_t, m_t)

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
        )


class mLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            [
                mLSTMCell(input_size if layer == 0 else hidden_size, hidden_size, bias)
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Permute the input tensor if batch_first is True
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if hidden_states is None:
            hidden_states = self.init_hidden(x.size(1), device=x.device, dtype=x.dtype)
        else:
            # Check if the hidden states are of the correct length
            if len(hidden_states) != self.num_layers:
                raise ValueError(
                    f"Expected hidden states of length {self.num_layers}, but got {len(hidden_states)}"
                )
            if any(state[0].size(0) != x.size(1) for state in hidden_states):
                raise ValueError(
                    f"Expected hidden states of batch size {x.size(1)}, but got {hidden_states[0][0].size(0)}"
                )

        H, C, N, M = [], [], [], []

        for layer, cell in enumerate(self.cells):
            lh, lc, ln, lm = [], [], [], []
            for t in range(x.size(0)):
                h_t, hidden_states[layer] = (
                    cell(x[t], hidden_states[layer])
                    if layer == 0
                    else cell(H[layer - 1][t], hidden_states[layer])
                )
                lh.append(h_t)
                lc.append(hidden_states[layer][0])
                ln.append(hidden_states[layer][1])
                lm.append(hidden_states[layer][2])

            H.append(torch.stack(lh, dim=0))
            C.append(torch.stack(lc, dim=0))
            N.append(torch.stack(ln, dim=0))
            M.append(torch.stack(lm, dim=0))

        H = torch.stack(H, dim=0)
        C = torch.stack(C, dim=0)
        N = torch.stack(N, dim=0)
        M = torch.stack(M, dim=0)

        return H[-1], (H, C, N, M)

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return [cell.init_hidden(batch_size, **kwargs) for cell in self.cells]
