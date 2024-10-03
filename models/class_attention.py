import torch.nn.functional as F
from models.loftr_module.linear_attention import LinearAttention, FullAttention
import copy
import torch
import torch.nn as nn

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class ClassTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(ClassTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model_class']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model_class'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ClassNum, feat, feat_class, mask0=None):

        feat_c = feat.clone().detach()
        feat_classk = feat_class.clone().detach()
        batch_size = feat.shape[0]  # 获取批次大小
        feat_list = []
        for i in range(batch_size):
            feat_i = feat_c[i:i + 1]  # 选择当前批次的第 i 个张量
            feat_classi = feat_classk[i:i + 1]
            if mask0 is None:
                mask_c0 = None
            else:
                mask_c0 = mask0[i:i + 1]

            feat_classi = feat_classi.squeeze(0)  # Shape: (10/20, 256)
            feat_i = feat_i.squeeze(0)  # Shape: (128, 256)
            # Compute cosine similarity between feat_c_8_class10 and feat_c_8_ab
            cos_similarity_matrix = F.cosine_similarity(feat_classi.unsqueeze(1), feat_i.unsqueeze(0) , dim=2)  # Shape: (10, 128)
            # 存储十/二十组索引列表
            ten_lists = [[] for _ in range(ClassNum)]  # 存储100组索引列表
            for i in range(cos_similarity_matrix.shape[1]):  # 遍历相似度矩阵的每一列
                col = cos_similarity_matrix[:, i]  # 获取一列相似度值
                top_index = torch.argmax(col)  # 获取最大值的索引
                ten_lists[top_index].append(i)  # 将索引加入到对应的组中

            feat_i = feat_i.unsqueeze(0)
            assert self.d_model == feat_i.size(2), "the feature number of src and transformer must be equal"

            # 对于每一层的处理，这里只做 self-attention
            selected_features = {}
            for idx, indices in enumerate(ten_lists, start=1):
                feat_selected = None
                for layer, name in zip(self.layers, self.layer_names):
                    if name == 'self':
                        if indices is not None:
                            feat_selected = feat_i[:, indices, :]  # 选择部分向量
                            feat_selected = layer(feat_selected, feat_selected)
                        else:
                            feat_i = layer(feat_i, feat_i)
                    else:
                        raise KeyError

                selected_features[f"feat_selected_{idx}"] = feat_selected

            for idx, indices in enumerate(ten_lists, start=1):
                feat_selected = selected_features[f"feat_selected_{idx}"]
                feat_i[:, indices, :] = feat_selected

            feat_list.append(feat_i)  # 将处理后的结果添加到列表中

        feat = torch.cat(feat_list, dim=0)
        return feat



