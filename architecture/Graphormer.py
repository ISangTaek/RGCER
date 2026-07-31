import torch
import torch.nn as nn
from architecture.abstract_arch import AbsArchitecture


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, reshape_size):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, reshape_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias = None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [batch, head, len, dim]
        k = k.transpose(1, 2).transpose(2, 3)  # [batch, head, dim, len]
        v = v.transpose(1, 2)  # [batch, head, len, dim]

        # Scaled Dot-Product Attention
        q = q * self.scale
        A = torch.matmul(q, k)  # [batch, head, len, len]

        # attn_bias
        if attn_bias is not None:
            A = A + attn_bias

        A = torch.softmax(A, dim = 3)
        A = self.att_dropout(A)
        out = torch.matmul(A, v)  # [batch, head, len, dim]
        out = out.transpose(1, 2).contiguous()  # [batch, len, head, dim]
        out = out.view(batch_size, -1, self.num_heads * d_v)
        out = self.output_layer(out)
        assert out.size() == orig_q_size
        return out


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, ffn_size, reshape_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate, reshape_size)

    def forward(self, x, attn_bias = None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        x = x + y
        return x


class AtomsEmbedder(nn.Module):
    def __init__(self,
                 n_layers,
                 num_heads,
                 hidden_dim,
                 dropout_rate,
                 input_dropout_rate,
                 ffn_dim,
                 reshape_dim,
                 attention_drop_rate,
                 readout_dim,
                 ):
        super().__init__()
        self.num_heads = num_heads
        # Hyperparameters for embedding
        self.atom_encoder = nn.Embedding(512, hidden_dim, padding_idx = 0)
        self.spatial_pos_encoder = nn.Embedding(512, num_heads, padding_idx = 0)
        self.in_degree_encoder = nn.Embedding(32, hidden_dim, padding_idx = 0)
        self.out_degree_encoder = nn.Embedding(32, hidden_dim, padding_idx = 0)

        self.input_dropout = nn.Dropout(input_dropout_rate)
        blocks = [AttentionBlock(hidden_dim, ffn_dim, reshape_dim, dropout_rate, attention_drop_rate, num_heads)
                    for _ in
                    range(n_layers)]
        self.layers = nn.ModuleList(blocks)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_vitural_distance = nn.Embedding(1, num_heads)  # [1, num_heads]
        self.readout_layer = nn.Linear(hidden_dim, readout_dim)
        self.gelu = nn.GELU()

    def forward(self, batched_data, perturb = None):
        attn_bias = batched_data.attn_bias  # [n_graph,n_node+1, n_node+1]
        spatial_pos = batched_data.spatial_pos  # [n_graph, n_node, n_node]
        x = batched_data.x  # [n_graph, n_node, n_node_features]
        in_degree = batched_data.in_degree  # [n_graph, n_node]
        out_degree = batched_data.out_degree  # [n_graph, n_node]

        # add the VNode for readout
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()  # [n_graph, n_node+1, n_node+1]
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1,
                                                              1)  # [n_graph, n_head, n_node+1, n_node+1], '1' is the VNode

        # spatial pos [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # all nodes has lined to the VNode, the shortest dis between node and VNode is 1
        t = self.graph_token_vitural_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)

        # node feature + graph token
        # x[n_graph, node, feature, hidden_dim] - > [n_graph, n_node, hidden_dim]
        node_feature = self.atom_encoder(x).sum(dim = -2)
        if perturb is not None:
            node_feature = node_feature + perturb
        # according to in_degree and out_degree, add embedding
        node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

        # VNode feature----->graph_token[1, hidden_dim]->[n_graph, 1, hidden_dim]
        # graph_node_feature [n_graph, n_node+1, hidden_dim]
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim = 1)

        # Transformer
        out = self.input_dropout(graph_node_feature)
        for encoder_layer in self.layers:
            out = encoder_layer(out, graph_attn_bias)
        out = self.final_ln(out)

        # output[n_graph, n_node+1, feature]
        # the last layer of the VNode
        readout = self.readout_layer(out[:, 0, :])

        return readout



class Encoder(nn.Module):
    def __init__(self, atoms_num_heads, task_name, atoms_embedders_num, atoms_hidden_dim, atoms_dropout_rate,
                 atoms_input_dropout_rate, atoms_ffn_dim, atoms_reshape_dim, atoms_attention_dropout_rate,
                 atoms_readout_dim, device):
        super().__init__()
        # atom-level embedding
        self.atoms_emb = AtomsEmbedder(n_layers = atoms_embedders_num, num_heads = atoms_num_heads,
                                       hidden_dim = atoms_hidden_dim,
                                       dropout_rate = atoms_dropout_rate,
                                       input_dropout_rate = atoms_input_dropout_rate,
                                       ffn_dim = atoms_ffn_dim, reshape_dim = atoms_reshape_dim,
                                       attention_drop_rate = atoms_attention_dropout_rate,
                                       readout_dim = atoms_readout_dim)

        # stack all the task tensor to one tensor
        self.stack_task = []
        self.task_name = task_name
        self.device = device
        # transform the task_prompt queries, [bs, 100] --> [bs, 96]
        # self.task_prompt_transform = nn.Linear(100, 96)
        # molecules-level self-attention

    def forward(self, input, task_name, mode):
        atoms_embedder_out = self.atoms_emb(input)  # readout [bs, 96]
        moles_embedder_out = atoms_embedder_out

        if mode == 'train':
            # search the index of the task
            task_index = self.task_name.index(task_name)

            # Aggregate each task data in one tensor
            self.stack_task.append(moles_embedder_out)
            if task_index == len(self.task_name) - 1:
                tasks_tensor = torch.stack(self.stack_task).to(self.device) # [task, bs, 96]

                out = tasks_tensor

                self.stack_task = []
                return out
            else:
                return None
        else:
            # test / val mode
            tasks_tensor = moles_embedder_out.unsqueeze(0)

            out = tasks_tensor
            return out


class Graphormer(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, device, args, **kwargs):
        # task_name: names of all tasks
        super(Graphormer, self).__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.encoder = encoder_class(atoms_num_heads = args.a_heads, task_name = task_name,
                                     atoms_embedders_num = args.a_layers,
                                     atoms_hidden_dim = args.hidden_dim, atoms_dropout_rate = 0.1,
                                     atoms_input_dropout_rate = 0,
                                     atoms_ffn_dim = args.mid_dim, atoms_reshape_dim = args.hidden_dim,
                                     atoms_attention_dropout_rate = 0.1,
                                     atoms_readout_dim = args.hidden_dim, device = device)
    def forward(self, inputs, task_name = None, mode = None):
        # task_name: the name of the task in processing now
        rep = self.encoder(inputs, task_name, mode)  #rep [task, bs, 256]
        if torch.is_tensor(rep):
            if mode == 'train':
                out = {self.task_name[i]: self.decoders[self.task_name[i]](rep[i]) for i in
                   range(len(self.task_name))}

                return out
            else:
                out = {task_name: self.decoders[task_name](rep[0])}

                return out
        else:
            return None, None
