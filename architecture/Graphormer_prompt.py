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
        self.attention_map = None

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
        self.attention_map = A.detach()
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

        # attn_bias = batched_data['attn_bias']
        # spatial_pos = batched_data['spatial_pos']
        # x = batched_data['x']
        # in_degree = batched_data['in_degree']
        # out_degree = batched_data['out_degree']

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
class Task_Prompt_block(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate = 0.1):
        super().__init__()
        self.cross_attention = MultiHeadAttention(hidden_dim, dropout_rate, num_heads)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_size = hidden_dim, ffn_size = 128, dropout_rate = dropout_rate,
                                      reshape_size = hidden_dim)
    def forward(self, mol_rep, prompt, bias = None):

        # cross-attention
        q = self.norm_q(prompt)
        kv = self.norm_kv(mol_rep)
        mol_out = self.cross_attention(q, kv, kv, attn_bias = bias)
        mol_out = self.dropout(mol_out)
        mol_residual = mol_out + mol_rep

        mol_out = self.ffn_norm(mol_residual)
        mol_out = self.ffn(mol_out)
        mol_residual = mol_residual + mol_out
        return mol_residual

class Task_prompt_Encoder(nn.Module):
    def __init__(self,moles_hidden_dim, n_layers, num_heads, dropout_rate, num_tasks):
        super().__init__()
        self.task_layers = nn.ModuleList(
            [Task_Prompt_block(moles_hidden_dim, num_heads, dropout_rate) for _ in range(n_layers)]
        )
        # self.final_norm = nn.LayerNorm(moles_hidden_dim)
        # task bias
        self.num_heads = num_heads
        # self.taskBias = nn.Parameter((torch.randn(1, num_tasks, num_tasks) * 0.01))
    def forward(self, mol_rep, prompt):
        B, T, H = mol_rep.size()
        prompt = prompt.expand(B, T, H)  # 训练,测试 [1, T, H] -> [B, T, H]
        # taskBias = self.taskBias.unsqueeze(0).expand(B, self.num_heads, T, T)

        taskBias = None
        for layer in self.task_layers:
            mol_rep = layer(mol_rep, prompt, taskBias)

        # mol_rep = self.final_norm(mol_rep)
        return mol_rep


class Encoder(nn.Module):
    def __init__(self, atoms_num_heads, task_name, atoms_embedders_num, atoms_hidden_dim, atoms_dropout_rate,
                 atoms_input_dropout_rate, atoms_ffn_dim, atoms_reshape_dim, atoms_attention_dropout_rate,
                 atoms_readout_dim, moles_hidden_dim, task_layers, device,task_num_heads, task_dropout_rate):
        super().__init__()
        # atom-level embedding
        self.atoms_emb = AtomsEmbedder(n_layers = atoms_embedders_num, num_heads = atoms_num_heads,
                                       hidden_dim = atoms_hidden_dim,
                                       dropout_rate = atoms_dropout_rate,
                                       input_dropout_rate = atoms_input_dropout_rate,
                                       ffn_dim = atoms_ffn_dim, reshape_dim = atoms_reshape_dim,
                                       attention_drop_rate = atoms_attention_dropout_rate,
                                       readout_dim = atoms_readout_dim)
        # task-level prompting
        self.task_prompt_encoder = Task_prompt_Encoder(moles_hidden_dim = moles_hidden_dim, n_layers = task_layers,
                                                       num_heads = task_num_heads, dropout_rate = task_dropout_rate,
                                                       num_tasks = len(task_name))

        # stack all the task tensor to one tensor
        self.stack_task = []
        self.task_name = task_name
        self.device = device
        # prompt define
        self.prompts = nn.Parameter(torch.empty(1, len(task_name), moles_hidden_dim))
        nn.init.xavier_normal_(self.prompts)
        # EMA
        self.ema_decay = 0.9
        self.register_buffer('ema_prompts', self.prompts.data.clone())
        # # init
        # self.apply(self._init_weights)

    # def _ini t_weights(self, module):
    #     """统一对 nn.Linear 做 Xavier，bias 置 0；对 Embedding 也做小范围均匀初始化。"""
    #     if isinstance(module, nn.Linear):
    #          nn.init.xavier_uniform_(module.weight)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.Embedding):
        #     # （可选）对 embedding 做小范围均匀初始化
        #     nn.init.uniform_(module.weight, -0.1, 0.1)

    @torch.no_grad()
    def _update_ema_prompts(self):
        """
         Update EMA prompts based on current prompts.
        """
        # self.ema_prompts = self.ema_decay * self.ema_prompts + (1 - self.ema_decay) * self.prompts.data
        self.ema_prompts.mul_(self.ema_decay).add_(self.prompts.data, alpha = 1 - self.ema_decay)



    def forward(self, input, task_name, mode):
        atoms_embedder_out = self.atoms_emb(input)  # readout [bs, 96]
        if mode == 'train':
            # search the index of the task
            task_index = self.task_name.index(task_name)

            # Aggregate each task data in one tensor
            self.stack_task.append(atoms_embedder_out)
            if task_index == len(self.task_name) - 1:
                self._update_ema_prompts()
                tasks_tensor = torch.stack(self.stack_task).transpose(0,1).contiguous().to(self.device) # [task, bs, 96]
                prompt = self.prompts
                out = self.task_prompt_encoder(tasks_tensor, prompt).transpose(0, 1).contiguous()

                self.stack_task = []
                return out
            else:
                return None
        else:
            # test / val mode
            B, H = atoms_embedder_out.shape
            T = len(self.task_name)
            mole_tensor = atoms_embedder_out.unsqueeze(1).expand(B, T, H).contiguous()  # [B, T, H]
            # use EMA prompt
            prompt = self.ema_prompts  # [1, T, H]
            # prompt = self.prompts
            out = self.task_prompt_encoder(mole_tensor, prompt).transpose(0, 1).contiguous()
            return out


class Graphormer_prompt(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, device, args, **kwargs):
        # task_name: names of all tasks
        super(Graphormer_prompt, self).__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.encoder = encoder_class(atoms_num_heads = args.a_heads, task_name = task_name,
                                     atoms_embedders_num = args.a_layers,
                                     atoms_hidden_dim = args.hidden_dim, atoms_dropout_rate = 0.1, atoms_input_dropout_rate = 0,
                                     atoms_ffn_dim = args.mid_dim, atoms_reshape_dim = args.hidden_dim, atoms_attention_dropout_rate = 0.1,
                                     atoms_readout_dim = args.hidden_dim, moles_hidden_dim = args.hidden_dim, task_layers = 1,task_num_heads = 1,
                                     task_dropout_rate = 0.1, device = device)

    def forward(self, inputs, task_name = None, mode = None):
        # task_name: the name of the task in processing now
        rep = self.encoder(inputs, task_name, mode)  #rep [task, bs, 256]
        task_index = self.task_name.index(task_name)
        if torch.is_tensor(rep):
            if mode == 'train':
                out = {self.task_name[i]: self.decoders[self.task_name[i]](rep[i]) for i in
                   range(len(self.task_name))}

                return out
            else:
                out = {
                    self.task_name[i]: self.decoders[self.task_name[i]](rep[i])
                    for i in range(len(self.task_name))
                }

                return out
        else:
            return None, None
