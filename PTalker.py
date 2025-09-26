import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hubert import HubertModel
from functools import reduce
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForXVector,  AutoFeatureExtractor
from wav2vec import Wav2Vec2Model

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def GRL(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


def orthogonal_loss(content_features, style_features):
    batch_size = content_features.size(0)
    frame_num_content = content_features.size(1)
    frame_num_style = style_features.size(1)

    if frame_num_content > frame_num_style:
        style_features = F.adaptive_avg_pool1d(style_features.transpose(1, 2), frame_num_content).transpose(1, 2)
    elif frame_num_content < frame_num_style:
        content_features = F.adaptive_avg_pool1d(content_features.transpose(1, 2), frame_num_style).transpose(1, 2)

    dot_product = torch.bmm(content_features.transpose(1, 2), style_features)

    loss = torch.norm(dot_product, p='fro') / batch_size

    return loss

def style_similarity_loss(style_emb_1, style_emb_2, style_emb_3, loss_type='cosine', weight=1.0):
    if loss_type == 'cosine':
        similarity_12 = F.cosine_similarity(style_emb_1, style_emb_2, dim=-1)
        similarity_13 = F.cosine_similarity(style_emb_1, style_emb_3, dim=-1)
        similarity_23 = F.cosine_similarity(style_emb_2, style_emb_3, dim=-1)

        similarity_loss = (1.0 - similarity_12.mean()) + (1.0 - similarity_13.mean()) + (1.0 - similarity_23.mean())

    elif loss_type == 'l2':
        loss_12 = F.mse_loss(style_emb_1, style_emb_2)
        loss_13 = F.mse_loss(style_emb_1, style_emb_3)
        loss_23 = F.mse_loss(style_emb_2, style_emb_3)

        similarity_loss = loss_12 + loss_13 + loss_23

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return weight * similarity_loss


class AuxiliaryStyleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AuxiliaryStyleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, style_code):
        return self.fc(style_code)



class StyleClassifier(nn.Module):
    def __init__(self, feature_dim, num_speakers):
        super(StyleClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_speakers)
    def forward(self, x):
        return self.fc(x)

class ContentContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_weight=0.5, reg_weight=0.001, topk_ratio=0.15):
        super().__init__()
        self.temperature  = nn.Parameter(torch.tensor(temperature))
        self.lambda_weight = lambda_weight
        self.reg_weight = reg_weight
        self.topk_ratio = topk_ratio
        self.topk_history = []

    def forward(self, content_codes, audio_features):

        seq_len = min(content_codes.size(1), audio_features.size(1))
        content = content_codes[:, :seq_len, :]
        audio   = audio_features[:, :seq_len, :]

        top_k = max(1, round(seq_len * self.topk_ratio))
        self.topk_history.append(top_k)

        sim_a2c = self.compute_similarity(audio,   content)
        sim_c2a = self.compute_similarity(content, audio  )

        loss_a2c = self._contrastive(sim_a2c, top_k)
        loss_c2a = self._contrastive(sim_c2a, top_k)
        cont_loss = self.lambda_weight * loss_a2c + (1 - self.lambda_weight) * loss_c2a

        reg = self.reg_weight * (torch.norm(content) + torch.norm(audio))

        return cont_loss + reg

    def compute_similarity(self, query, keys):
        qn = F.normalize(query, p=2, dim=-1)
        kn = F.normalize(keys, p=2, dim=-1)
        return torch.matmul(qn, kn.transpose(-1, -2)) / self.temperature

    def _contrastive(self, sim_mat, top_k, alpha=1.5, beta=0.7):

        batch, L, _ = sim_mat.size()
        loss = 0.0
        for i in range(batch):
            values, _ = torch.topk(sim_mat[i], top_k, dim=-1)  # (L, top_k)
            pos = torch.diagonal(values, dim1=-2, dim2=-1)      # (L,)
            exp_all = torch.exp(values)
            sum_all = exp_all.sum(dim=-1)                       # (L,)
            idx = torch.arange(top_k, device=values.device)
            sum_pos = sum_all.gather(0, idx)                    # (top_k,)
            loss += -alpha * torch.log(torch.exp(pos) / (beta * sum_pos)).mean()
        return loss / batch

    def reset_history(self):
        self.topk_history = []

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalKLDivergence(nn.Module):
    """
    """

    def __init__(self, eps=1e-6):
        super(CrossModalKLDivergence, self).__init__()
        self.eps = eps

    def forward(self, content_features, motion_features):
        seq_len = min(content_features.size(1), motion_features.size(1))
        content_features = content_features[:, :seq_len, :]
        motion_features = motion_features[:, :seq_len, :]

        global_content = torch.mean(content_features, dim=1)
        global_motion = torch.mean(motion_features, dim=1)

        var_content = torch.var(content_features, dim=1) + self.eps  # (batch, feature_dim)
        var_motion = torch.var(motion_features, dim=1) + self.eps  # (batch, feature_dim)

        kl_div = 0.5 * torch.sum(torch.log(var_motion / var_content) +
                                 (var_content + (global_content - global_motion) ** 2) / var_motion - 1, dim=1)
        return torch.mean(kl_div)

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // (period)
    bias = - torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i * 2:i * 2 + 2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask == 1).to(device=device)


class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, content_features, style_features):
        """
        content_features: (batch, seq_len_c, feature_dim)
        style_features: (batch, seq_len_s, feature_dim)
        """
        seq_len_c = content_features.size(1)
        seq_len_s = style_features.size(1)
        if seq_len_c > seq_len_s:
            style_features = F.adaptive_avg_pool1d(style_features.transpose(1, 2), seq_len_c).transpose(1, 2)
        elif seq_len_c < seq_len_s:
            content_features = F.adaptive_avg_pool1d(content_features.transpose(1, 2), seq_len_s).transpose(1, 2)

        content_global = torch.mean(content_features, dim=1)  # (batch, feature_dim)
        style_global = torch.mean(style_features, dim=1)  # (batch, feature_dim)

        batch_size = content_global.size(0)
        sim_matrix = self.cosine_similarity(content_global.unsqueeze(1), style_global.unsqueeze(0))

        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=content_global.device)
        positives = sim_matrix[pos_mask].view(batch_size, 1)
        negatives = sim_matrix[~pos_mask].view(batch_size, batch_size - 1)
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=content_global.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss


# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, period, d_model)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.Wq = nn.Linear(1, embed_dim)
        self.Wk = nn.Linear(1, embed_dim)
        self.Wv = nn.Linear(1, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, obj_vector, my_obj_vector):
        obj_vector = obj_vector.unsqueeze(-1)
        my_obj_vector = my_obj_vector.unsqueeze(-1)

        Q = self.Wq(obj_vector)
        K = self.Wk(my_obj_vector)
        V = self.Wv(my_obj_vector)

        attention_output, _ = self.attention(Q, K, V)

        attention_output = self.head(self.norm(attention_output)).squeeze(-1)

        return attention_output


class ResidualConnect(nn.Module):
    def __init__(self):
        super(ResidualConnect, self).__init__()

    def forward(self, inputs, residual):
        return F.relu(inputs + residual)


class Onehot_Style_Encoder(nn.Module):
    def __init__(self, args, deep=4):
        super(Onehot_Style_Encoder, self).__init__()
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        self.my_obj_vector = nn.Embedding(len(args.train_subjects.split()), args.feature_dim)

        self.dropout = nn.Dropout(0.1)
        self.attention = Attention(4, num_heads=4)

        self.residual_connect = ResidualConnect()

    def forward(self, one_hot):
        obj_embedding = self.obj_vector(one_hot)  # (1, feature_dim)
        my_obj_embedding = self.my_obj_vector(torch.argmax(one_hot, dim=1))
        attention_output = self.dropout(self.attention(obj_embedding, my_obj_embedding))
        obj_embedding = self.residual_connect(obj_embedding, attention_output)

        return obj_embedding

class FeatureProjection(nn.Module):
    def __init__(self, feature_dim, hidden_size, feat_proj_dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_size)
        self.dropout = nn.Dropout(feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states

class Conv1DIN(nn.Module):
    def __init__(self, in_conv_dim, out_conv_dim, kernel, stride, padding, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.IN = nn.InstanceNorm1d(out_conv_dim, affine=False)
        self.activation = F.relu

    def forward(self, hidden_states):  # hidden_states: (B,C,L)
        hidden_states = self.conv(hidden_states)

        hidden_states = self.IN(hidden_states)

        hidden_states = self.activation(hidden_states)
        return hidden_states

class FeatureProjectionIN(nn.Module):
    def __init__(self, feature_dim, hidden_size, feat_proj_dropout=0.0):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_size)
        self.dropout = nn.Dropout(feat_proj_dropout)

    def forward(self, hidden_states):  # hidden_states (B,L,C)
        hidden_states = hidden_states.transpose(-2,-1)  # (B,C,L)
        norm_hidden_states = self.instance_norm(hidden_states)
        norm_hidden_states = norm_hidden_states.transpose(-2,-1)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class Motion_ContentEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.cfg = args
        self.motion_map = nn.Linear(args.feature_dim, args.feature_dim)
        conv_layers = [
            Conv1DIN(in_conv_dim=args.feature_dim, out_conv_dim=args.feature_dim, kernel=3, stride=1, padding=1,
                     bias=False) for i in range(6)]
        self.feature_projection = FeatureProjectionIN(feature_dim=args.feature_dim, hidden_size=args.feature_dim * 2)
        self.conv_layers = nn.ModuleList(conv_layers)

        self.PE = PeriodicPositionalEncoding(args.feature_dim * 2, period=args.period)

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.feature_dim * 2, nhead=2,
                                                   dim_feedforward=3 * args.feature_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.feature_map = nn.Linear(args.feature_dim * 2, args.feature_dim)

    def forward(self, input):
        extract_features = self.motion_map(input)  # (B,L,C)
        extract_features = input.transpose(1, 2)  # (B,C,L)
        for conv_layer in self.conv_layers:
            extract_features = conv_layer(extract_features)
        extract_features = extract_features.transpose(1, 2)  # (B,L,C)
        hidden_states, extract_features = self.feature_projection(extract_features)  # # (B,L,C)
        hidden_states = self.PE(hidden_states)
        hidden_states = self.transformer_encoder(hidden_states)  # (B,L,C)
        content_code = self.feature_map(hidden_states)

        return content_code

class Conv1DLN(nn.Module):
    def __init__(self, in_conv_dim, out_conv_dim, kernel, stride, padding, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.layer_norm = nn.LayerNorm(out_conv_dim, elementwise_affine=True)
        self.activation = F.relu

    def forward(self, hidden_states):  # hidden_states: (B,C,L)
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)  # (B,L,C)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)  # (B,C,L)

        hidden_states = self.activation(hidden_states, inplace=True)
        return hidden_states


class Motion_StyleEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.cfg = args
        self.motion_map = nn.Linear(args.feature_dim, args.feature_dim)
        conv_layers = [
            Conv1DLN(in_conv_dim=args.feature_dim, out_conv_dim=args.feature_dim, kernel=3, stride=1, padding=1,
                     bias=False) for i in range(6)]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.feature_projection = FeatureProjection(feature_dim=args.feature_dim, hidden_size=args.feature_dim * 2)
        self.PE = PeriodicPositionalEncoding(args.feature_dim * 2, period=args.period)
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.feature_dim * 2, nhead=2,
                                                   dim_feedforward=args.feature_dim * 3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.feature_map = nn.Linear(args.feature_dim * 2, args.feature_dim)

    def forward(self, input):
        extract_features = self.motion_map(input)  # (B,L,C)
        extract_features = input.transpose(1,2)  # (B,C,L)
        for conv_layer in self.conv_layers:
            extract_features = conv_layer(extract_features)
        extract_features = extract_features.transpose(1, 2)  # (B,L,C)
        hidden_states, extract_features = self.feature_projection(extract_features)  # # (B,L,C)
        hidden_states = self.PE(hidden_states)
        hidden_states = self.transformer_encoder(hidden_states)  # (B,L,C)
        style_code1 = self.feature_map(hidden_states)
        style_code = torch.mean(style_code1, dim=-2)
        return style_code, style_code1


class MergeBlock(nn.Module):
    def __init__(self, in_channels, merge_scale, num_wtok, expand=2):
        super().__init__()

        out_channels = in_channels * expand
        # self.MS = merge_scale
        self.MS = 2
        self.num_wtok = num_wtok
        self.pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor):
        x_wtok, x_fea = x[:, :self.num_wtok], x[:, self.num_wtok:]  #（1, 318, 128）  (1, 5, 128)

        B, T, C = x_fea.shape
        ms = T if self.MS == -1 else self.MS

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x_fea = F.pad(x_fea, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        x_fea = x_fea.view(B, T // ms, ms, C)
        x_fea = self.pool(x_fea).squeeze(dim=-2)

        x = torch.cat((x_wtok, x_fea), dim=1)
        x = self.norm(self.fc(x))

        return x


class AudioStyle(nn.Module):
    def __init__(self):
        super(AudioStyle, self).__init__()
        self.wav2vec2_emo = Wav2Vec2Model.from_pretrained(
            "wav2emo"
        )
        self.wav2vec2_emo.feature_extractor._freeze_parameters()

        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.tanh = nn.Tanh()
        self.fc3 = nn.Linear(1024, 128)

    def forward(self, audio):
        emo_features = self.wav2vec2_emo(audio).last_hidden_state

        pooled_features = self.maxpool(emo_features.transpose(1, 2)).squeeze(-1)

        x = self.fc1(pooled_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        style_code = self.tanh(x)
        style_vector = self.fc3(emo_features)

        return style_code, style_vector

class AdaIN(nn.Module):
    def __init__(self, c_cond: int, c_h: int):
        super(AdaIN, self).__init__()
        self.c_h = c_h
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.linear_layer = nn.Linear(c_cond, c_h * 2)

    def forward(self, x, x_cond):
        x_cond = self.linear_layer(x_cond)
        mean, std = x_cond[:, : self.c_h], x_cond[:, self.c_h :]
        mean, std = mean.unsqueeze(-1), std.unsqueeze(-1)
        x = x.transpose(1,2)  # (N,C,L)
        x = self.norm_layer(x)
        x = x * std + mean
        x = x.transpose(1,2)  # (N,L,C)
        return x


class PTalker(nn.Module):
    def __init__(self, args, mouth_loss_ratio=0.6, **kwargs):
        super(PTalker, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        self.device = args.device
        self.audio_encoder = HubertModel.from_pretrained(
            "hubert-large-ls960-ft")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(1024, args.feature_dim)
        self.vertice_encoder = nn.Linear(args.vertice_dim, args.feature_dim)
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period)
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4,
                                                    dim_feedforward=2 * args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.vertice_decoder = nn.Linear(args.feature_dim, args.vertice_dim)
        self.onehot_style_encoder = Onehot_Style_Encoder(args)
        nn.init.constant_(self.vertice_decoder.weight, 0)
        nn.init.constant_(self.vertice_decoder.bias, 0)
        with open(args.lip_region) as f:
            maps = f.read().split(", ")
            self.mouth_map = [int(i) for i in maps]
        self.mouth_loss_ratio = mouth_loss_ratio

        self.AudioStyle = AudioStyle()
        self.auxiliary_style_classifier = AuxiliaryStyleClassifier(input_dim=128, num_classes=6)
        self.content_motion = Motion_ContentEncoder(args)
        self.style_motion = Motion_StyleEncoder(args)
        self.orthogonal_loss = orthogonal_loss
        self.cons = ContentContrastiveLoss()
        self.style_classifier = StyleClassifier(args.feature_dim, len(args.train_subjects.split()))
        self.info_loss_fn = InfoNCELoss(temperature=0.1)
        self.kl_loss = CrossModalKLDivergence()
        self.start_token = nn.Parameter(torch.zeros(1, 1, args.feature_dim), requires_grad=True)

    def forward(self, audio, template, vertice, one_hot, criterion, teacher_forcing=True, sampling_rate=1.0):
        # template: (batch, V*3) -> (batch, 1, V*3)
        template = template.unsqueeze(1)

        style_emb = self.onehot_style_encoder(one_hot)

        style_logits = self.auxiliary_style_classifier(style_emb)
        identity_labels = torch.argmax(one_hot, dim=-1)
        aux_loss = nn.CrossEntropyLoss()(style_logits, identity_labels)

        frame_num = vertice.shape[1]

        style_audio, style_audio_vector = self.AudioStyle(audio)
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        if self.dataset == "BIWI":
            if hidden_states.shape[1] < frame_num * 2:
                vertice = vertice[:, :hidden_states.shape[1] // 2]
                frame_num = hidden_states.shape[1] // 2
        hidden_states = self.audio_feature_map(hidden_states)  # (batch, seq_len, feature_dim)
        init = self.start_token.expand(1, -1, -1)

        content_global = torch.mean(hidden_states, dim=1)  # (batch, feature_dim)
        reversed_content = GRL(content_global, alpha=1.0)
        speaker_logits = self.style_classifier(reversed_content)
        speaker_labels = torch.argmax(one_hot, dim=1)
        adv_loss = F.cross_entropy(speaker_logits, speaker_labels)


        if sampling_rate > 0:
            vertice_input = torch.cat((template, vertice[:, :-1]), 1)
            vertice_input = vertice_input - template
            vertice_input = self.vertice_encoder(vertice_input)    # (batch, seq_len, feature_dim)
            content_motion = self.content_motion(vertice_input)      # (batch, seq_len, feature_dim)
            style_motion, style_motion_vector = self.style_motion(vertice_input)  # (batch, feature_dim)
            vertice_input = vertice_input + style_emb + style_audio + style_motion + init.squeeze(1)
            vertice_input = self.PPE(vertice_input)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_decoder(vertice_out)
        else:
            for i in range(frame_num):
                if i == 0:
                    vertice_emb = style_emb.unsqueeze(0)
                    vertice_input = self.PPE(vertice_emb + style_audio)
                else:
                    vertice_input = self.PPE(vertice_emb)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_decoder(vertice_out)
                new_output = self.vertice_encoder(vertice_out[:, -1, :]).unsqueeze(1)
                if i == 0:
                    new_output = new_output + style_emb + style_audio
                else:
                    content_motion = self.content_motion(vertice_input)
                    style_motion, style_motion_vector = self.style_motion(vertice_input)
                    new_output = new_output + style_emb + style_audio + style_motion
                vertice_emb = torch.cat((vertice_emb, new_output), 1)
        vertice_out = vertice_out + template

        or_loss_audio = orthogonal_loss(hidden_states, style_audio_vector)

        or_loss_motion = orthogonal_loss(content_motion, style_motion_vector)

        cons_loss = self.cons(hidden_states, content_motion)

        kl_loss = self.kl_loss(hidden_states, content_motion)

        info_loss_audio = self.info_loss_fn(hidden_states, style_audio_vector)

        info_loss_motion = self.info_loss_fn(content_motion, style_motion_vector)

        rec_loss = torch.mean(criterion(vertice_out, vertice))
        mouth_loss = criterion(vertice_out[:, :, self.mouth_map], vertice[:, :, self.mouth_map])

        constraint_loss = 0.001 * or_loss_audio + 0.001 * info_loss_audio + 0.0001 * info_loss_motion + \
                          0.0001 * or_loss_motion + 0.001 * cons_loss + 0.001 * kl_loss + 0.001 * adv_loss# (batch, seq_len, V*3)

        return rec_loss, constraint_loss, mouth_loss, vertice_out, vertice

    def predict_seen(self, audio, template, one_hot):

        template = template.unsqueeze(1)  # (batch, 1, V*3)
        style_emb = self.onehot_style_encoder(one_hot)
        style_audio, _ = self.AudioStyle(audio)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1] // 2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)
        for i in range(frame_num):
            if i == 0:
                vertice_emb = style_emb.unsqueeze(1)
                vertice_input = self.PPE(vertice_emb + style_audio)
            else:
                vertice_input = self.PPE(vertice_emb)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_decoder(vertice_out)
            new_output = self.vertice_encoder(vertice_out[:, -1, :]).unsqueeze(1)
            if i == 0:
                new_output = new_output + style_emb + style_audio
            else:
                style_motion, _ = self.style_motion(vertice_input)
                new_output = new_output + style_emb + style_audio + style_motion
            vertice_emb = torch.cat((vertice_emb, new_output), dim=1)
        vertice_out = vertice_out + template
        return vertice_out

    def predict_unseen(self, audio, template, one_hot):
        template = template.unsqueeze(1)  # (batch, 1, V*3)
        style_emb = self.onehot_style_encoder(one_hot)
        style_audio, _ = self.AudioStyle(audio)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        init = self.start_token.expand(1, -1, -1)
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1] // 2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)

        for i in range(frame_num):
            if i == 0:
                vertice_emb = init.squeeze(1) + style_audio.unsqueeze(1)
                vertice_input = self.PPE(vertice_emb)
            else:
                vertice_input = self.PPE(vertice_emb)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_decoder(vertice_out)
            new_output = self.vertice_encoder(vertice_out[:, -1, :]).unsqueeze(1)
            if i == 0:
                new_output = new_output + style_audio
            else:
                style_motion, _ = self.style_motion(vertice_input)
                new_output = new_output + style_audio + style_motion
            vertice_emb = torch.cat((vertice_emb, new_output), dim=1)
        vertice_out = vertice_out + template
        return vertice_out

    def predict_with_ref_motion(self, audio, template, ref_motion_seq):
        """
        """
        device = self.device
        template = template.unsqueeze(1)  # (batch,1,V*3)
        style_audio, _ = self.AudioStyle(audio)

        first_ref = ref_motion_seq[:, 0, :]  # (batch, V*3)
        vertice_emb = self.vertice_encoder(first_ref)  # (batch, feature_dim)
        vertice_emb = vertice_emb.unsqueeze(1)  # (batch,1,feature_dim)


        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        frame_num = hidden_states.shape[1] // (2 if self.dataset == "BIWI" else 1)
        hidden_states = self.audio_feature_map(hidden_states)

        for i in range(frame_num):
            vertice_input = self.PPE(vertice_emb)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].to(device)
            memory_mask = enc_dec_mask(device, self.dataset,
                                       vertice_input.shape[1], hidden_states.shape[1])

            vertice_out = self.transformer_decoder(
                vertice_input, hidden_states,
                tgt_mask=tgt_mask, memory_mask=memory_mask
            )
            vertice_out = self.vertice_decoder(vertice_out)
            last = vertice_out[:, -1, :]  # (batch, V*3)
            new_output = self.vertice_encoder(last).unsqueeze(1)

            if i == 0:

                new_output = new_output + style_audio.unsqueeze(1)
            else:
                style_motion, _ = self.style_motion(vertice_input)
                new_output = new_output + style_audio.unsqueeze(1) + style_motion.unsqueeze(1)

            vertice_emb = torch.cat((vertice_emb, new_output), dim=1)

        vertice_out = vertice_out + template

        return vertice_out

