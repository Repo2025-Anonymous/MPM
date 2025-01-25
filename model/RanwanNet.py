import model.resnet as resnet
from segment_anything.modeling import Sam
import torch
from torch import nn
import torch.nn.functional as F
import math
import ot
import numpy as np


class cycle_attention(nn.Module):
    def __init__(self, in_channels, agent_num, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(cycle_attention, self).__init__()
        self.scare = in_channels ** (-0.5)
        self.agent_num = agent_num

        self.q_fc = nn.Linear(in_channels, in_channels, bias=qkv_bias)
        self.k_fc = nn.Linear(in_channels, in_channels, bias=qkv_bias)
        self.v_fc = nn.Linear(in_channels, in_channels, bias=qkv_bias)

        self.part_tokens = nn.Parameter(torch.zeros(1, self.agent_num, 256))
        nn.init.normal_(self.part_tokens.permute(2, 0, 1), std=0.001)

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

    def supp_to_prototype(self, supp_feat_list, supp_mask_list, memory_prototype_list):

        for i in range(len(supp_feat_list)):
            supp_mask = supp_mask_list[i]
            supp_feat = supp_feat_list[i]

            B, C, H, W = supp_feat.shape

            supp_proto_fg = self.masked_average_pooling(supp_feat, (supp_mask == 1).float())[None, :]
            supp_proto_fg = supp_proto_fg.unsqueeze(-1).unsqueeze(-1)
            supp_proto_fg = supp_proto_fg.view(B, -1, C) + self.part_tokens.expand(B, self.agent_num, C)
            q = self.q_fc(supp_proto_fg)

            supp_mask = F.interpolate(supp_mask.unsqueeze(1).float(), supp_feat.size()[2:], mode='bilinear', align_corners=False)
            supp_feat = supp_feat * supp_mask
            supp_feat = supp_feat.view(B, C, -1).transpose(-2, -1)

            k = self.k_fc(supp_feat)
            v = self.v_fc(supp_feat)

            attn = q @ k.transpose(-2, -1)

            supp_mask = (~(supp_mask.to(torch.int32))).float()
            supp_mask = supp_mask * -10000.0
            supp_mask = supp_mask.unsqueeze(1).view(B, 1, H*W)
            attn = attn + supp_mask

            attn = F.softmax(attn, dim=-1)

            attn_list = []
            for i in range(attn.shape[0]):
                attn_i = attn[i]
                attn_fg = torch.masked_select(attn_i, attn_i > 0).view(attn_i.shape[0], -1)
                cost = (1-attn_fg).detach().cpu()

                r, c = attn_fg.size()
                attn_fg_new = ot.sinkhorn(np.ones(r) / r, np.ones(c) / c, np.array(cost), 0.5)
                attn_fg_new = torch.Tensor(attn_fg_new).cuda()
                attn_i_new = torch.zeros_like(attn_i)
                attn_i_new[attn_i > 0] = attn_fg_new.view(-1)
                attn_list.append(attn_i_new)
            attn = torch.stack(attn_list, dim=0)
            
            supp_proto_fg = supp_proto_fg + attn @ v

            memory_prototype_list.append(supp_proto_fg)
        return memory_prototype_list


    def query_to_prototype(self, query_feat, query_prediction, memory_prototype_list):

        curr_prototype = torch.cat(memory_prototype_list, dim=1)
        b, n, c = curr_prototype.size()

        query_prediction = query_prediction.softmax(dim=1)

        foreground_mask = query_prediction[:, 1, :, :].unsqueeze(1)
        foreground_mask = F.interpolate(foreground_mask, query_feat.size()[2:], mode='bilinear', align_corners=False)

        foreground_mask[foreground_mask > 0.7] = 1
        foreground_mask[foreground_mask < 0.3] = 0

        query_feat = query_feat * foreground_mask
        query_feat = query_feat.view(b, c, -1).transpose(-2, -1)

        attn = (curr_prototype @ query_feat.transpose(-2, -1)) * self.scare
        attn = F.softmax(attn, dim=-1)
        curr_prototype = curr_prototype + attn @ query_feat

        memory_prototype_list.append(curr_prototype)
        curr_prototype = torch.mean(torch.cat(memory_prototype_list, dim=1), dim=1, keepdim=True)
        return memory_prototype_list, curr_prototype

    def forward(self,
                supp_feat_list,
                supp_mask_list,
                query_feat,
                query_prediction,
                memory_prototype_list):

        memory_prototype_list = self.supp_to_prototype(supp_feat_list, supp_mask_list, memory_prototype_list)
        memory_prototype_list, curr_prototype = self.query_to_prototype(query_feat, query_prediction, memory_prototype_list)
        return memory_prototype_list, curr_prototype

class QuickGELU(nn.Module):
    def forward(self, x:torch.Tensor):
        return x*torch.sigmoid(1.702*x)

class adapter(nn.Module):
    def __init__(self, c=768, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c//r, bias=True),
            QuickGELU(),
            nn.Linear(c//r, c, bias=True))
        self.IN = nn.LayerNorm(c)
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init_weights)
    
    def forward(self, x):
        ori = x
        b, h, w, c = x.size()
        out1 = self.IN(x.view(b, h*w, c))
        out = self.fc(out1)
        return ori+out.view(b, h, w, c)


class _LoRA_qkv(nn.Module):

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    def __init__(self, sam_model: Sam, r: int, backbone, shot, lora_layer=None):
        super(LoRA_Sam, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.shot = shot
        self.cycle = 3

        # for 1-shot, agent_num = 10
        # for 5-shot, agent_num = 2
        self.cnn_cycle_attention = cycle_attention(in_channels=256, agent_num=10)
        self.sam_cycle_attention = cycle_attention(in_channels=256, agent_num=10)

        self.adapter = nn.ModuleList([adapter() for i in range(12)])

        assert r > 0

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))
        self.w_As = []
        self.w_Bs = []

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.reset_parameters()
        self.sam = sam_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, img_s_list, mask_s_list, img_q, mask_q):
        b, c, h, w = img_q.size()

        # feature maps of support images
        feature_s_list = []
        for k in range(len(img_s_list)):
            with torch.no_grad():
                s_0 = self.layer0(img_s_list[k])
                s_0 = self.layer1(s_0)
            s_0 = self.layer2(s_0)
            s_0 = self.layer3(s_0)

            s_0_feat_1, s_0_feat_2, s_0_feat_3, s_0_feat_4 = torch.chunk(s_0, 4, dim=1)
            s_0 = torch.mean(torch.stack((s_0_feat_1, s_0_feat_2, s_0_feat_3, s_0_feat_4), dim=0), dim=0)

            feature_s_list.append(s_0)
            del s_0

        feature_s_ls = torch.cat(feature_s_list, dim=0)

        with torch.no_grad():
            q_0 = self.layer0(img_q)
            q_0 = self.layer1(q_0)
        q_0 = self.layer2(q_0)
        feature_q = self.layer3(q_0)

        query_feat_1, query_feat_2, query_feat_3, query_feat_4 = torch.chunk(feature_q, 4, dim=1)
        feature_q = torch.mean(torch.stack((query_feat_1, query_feat_2, query_feat_3, query_feat_4), dim=0), dim=0)

        # 计算support图像的前景区域和背景区域的原型
        cnn_supp_proto_fg_list = []
        cnn_supp_proto_bg_list = []

        for k in range(len(img_s_list)):
            cnn_supp_proto_fg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 1).float())[None, :]
            cnn_supp_proto_bg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 0).float())[None, :]

            cnn_supp_proto_fg_list.append(cnn_supp_proto_fg)
            cnn_supp_proto_bg_list.append(cnn_supp_proto_bg)

        # average K foreground prototypes and K background prototypes
        cnn_supp_fp = torch.mean(torch.cat(cnn_supp_proto_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        cnn_supp_bp = torch.mean(torch.cat(cnn_supp_proto_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        # use SAM to extract suport & query's features
        supp_embeddings_list = []
        for idx in range(len(img_s_list)):
            supp_img = F.interpolate(img_s_list[idx], (512, 512), mode='bilinear', align_corners=True)
            supp_image_patch = self.sam.image_encoder.patch_embed(supp_img)
            supp_image_patch = supp_image_patch + self.sam.image_encoder.pos_embed

            for i in range(12):
                supp_image_patch = self.sam.image_encoder.blocks[i](supp_image_patch, self.adapter[i])
            supp_embedding = self.sam.image_encoder.neck(supp_image_patch.permute(0, 3, 1, 2))

            if supp_embedding.size() != feature_q.size():
                supp_embedding = F.interpolate(supp_embedding, feature_q.size()[2:], mode='bilinear', align_corners=False)
            supp_embeddings_list.append(supp_embedding)

        supp_embeddings_ls = torch.cat(supp_embeddings_list, dim=0)

        query_img = F.interpolate(img_q, (512, 512), mode='bilinear', align_corners=True)
        query_image_patch = self.sam.image_encoder.patch_embed(query_img)
        query_image_patch = query_image_patch + self.sam.image_encoder.pos_embed

        for i in range(12):
            query_image_patch = self.sam.image_encoder.blocks[i](query_image_patch, self.adapter[i])

        query_embedding = self.sam.image_encoder.neck(query_image_patch.permute(0, 3, 1, 2))

        if query_embedding.size() != feature_q.size():
            query_embedding = F.interpolate(query_embedding, feature_q.size()[2:], mode='bilinear', align_corners=False)

        # foreground(target class) and background prototypes pooled from K support features
        sam_feature_fg_list = []
        sam_feature_bg_list = []

        for k in range(len(img_s_list)):
            sam_feature_fg = self.masked_average_pooling(supp_embeddings_list[k], (mask_s_list[k] == 1).float())[None, :]
            sam_feature_bg = self.masked_average_pooling(supp_embeddings_list[k], (mask_s_list[k] == 0).float())[None, :]

            sam_feature_fg_list.append(sam_feature_fg)
            sam_feature_bg_list.append(sam_feature_bg)

        # average K foreground prototypes and K background prototypes
        sam_supp_fp = torch.mean(torch.cat(sam_feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)     # [b, 256, 1, 1]
        sam_supp_bp = torch.mean(torch.cat(sam_feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)     # [b, 256, 1, 1]

        sam_curr_proto = sam_supp_fp
        cnn_curr_proto = cnn_supp_fp

        B, C, H, W = cnn_curr_proto.size()

        sam_memory_prototype_list = []
        cnn_memory_prototype_list = []
        sam_memory_prototype_list.append(sam_curr_proto.view(B, C, -1).transpose(-2, -1))
        cnn_memory_prototype_list.append(cnn_curr_proto.view(B, C, -1).transpose(-2, -1))

        cnn_out_ls = []
        sam_out_ls = []

        for i in range(self.cycle):
            cnn_out, cnn_supp_out, new_cnn_curr_proto, new_cnn_bp = self.iter_BFP(cnn_curr_proto, cnn_supp_bp,
                                                                                  feature_s_ls, feature_q, 256)
            sam_out, sam_supp_out, new_sam_curr_proto, new_sam_bp = self.iter_BFP(sam_curr_proto, sam_supp_bp,
                                                                                  supp_embeddings_ls, query_embedding,256)

            cnn_memory_prototype_list.append(new_cnn_curr_proto.view(B, -1, C))
            sam_memory_prototype_list.append(new_sam_curr_proto.view(B, -1, C))

            cnn_memory_prototype_list, cnn_curr_proto = self.cnn_cycle_attention(feature_s_list, mask_s_list, feature_q,
                                                                                 cnn_out, cnn_memory_prototype_list)

            sam_memory_prototype_list, sam_curr_proto = self.sam_cycle_attention(supp_embeddings_list, mask_s_list,
                                                                                 query_embedding, sam_out, sam_memory_prototype_list)

            cnn_curr_proto = cnn_curr_proto.transpose(-2, -1).reshape(B, C, H, W)
            sam_curr_proto = sam_curr_proto.transpose(-2, -1).reshape(B, C, H, W)

            cnn_out = F.interpolate(cnn_out, size=(h, w), mode="bilinear", align_corners=True)
            sam_out = F.interpolate(sam_out, size=(h, w), mode="bilinear", align_corners=True)

            cnn_out_ls.append(cnn_out)
            sam_out_ls.append(sam_out)

        return cnn_out_ls, sam_out_ls


    def SSP_func(self, feature_q, out, channal):

        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7
            bg_thres = 0.6
            cur_feat = feature_q[epi].view(channal, -1)
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices] #.mean(-1)
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices] #.mean(-1)

            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(channal, f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(channal, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local


    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out


    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature
    

    def iter_BFP(self, FP, BP, feature_s_ls, feature_q, channal):
        out_0 = self.similarity_func(feature_q, FP, BP)
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0, channal)
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
        out_1 = self.similarity_func(feature_q, FP_1, BP_1)

        if self.shot > 1:
            FP_nshot = FP.repeat_interleave(self.shot, dim=0)
            FP_1 = FP_1.repeat_interleave(self.shot, dim=0)
            BP_1 = BP_1.repeat_interleave(self.shot, dim=0)
        supp_out_0 = self.similarity_func(feature_s_ls, FP_1, BP_1)
        SSFP_supp, SSBP_supp, ASFP_supp, ASBP_supp = self.SSP_func(feature_s_ls, supp_out_0, channal)

        if self.shot > 1:
            FP_supp = FP_nshot * 0.5 + SSFP_supp * 0.5
        else:
            FP_supp = FP * 0.5 + SSFP_supp * 0.5
        BP_supp = SSBP_supp * 0.3 + ASBP_supp * 0.7
        supp_out_1 = self.similarity_func(feature_s_ls, FP_supp, BP_supp)

        if self.shot > 1:
            for i in range(FP_supp.shape[0]//self.shot):
                for j in range(self.shot):
                    if j == 0:
                        FP_supp_avg = FP_supp[i * self.shot + j]
                        BP_supp_avg = BP_supp[i * self.shot + j]
                    else:
                        FP_supp_avg = FP_supp_avg + FP_supp[i * self.shot + j]
                        BP_supp_avg = BP_supp_avg + BP_supp[i * self.shot + j]

                FP_supp_avg = FP_supp_avg / self.shot
                BP_supp_avg = BP_supp_avg / self.shot
                FP_supp_avg = FP_supp_avg.reshape(1, FP_supp.shape[1], FP_supp.shape[2], FP_supp.shape[3])
                BP_supp_avg = BP_supp_avg.reshape(1, BP_supp.shape[1], BP_supp.shape[2], BP_supp.shape[3])

                if i == 0:
                    new_FP_supp = FP_supp_avg
                    new_BP_supp = BP_supp_avg
                else:
                    new_FP_supp = torch.cat((new_FP_supp, FP_supp_avg), dim=0)
                    new_BP_supp = torch.cat((new_BP_supp, BP_supp_avg), dim=0)

            FP_supp = new_FP_supp
            BP_supp = new_BP_supp

        return out_1, supp_out_1, FP_supp, BP_supp