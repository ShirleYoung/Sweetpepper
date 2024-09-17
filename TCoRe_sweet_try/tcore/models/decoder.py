
import tcore.models.blocks as blocks
import torch
from tcore.models.positional_encoder import PositionalEncoder
from tcore.utils.interpolate import knn_up
from torch import nn


class TransformerDecoder(nn.Module):
    def __init__(self, cfg, bb_cfg):
        super().__init__()
        hidden_dim = int(cfg.HIDDEN_DIM * cfg.CR)
        nheads = cfg.NHEADS

        cfg.POS_ENC.FEAT_SIZE = hidden_dim
        self.pe_layer = PositionalEncoder(cfg.POS_ENC)

        self.num_layers = cfg.FEATURE_LEVELS * cfg.DEC_BLOCKS
        self.nheads = nheads
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.aux_outputs = cfg.AUX_OUTPUTS
        self.offset_scaling = cfg.OFFSET_SCALING
        self.iterative_template = cfg.ITERATIVE_TEMPLATE
        self.template_knn_up = knn_up(50)

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                blocks.SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0)
            )

            self.transformer_cross_attention_layers.append(
                blocks.CrossAttentionLayer(
                    d_model=hidden_dim, nhead=nheads, dropout=0.0
                )
            )

            self.transformer_ffn_layers.append(
                blocks.FFNLayer(
                    d_model=hidden_dim, dim_feedforward=cfg.DIM_FEEDFORWARD, dropout=0.0
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = cfg.NUM_QUERIES

        self.query_feat = nn.Embedding(cfg.NUM_QUERIES, hidden_dim)

        self.query_embed = nn.Embedding(cfg.NUM_QUERIES, hidden_dim)

        self.num_feature_levels = cfg.FEATURE_LEVELS
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.mask_feat_proj = nn.Sequential()
        cr = bb_cfg.CR
        in_channels = [int(cr * x) for x in bb_cfg.CHANNELS]

        if in_channels[-1] != hidden_dim:
            self.mask_feat_proj = nn.Linear(in_channels[-1], hidden_dim)

        in_channels = in_channels[:-1] 
        in_channels = in_channels[-self.num_feature_levels :]

        self.input_proj = nn.ModuleList()
        for i in range(self.num_layers):
            self.input_proj.append(nn.Linear(in_channels[-1], hidden_dim))

        # output FFNs
        self.offset_head = blocks.MLP(
            hidden_dim, hidden_dim, output_dim=1, num_layers=3, tanh=False
        )

    def forward(self, feats, coors, pad_masks, template_points):
        bs = template_points.shape[0]

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        output = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)

        all_point_templates = []
        predictions_offsets = []

        (
            offset,
            pt_template,
            template_features,
        ) = self.forward_prediction_heads(
            output,
            template_points,
            pad_masks,
            feats,
            coors,
            update_initial_template=False,
        )

        predictions_offsets.append(offset)
        all_point_templates.append(pt_template)

        for i in range(self.num_layers):
            src = self.input_proj[i](template_features)
            pos = self.pe_layer(pt_template)
            # cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src,
                attn_mask=None,
                padding_mask=None,
                pos=pos,
                query_pos=query_embed,
            )

            # self-attention
            output = self.transformer_self_attention_layers[i](
                output, attn_mask=None, padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            (
                offset,
                pt_template,
                template_features,
            ) = self.forward_prediction_heads(
                output,
                all_point_templates[-1],
                pad_masks,
                feats,
                coors,
                update_initial_template=self.iterative_template,
            )

            predictions_offsets.append(offset)
            all_point_templates.append(pt_template)

        assert len(predictions_offsets) == self.num_layers + 1

        out = {
            "offsets": predictions_offsets[-1],
            "previous_template_points": all_point_templates[-1],
        }

        if self.aux_outputs:
            out["aux_outputs"] = self.set_aux(
                predictions_offsets, all_point_templates
            )

        return out

    def forward_prediction_heads(
        self,
        output,
        current_template_points,
        pad_masks,
        pt_feats,
        pt_coors,
        update_initial_template,
    ):
        decoder_output = self.decoder_norm(output)  # Layer norm
        offset = self.offset_head(decoder_output)  # MLP

        if update_initial_template:
            offsets_scaled = torch.sigmoid(offset) * self.offset_scaling
            pt_template = current_template_points * offsets_scaled
        else:
            pt_template = current_template_points

        template_features = self.point_feats_to_template_feats(
            pad_masks, pt_feats, pt_coors, pt_template
        )

        return offset, pt_template, template_features

    @torch.jit.unused
    def set_aux(self, outputs_offset, template_points):
        return [
            {"offsets": b, "previous_template_points": c}
            for b, c in zip(
                outputs_offset[:-1], template_points[:-1]
            )
        ]

    def point_feats_to_template_feats(self, pad_masks, pt_feats, pt_coors, pt_template):
        template_feats = []
        for pmask, feat, coor, tmp in zip(
            pad_masks[-1], pt_feats[-1], pt_coors[-1], pt_template
        ):
            template_feat = self.template_knn_up(
                coor[~pmask].squeeze(), feat[~pmask].squeeze(), tmp.squeeze()
            )

            template_feats.append(template_feat.unsqueeze(0))

        return torch.cat(template_feats, 0)
