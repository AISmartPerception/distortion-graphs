import torch
import torch.nn as nn
import torch.nn.functional as F
import einops, helper
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class VisionProjector(nn.Module):
    def __init__(self, config):
        super(VisionProjector, self).__init__()
        self.config = config
        self.dim = config['general']['feature_extractor_dim']
        self.expansion_factor = config['train']['model']['expansion_factor']
        hidden = int(self.expansion_factor*self.dim)
        self.imgA_conv = nn.Conv2d(in_channels=self.dim,
                                   out_channels=hidden,
                                   kernel_size=1,
                                   padding=0,
                                   stride=1,
                                   groups=1,
                                   bias=True)
        self.imgA_proj = nn.Conv2d(in_channels=hidden,
                                   out_channels=self.dim,
                                   kernel_size=1,
                                   padding=0,
                                   stride=1,
                                   groups=1,
                                   bias=True)
        self.imgB_conv = nn.Conv2d(in_channels=self.dim,
                                   out_channels=hidden,
                                   kernel_size=1,
                                   padding=0,
                                   stride=1,
                                   groups=1,
                                   bias=True)
        self.imgB_proj = nn.Conv2d(in_channels=hidden,
                                    out_channels=self.dim,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    groups=1,
                                    bias=True)
        
    def forward(self, imgA_ft, imgB_ft):
        imgA = self.imgA_proj(F.gelu(self.imgA_conv(imgA_ft)))
        imgB = self.imgB_proj(F.gelu(self.imgB_conv(imgB_ft)))
        return imgA, imgB

class MaskProjector(nn.Module):
    def __init__(self, config):
        super(MaskProjector, self).__init__()
        out_chn = config["general"]["feature_extractor_dim"]
        patch_size = config["general"]["patch_size"]
        self.conv = nn.Conv2d(in_channels=1,
                               out_channels=out_chn,
                               kernel_size=patch_size,
                               stride=patch_size)
    def forward(self, x):
        return self.conv(x)

class MaskFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=True):
        super(MaskFFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, 
                                    kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, 
                                     kernel_size=1, bias=bias)
    def forward(self, x):
        x = F.gelu(self.project_in(x))
        return self.project_out(x)

class MaskTransformer(nn.Module):
    def __init__(self, dim, num_heads,
                 qkv_bias,
                 expansion_factor):
        super(MaskTransformer, self).__init__()
        self.x_norm = nn.LayerNorm(dim)
        self.y_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, 
                                                    num_heads=num_heads,
                                                    bias=qkv_bias,
                                                    batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim,
                                                     num_heads=num_heads,
                                                     bias=qkv_bias,
                                                     batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MaskFFN(dim, expansion_factor)
    
    def forward(self, x, y):
        b, r, d, h, w = x.shape
        x = einops.rearrange(x, 'b r d h w -> b r h w d')
        y = einops.rearrange(y, 'b d h w -> b h w d')
        x = self.x_norm(x)
        y = self.y_norm(y)
        # self attention on y
        sa_qkv = einops.rearrange(y, 'b h w d -> b (h w) d')
        y_sa, _ = self.self_attention(sa_qkv, sa_qkv, sa_qkv)
        y_sa = einops.rearrange(y_sa, 'b (h w) d -> b h w d', h=h, w=w)
        y = y + y_sa # pre-norm residuals should be added
        # cross attention on regions w.r.t to the image
        ca_q = einops.rearrange(x, 'b r h w d -> (b r) (h w) d', b=b, r=r, h=h, w=w, d=d) #(b*r, h*w, d)
        y_kv = y.unsqueeze(1).expand(-1, r, -1, -1, -1) # (b, r, h, w, d)
        y_kv = einops.rearrange(y_kv, 'b r h w d -> (b r) (h w) d', b=b, r=r, h=h, w=w, d=d) #(b*r, h*w, d)
        attn_x, _ = self.cross_attention(ca_q, y_kv, y_kv)
        attn_x = einops.rearrange(attn_x, '(b r) (h w) d -> b r h w d', b=b, r=r, h=h, w=w, d=d)
        x = x + attn_x # pre-norm residuals should be added
        # ffn
        ffn_norm = self.norm2(x)
        ffn_norm = einops.rearrange(ffn_norm, 'b r h w d -> (b r) d h w', b=b, r=r, h=h, w=w, d=d)
        x = einops.rearrange(x, 'b r h w d -> b r d h w', b=b, r=r, h=h, w=w, d=d)
        y = einops.rearrange(y, 'b h w d -> b d h w', b=b, h=h, w=w, d=d)
        x = x + self.ffn(ffn_norm).reshape(b, r, d, h, w)
        return x, y

class PositionEncoding(nn.Module):
    def __init__(self, upperbound_tokens, dim, height, width):
        super(PositionEncoding, self).__init__()
        self.x_region_pe = nn.Parameter(torch.zeros(1, upperbound_tokens, dim, 1, 1))
        self.x_spatial_pe = nn.Parameter(torch.zeros(1, 1, dim, height, width))
        self.y_spatial_pe = nn.Parameter(torch.zeros(1, dim, height, width))
        nn.init.trunc_normal_(self.x_region_pe, std=0.02)
        nn.init.trunc_normal_(self.x_spatial_pe, std=0.02)
        nn.init.trunc_normal_(self.y_spatial_pe, std=0.02)

    def forward(self, x, y):
        b, r, d, h, w = x.shape
        y = y + self.y_spatial_pe
        x = x + self.x_region_pe[:, :r, :, :, :] + self.x_spatial_pe
        return x, y

class MaskDecoder(nn.Module):
    def __init__(self, config):
        super(MaskDecoder, self).__init__()
        self.dim = config['general']['feature_extractor_dim']
        self.resize_hw = config['general']['resize_shape']
        self.num_heads = config['train']['model']['num_heads']
        self.qkv_bias = config['train']['model']['qkv_bias']
        self.expansion_factor = config['train']['model']['expansion_factor']
        no_transformer_layers = config['train']['model']['decoder_transformer_blocks']
        h = w = self.resize_hw//config['general']['patch_size']
        self.positional_encoding = PositionEncoding(config['train']['model']['no_token_mark'], 
                                                    self.dim, h, w)
        transformer_layers = []
        for _ in range(no_transformer_layers):
            transformer_layers.append(MaskTransformer(self.dim, self.num_heads,
                                      self.qkv_bias,
                                      self.expansion_factor))
        self.transformer = helper.MultipleSequential(*transformer_layers)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, imgB_proj_fts, imgA_mask_fts):
        x, y = self.positional_encoding(imgA_mask_fts, imgB_proj_fts)
        for transformer_layer in self.transformer:
            x, y = transformer_layer(x, y)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return x

class PredictionHead(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PredictionHead, self).__init__()
        self.projection = nn.Linear(input_shape, input_shape//2)
        self.ln1 = nn.LayerNorm(input_shape//2)
        self.projection_02 = nn.Linear(input_shape//2, input_shape//4)
        self.ln2 = nn.LayerNorm(input_shape//4)
        self.output = nn.Linear(input_shape//4, output_shape)

    def forward(self, x):
        x = F.gelu(self.ln1(self.projection(x)))
        x = F.gelu(self.ln2(self.projection_02(x)))
        return self.output(x)

class PandaDG(nn.Module):
    def __init__(self, config, device):
        super(PandaDG, self).__init__()
        self.device = device
        self.feature_extractor_type = config['general']['feature_extractor']
        self.resize_hw = config['general']['resize_shape']
        self.backbone_ckpt = config['general']['backbone_ckpt']
        self.backbone = self.build_feature_extractor(self.feature_extractor_type)
        self.dim = config['general']['feature_extractor_dim']
        self.token_pool_size = int(config['train']['model']['no_token_mark'])

        # vision projection
        self.vision_projection = VisionProjector(config)
        self.mask_projection = MaskProjector(config)

        self.anchor_decoder = MaskDecoder(config)
        self.target_decoder = MaskDecoder(config)
        self.a_token_pool = nn.Parameter(torch.randn(self.token_pool_size, self.resize_hw, self.resize_hw))
        self.t_token_pool = nn.Parameter(torch.randn(self.token_pool_size, self.resize_hw, self.resize_hw))
        
        # prediction/regression head MLPs
        self.region_prediction = PredictionHead(self.dim, 5)
        self.a_distortion_prediction = PredictionHead(self.dim, 15) # one of the 14 distortions + 1 for clean
        self.t_distortion_prediction = PredictionHead(self.dim, 15) # one of the 14 distortions + 1 for clean
        self.a_severity_prediction = PredictionHead(self.dim, 4) # either none, 1, 2 or 3
        self.t_severity_prediction = PredictionHead(self.dim, 4) # either none, 1, 2 or 3
        self.a_score_regressor = PredictionHead(self.dim, 1) # regression head for FR scores
        self.t_score_regressor = PredictionHead(self.dim, 1) # regression head for FR scores

    def build_feature_extractor(self, feature_extractor_type):
        if feature_extractor_type == "dinov2":
            backbone = torch.hub.load('facebookresearch/dinov2', self.backbone_ckpt).to(self.device).eval()
            for param in backbone.parameters():
                param.requires_grad = False
            return backbone
        else:
            raise ValueError("Invalid Detector!")

    def get_features_from_backbone(self, img):
        if self.feature_extractor_type == "dinov2":
            # get patch features not CLS
            encoded_features = self.backbone.forward_features(img)["x_norm_patchtokens"]
            if len(encoded_features.shape) == 3:
                # the patch tokens are 256 in Dino's case
                h = w = int(encoded_features.shape[1]**0.5) # assuming square patch grid
                assert h*w == encoded_features.shape[1]
            x = encoded_features.reshape(shape=(encoded_features.shape[0], h, w, -1))
            x = torch.einsum('nhwc->nchw', x) # B, d, H, W
        else:
            raise ValueError("Invalid Feature Extractor Type.")
        return x, h, w
    
    # this is for mask
    def extract_mask_img_features(self, img, masks, mode="anchor"):
        x, _, _ = self.get_features_from_backbone(img) # (b, d, h//14, w//14)
        selected_token_pool = self.uniform_sampling(masks, mode) # (b, r, h, w)
        spatial_tok_pool = self.compute_spatial_token_pool(selected_token_pool, masks) # (b, r, d, h//14, w//14)
        spatial_tok_pool = self.masked_input(x, spatial_tok_pool) # (b, r, d, h//14, w//14)
        return x, spatial_tok_pool
    
    def masked_input(self, img_fts, spatial_tok_pool):
        # img_fts: (b, d, h/14, w/14) | spatial_tok_pool: (b, r, d, h/14, w/14)
        img_fts = img_fts.unsqueeze(1) # (b, 1, d, h/14, w/14)
        img_weighted_toks_up = img_fts * spatial_tok_pool # (b, r, d, h/14, w/14)
        return img_weighted_toks_up

    def compute_spatial_token_pool(self, selected_token_pool, masks):
        # selected_token_pool: (b, r, h, w) | masks: (b, r, h, w)
        masked_attn_pool = selected_token_pool * masks
        b, r, h, w = masked_attn_pool.shape
        masked_attn_pool = masked_attn_pool.reshape(b*r, 1, h, w) # (b*r, 1, h, w)
        masked_attn_pool = self.mask_projection(masked_attn_pool) # (b*r, d, h//14, w//14)
        masked_attn_pool = masked_attn_pool.reshape(b, r, self.dim,
                                                    masked_attn_pool.shape[-2],
                                                    masked_attn_pool.shape[-1]) # (b, r, d, h//14, w//14)
        return masked_attn_pool

    def uniform_sampling(self, masks, mode="anchor"):
        # without replacement
        b, r, h, w = masks.shape
        def do_for_one(_):
            return torch.randperm(self.token_pool_size, 
                                  device=self.a_token_pool.device)[:r]
        indices = torch.func.vmap(do_for_one, randomness="different")(
            torch.arange(b, device=self.a_token_pool.device))
        if mode == "anchor": return self.a_token_pool[indices]
        else: return self.t_token_pool[indices]

    # losses
    def compute_region_loss(self, pred, 
                            target, mask):
        loss_fn = nn.CrossEntropyLoss(reduction='none',
                                      ignore_index=-1,
                                      label_smoothing=0.3)
        # remove whole-image at 0th index
        target = target[:, 1:] # (b, r)
        target = target.reshape(-1).long() # (b*r, )
        loss = loss_fn(pred, target)
        masked_loss = (loss * mask).sum() / mask.sum()
        return masked_loss
    
    def compute_region_dist_loss(self, a_preds, t_preds,
                                 gt_distortions, mask):
        loss_fn = nn.CrossEntropyLoss(reduction='none',
                                      ignore_index=-1,
                                      label_smoothing=0.5)
        # splitting anchor and target gt distortions
        a_gt = gt_distortions[:,:,0].reshape(-1).long() # (b, r, 2) -> (b*r,)
        t_gt = gt_distortions[:,:,1].reshape(-1).long() # (b, r, 2) -> (b*r,)
        loss_a = loss_fn(a_preds, a_gt)
        loss_t = loss_fn(t_preds, t_gt)
        masked_loss_a = (loss_a * mask).sum() / mask.sum()
        masked_loss_t = (loss_t * mask).sum() / mask.sum()
        return (masked_loss_a + masked_loss_t) / 2
    
    def compute_severity_loss(self, a_preds, t_preds,
                              gt_severities, mask):
        loss_fn = nn.CrossEntropyLoss(reduction='none',
                                      ignore_index=-1,
                                      label_smoothing=0.5)
        a_gt = gt_severities[:,:,0].reshape(-1).long() # (b, r) -> (b*r,)
        t_gt = gt_severities[:,:,1].reshape(-1).long() # (b, r) -> (b*r,)
        loss_a = loss_fn(a_preds, a_gt)
        loss_t = loss_fn(t_preds, t_gt)
        masked_loss_a = (loss_a * mask).sum() / mask.sum()
        masked_loss_t = (loss_t * mask).sum() / mask.sum()
        return (masked_loss_a + masked_loss_t) / 2

    def compute_score_regression_loss(self, a_preds, t_preds,
                                      gt_scores, mask):
        loss_fn = nn.L1Loss(reduction='none')
        a_gt = gt_scores[:,:,0].reshape(-1) # (b, r, 2) -> (b*r,)
        t_gt = gt_scores[:,:,1].reshape(-1) # (b, r, 2) -> (b*r,)
        loss_a = loss_fn(a_preds.squeeze(1), a_gt)
        loss_t = loss_fn(t_preds.squeeze(1), t_gt)
        masked_loss_a = (loss_a * mask).sum() / mask.sum() 
        masked_loss_t = (loss_t * mask).sum() / mask.sum() 
        return (masked_loss_a + masked_loss_t) / 2
    
    def forward(self, imgA, imgT,
                anchor_mask, target_mask,
                gt_severities, gt_distortions,
                gt_comparisons, gt_scores,
                region_mask_flags): # region_mask_flags: (b*r, ) contains mask for padded regions
        """
            This method extracts features from masked regions of two input images (anchor 
            and target), projects these features into a common space, decodes them, and 
            generates multiple predictions related to region and scene comparisons. It 
            then computes losses based on the ground truth labels for training.

            Args:
                imgA (torch.Tensor): The anchor input image tensor of shape (B, C, H, W).
                imgT (torch.Tensor): The target input image tensor of shape (B, C, H, W).
                anchor_mask (list of torch.Tensor): List of masks indicating regions of 
                    interest in imgA. Each mask is typically (H, W).
                target_mask (list of torch.Tensor): List of masks indicating regions of 
                    interest in imgT.
                gt_severities (torch.Tensor): Ground truth severity labels for regions.
                gt_distortions (torch.Tensor): Ground truth distortion labels for regions.
                gt_comparisons (torch.Tensor): Ground truth comparison labels between 
                    corresponding regions/scenes in imgA and imgT.
                gt_scores (torch.Tensor): Ground truth continuous quality scores for regions.

            Returns:
                preds (list): List of prediction tensors for region comparison, scene 
                    comparison, distortions, severities, and scores for both images.
                losses (list): List of computed loss values corresponding to each prediction.
                imgA/B_detections (list): Regions for imgA/imgT
        """

        valid_indices = helper.get_valid_indices_from_padded(anchor_mask)
        imgA = imgA[valid_indices] # (b, 3, h, w)
        imgT = imgT[valid_indices] # (b, 3, h, w)
        gt_severities = gt_severities[valid_indices] # (b, r, 2)
        gt_distortions = gt_distortions[valid_indices] # (b, r, 2)
        gt_comparisons = gt_comparisons[valid_indices] # (b, r+1)
        gt_scores = gt_scores[valid_indices] # (b, r, 2)
        anchor_mask = anchor_mask[valid_indices] # (b, r, h, w)
        target_mask = target_mask[valid_indices] # (b, r, h, w)

        # getting features
        imgA_fts, imgA_mask_fts = self.extract_mask_img_features(imgA, anchor_mask, mode="anchor")
        imgT_fts, imgT_mask_fts = self.extract_mask_img_features(imgT, target_mask, mode="target")
        # imgA_fts: (b, d, h//14, w//14) | imgA_mask_fts: (b, r, d, h//14, w//14)
        # imgT_fts: (b, d, h//14, w//14) | imgT_mask_fts: (b, r, d, h//14, w//14)
        
        imgA_proj_fts, imgT_proj_fts = self.vision_projection(imgA_fts, imgT_fts) # imgA_proj_fts: (b, d, h//14, w//14) | imgT_proj_fts: (b, d, h//14, w//14)
        region_featsA = self.anchor_decoder(imgT_proj_fts,
                                            imgA_mask_fts).reshape(-1, self.dim) # (b, r, d) -> (b*r, d) 
        region_featsT = self.target_decoder(imgA_proj_fts,
                                            imgT_mask_fts).reshape(-1, self.dim) # (b, r, d) -> (b*r, d)
        
        # Forward pass through the prediction heads
        region_comparison_outputs = self.region_prediction(region_featsA) # (b*r, 5)
        # attributes are for each individual image in the pair
        a_distoriton_pred_outputs = self.a_distortion_prediction(region_featsA) # (b*r, 15)
        t_distortion_pred_outputs = self.t_distortion_prediction(region_featsT) # (b*r, 15)
        a_sev_preds_outputs = self.a_severity_prediction(region_featsA) # (b*r, 4)
        t_sev_preds_outputs = self.t_severity_prediction(region_featsT) # (b*r, 4)
        a_score_preds_outputs = self.a_score_regressor(region_featsA) # (b*r, 1)
        t_score_preds_outputs = self.t_score_regressor(region_featsT) # (b*r, 1)
        
        # compute losses
        # two comparison losses (which region/scene is better?)
        # region/scene relationship
        region_loss = self.compute_region_loss(region_comparison_outputs,
                                               gt_comparisons,
                                               region_mask_flags)
        # three per-region losses (what is the distortion/severity/score of each region?)
        # region attributes
        region_dist_loss = self.compute_region_dist_loss(a_distoriton_pred_outputs,
                                                         t_distortion_pred_outputs,
                                                         gt_distortions,
                                                         region_mask_flags)
        severity_loss = self.compute_severity_loss(a_sev_preds_outputs,
                                                   t_sev_preds_outputs,
                                                   gt_severities,
                                                   region_mask_flags)
        score_pred_loss = self.compute_score_regression_loss(a_score_preds_outputs,
                                                            t_score_preds_outputs,
                                                            gt_scores,
                                                            region_mask_flags)
        
        preds = [region_comparison_outputs, a_distoriton_pred_outputs, 
                 t_distortion_pred_outputs, a_sev_preds_outputs, 
                 t_sev_preds_outputs, a_score_preds_outputs, 
                 t_score_preds_outputs]
        losses = [region_loss, region_dist_loss,
                  severity_loss, score_pred_loss]
        return preds, losses, region_mask_flags