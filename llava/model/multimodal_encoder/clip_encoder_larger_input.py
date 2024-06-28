import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoModel

from transformers import CLIPPreTrainedModel
import math


class CLIPVisionEmbeddingsLargerInput(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    #     batch_size = pixel_values.shape[0]
    #     target_dtype = self.patch_embedding.weight.dtype
    #     patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
    #     patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    #     class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    #     embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    #     embeddings = embeddings + self.position_embedding(self.position_ids)
    #     return embeddings
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0] # (bs, 3, h=336, w=336)
        target_dtype = self.patch_embedding.weight.dtype
        # pachify by using conv2D
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # (bsz, emb_dim, p_size=24, p_size=24) -> (bsz, emb_dim, 24*24=576) -> (bsz, 576, emb_dim)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1) # (bsz, 1, emb_dim=1024)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        

        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, pixel_values.shape[2], pixel_values.shape[3])
        # use interpolation


        return embeddings # (bs, seqlen, dim)


    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        embeddings: (bsz, seqlen, dim)
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        # make this in init
        bsz = embeddings.shape[0]
        positional_embedding = self.position_embedding(self.position_ids)
        # positional_embedding: (seqlen, dim)


        num_patches = embeddings.shape[1] - 1

        # pos_embedding = positional_embedding.unsqueeze(0)
        num_positions = positional_embedding.shape[1] - 1 # number of positions of inputs


        if num_patches == num_positions and height == width: # if it matches
            return positional_embedding

        assert num_positions < num_patches, f'there must be an increased number of positions, got {num_positions} > {num_patches}'       


        class_pos_embed = positional_embedding[:, [0]] # (bsz, 1)
        patch_pos_embed = positional_embedding[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_size
        w0 = width // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        # patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(bsz, -1, dim) # (bsz, dim, h, w) -> (bsz, h, w, dim) -> (bsz, seqlen=h*w, dim)
        output = torch.cat((class_pos_embed, patch_pos_embed), dim=1)

        return output


'''
overwrite:
CLIPVisionEmbeddings
embeddings = CLIPVisionEmbeddingsLargerInput()
CLIPVisionModel.vision_model.embeddings = embeddings
'''

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        # self.select_layer = args.mm_vision_select_layer
        self.select_layer = getattr(args, 'mm_vision_select_layer', -2)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        print(f'loading vision model from {self.vision_tower_name}')
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        if 'clip' in self.vision_tower_name.lower():
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.vision_model.embeddings = CLIPVisionEmbeddingsLargerInput(self.vision_tower.config)

        elif 'internvit' in self.vision_tower_name.lower():
            self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        else:
            raise ValueError(f'Please implement the loading of vision encoder here')
        
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch': # default excluding cls_patch
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



if __name__ == '__main__':
    vision_tower = CLIPVisionTower('/wangbenyou/guimingchen/models/clip_vit_large_patch14_336', None)
    from PIL import Image
    import pdb

    # vision_tower.load_model()
    processor = vision_tower.image_processor

    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    image = Image.open('/wangbenyou/guimingchen/workspaces/vllm/data/images_miniset/openai_wewb_screenshot.png').convert('RGB')
    
    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    image = processor.preprocess(image, return_tensors='pt')['pixel_values'] # (1, 3, 336, 336)

    vision_tower(image)
