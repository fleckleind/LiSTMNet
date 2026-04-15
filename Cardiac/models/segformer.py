import torch
import torch.nn as nn
from einops import rearrange
from transformers import SegformerForSemanticSegmentation


class SegFormer(nn.Module):
    def __init__(self, num_classes):
        super(SegFormer, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "MemSAM/models/segformer_b3", local_files_only=True)
        self.model.num_labels = num_classes

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        logits = self.model(x).logits
        out = nn.functional.interpolate(
            logits, size=(h,w), mode="bilinear", align_corners=False)
        out = rearrange(out, "(b t) c h w -> b t c h w", b=b)
        return out
    

if __name__ == "__main__":
    x = torch.rand(4, 10, 3, 256, 256)
    model = SegFormer(num_classes=1)
    print(model(x).shape)
