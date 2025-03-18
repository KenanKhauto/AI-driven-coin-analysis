import torch
import torch.nn as nn
import torch.optim as optim

class MLPFusion(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        """
        MLP-based fusion of CLIP image and text embeddings.
        :param input_dim: Input feature dimension (512 for CLIP ViT-B/32).
        :param hidden_dim: Hidden layer size.
        :param output_dim: Output embedding size after fusion.
        """
        super(MLPFusion, self).__init__()

        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)  # Concatenate img + text embeddings
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Project to shared space

    def forward(self, img_emb, txt_emb):
        """
        Forward pass to fuse image and text embeddings.
        :param img_emb: Image embedding tensor (batch_size, 512).
        :param txt_emb: Text embedding tensor (batch_size, 512).
        :return: Fused embedding (batch_size, output_dim).
        """
        x = torch.cat((img_emb, txt_emb), dim=1)  # Concatenate embeddings
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Return fused representation


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, fused_img, fused_txt, labels):
        """
        Compute contrastive loss for multimodal fusion.
        :param fused_img: Fused image embeddings.
        :param fused_txt: Fused text embeddings.
        :param labels: 1 for matching pairs, -1 for non-matching pairs.
        """
        return self.cosine_loss(fused_img, fused_txt, labels)
