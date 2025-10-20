
import torch
import torch.nn as nn
from functools import partial

# =================================================================================
# Funções de Agrupamento de Pontos (Point Cloud Grouping)
# =================================================================================

def farthest_point_sample(xyz, npoint):
    """
    Usa amostragem iterativa do ponto mais distante para selecionar um subconjunto de pontos.
    
    Args:
        xyz (torch.Tensor): Nuvem de pontos de entrada. Shape: (B, N, 3)
        npoint (int): Número de pontos a serem amostrados.

    Returns:
        torch.Tensor: Índices dos pontos amostrados. Shape: (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn_group(xyz, centroids_xyz, k):
    """
    Encontra os k vizinhos mais próximos para cada centroide.

    Args:
        xyz (torch.Tensor): Nuvem de pontos completa. Shape: (B, N, 3)
        centroids_xyz (torch.Tensor): Pontos centrais dos grupos. Shape: (B, M, 3)
        k (int): Número de vizinhos a serem agrupados.

    Returns:
        torch.Tensor: Pontos agrupados. Shape: (B, M, k, 3)
    """
    dist = torch.cdist(centroids_xyz, xyz)
    _, topk_indices = torch.topk(dist, k, dim=-1, largest=False)
    
    B, M, _ = centroids_xyz.shape
    B, N, C = xyz.shape
    
    # Gather points
    batch_indices = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(B, M, k)
    grouped_xyz = xyz[batch_indices, topk_indices]
    
    return grouped_xyz

# =================================================================================
# Camada de Embedding para Nuvens de Pontos
# =================================================================================

class PointPatchEmbed(nn.Module):
    """
    Transforma uma nuvem de pontos em uma sequência de tokens (embeddings de patch).
    Análogo ao `PatchEmbed` para imagens.
    """
    def __init__(self, n_groups, group_size, embed_dim, in_chans=3):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        
        # Mini-PointNet para extrair features de cada grupo
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_chans, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, points_data):
        """
        Args:
            points_data (torch.Tensor): Nuvem de pontos de entrada. Shape: (B, N, C) onde C >= 3

        Returns:
            torch.Tensor: Tokens de características. Shape: (B, n_groups, embed_dim)
            torch.Tensor: Coordenadas dos centroides (usado como pos_embed). Shape: (B, n_groups, 3)
        """
        B, N, C = points_data.shape
        xyz = points_data[..., :3] # Assume que as 3 primeiras dimensões são XYZ
        
        # 1. Agrupamento (Grouping)
        centroid_indices = farthest_point_sample(xyz, self.n_groups)
        batch_indices = torch.arange(B, device=xyz.device).view(B, 1)
        centroids_xyz = xyz[batch_indices, centroid_indices]
        
        # knn_group usa a nuvem de pontos completa (com todas as features)
        grouped_points = knn_group(points_data, centroids_xyz, self.group_size)
        
        # 2. Tokenization (Embedding)
        # Normaliza apenas as coordenadas XYZ dos pontos em cada grupo
        grouped_points_normalized = grouped_points.clone()
        grouped_points_normalized[..., :3] = grouped_points[..., :3] - centroids_xyz.unsqueeze(2)
        
        # Transpõe para o formato que a Conv1d espera: (B * M, C, k)
        grouped_points_transposed = grouped_points_normalized.permute(0, 1, 3, 2).reshape(-1, C, self.group_size)
        
        # Extrai features com o mini-PointNet
        features = self.feature_extractor(grouped_points_transposed)
        
        # Max-pooling para obter um vetor de feature por grupo
        tokens = torch.max(features, dim=2)[0]
        
        # Reshape de volta para (B, M, embed_dim)
        tokens = tokens.reshape(B, self.n_groups, -1)
        
        # O positional embedding são as próprias coordenadas dos centroides
        positional_embedding = centroids_xyz
        
        return tokens, positional_embedding

# =================================================================================
# Modelo Point-MAE
# =================================================================================

class PointMAE(nn.Module):
    def __init__(self, embed_dim=768, n_groups=128, group_size=32, in_chans=3,
                 encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4.):
        super().__init__()

        # --- Encoder ---
        self.patch_embed = PointPatchEmbed(n_groups, group_size, embed_dim, in_chans=in_chans)
        
        # Positional embedding é projetado para a dimensão do embedding
        self.pos_embed_projection = nn.Linear(3, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=encoder_num_heads, dim_feedforward=int(embed_dim * mlp_ratio)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        # --- Decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # O decoder também precisa de positional embeddings
        self.decoder_pos_embed_projection = nn.Linear(3, decoder_embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_num_heads, dim_feedforward=int(decoder_embed_dim * mlp_ratio)
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # Camada final para reconstruir as coordenadas dos pontos nos patches mascarados
        self.decoder_pred = nn.Linear(decoder_embed_dim, group_size * 3, bias=True)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, xyz, mask_ratio):
        # Gera tokens e pos_embed a partir da nuvem de pontos
        tokens, pos_embed_xyz = self.patch_embed(xyz)
        
        # Projeta o pos_embed para a dimensão correta e adiciona aos tokens
        pos_embed = self.pos_embed_projection(pos_embed_xyz)
        x = tokens + pos_embed
        
        # Mascaramento
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Aplica o encoder do Transformer
        # Transformer do PyTorch espera (seq_len, batch, dim)
        x_masked = x_masked.permute(1, 0, 2)
        encoded_tokens = self.encoder(x_masked)
        encoded_tokens = encoded_tokens.permute(1, 0, 2)
        
        return encoded_tokens, mask, ids_restore, pos_embed_xyz

    def forward_decoder(self, x, ids_restore, pos_embed_xyz):
        x = self.decoder_embed(x)
        
        # Adiciona mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # Adiciona positional embedding do decoder
        decoder_pos_embed = self.decoder_pos_embed_projection(pos_embed_xyz)
        x = x_ + decoder_pos_embed
        
        # Aplica o decoder do Transformer
        x = x.permute(1, 0, 2)
        decoded_tokens = self.decoder(x)
        decoded_tokens = decoded_tokens.permute(1, 0, 2)

        # Projeção final para reconstruir os pontos
        pred = self.decoder_pred(decoded_tokens)
        return pred

    def forward_loss(self, xyz, pred, mask):
        """
        Calcula a perda de reconstrução (MSE) nos patches mascarados.
        """
        # Precisamos obter os pontos originais dos patches para calcular a perda
        tokens, pos_embed_xyz = self.patch_embed(xyz)
        grouped_xyz = knn_group(xyz, pos_embed_xyz, self.patch_embed.group_size)
        target = grouped_xyz - pos_embed_xyz.unsqueeze(2) # Normaliza
        target = target.reshape(target.shape[0], self.patch_embed.n_groups, -1)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], perda média por patch

        loss = (loss * mask).sum() / mask.sum()  # Perda média apenas nos patches removidos
        return loss

    def forward(self, xyz, mask_ratio=0.75):
        latent, mask, ids_restore, pos_embed_xyz = self.forward_encoder(xyz, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, pos_embed_xyz)
        loss = self.forward_loss(xyz, pred, mask)
        return loss, pred, mask


# =================================================================================
# Exemplo de Uso
# =================================================================================

if __name__ == '__main__':
    # Verifica se a GPU está disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cria o modelo PointMAE
    model = PointMAE(
        embed_dim=256,
        n_groups=64,       # Número de "patches"
        group_size=32,     # Tamanho de cada "patch" (k do k-NN)
        encoder_depth=6,
        encoder_num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=4
    ).to(device)

    print("Estrutura do modelo PointMAE:")
    print(model)

    # Cria uma nuvem de pontos aleatória para teste
    # Batch size = 4, 1024 pontos por nuvem, 3 coordenadas (x, y, z)
    dummy_point_cloud = torch.randn(4, 16793, 3).to(device)
    # dummy_point_cloud = torch.randn(4, 167936, 3).to(device)

    # Realiza uma passagem para frente (forward pass)
    print("Executando forward pass...")
    loss, pred, mask = model(dummy_point_cloud)

    print(f"Forward pass concluído.")
    print(f"  - Shape da nuvem de pontos de entrada: {dummy_point_cloud.shape}")
    print(f"  - Perda (Loss): {loss.item():.4f}")
    print(f"  - Shape da predição (pontos reconstruídos): {pred.shape}")
    print(f"  - Shape da máscara: {mask.shape}")
    print(f"  - Número de patches visíveis: {int(mask.numel() - mask.sum())}")
    print(f"  - Número de patches mascarados: {int(mask.sum())}")