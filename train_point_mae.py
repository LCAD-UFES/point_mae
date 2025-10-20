import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from pathlib import Path
import util.misc as misc

# Importa o modelo PointMAE do script que criamos anteriormente
from point_mae import PointMAE

# =================================================================================
# Funções de Leitura/Escrita para o formato PLY
# =================================================================================

def save_ply(points, filename):
    """Salva uma nuvem de pontos (N, C) em um arquivo ASCII .ply."""
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property float intensity
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt='%.6f')

# =================================================================================
# Dataset e Dataloader para Nuvens de Pontos
# =================================================================================

def create_dummy_dataset(path='dummy_point_cloud_data_ply', num_files=100, num_points=1024):
    """Cria um diretório com arquivos .ply de nuvens de pontos aleatórias para treinamento."""
    print(f"Verificando/Criando dataset de exemplo em '{path}'...")
    os.makedirs(path, exist_ok=True)
    if len(os.listdir(path)) >= num_files:
        print("Dataset de exemplo já existe.")
        return

    print(f"Gerando {num_files} arquivos de nuvem de pontos .ply...")
    for i in range(num_files):
        file_path = os.path.join(path, f'sample_{i}.ply')
        
        # Gera 4 canais: x, y, z, intensidade
        xyz = np.random.randn(num_points, 3)
        intensity = np.random.rand(num_points, 1)
        point_cloud = np.hstack([xyz, intensity]).astype(np.float32)
        
        save_ply(point_cloud, file_path)
    print("Dataset de exemplo criado com sucesso.")

class PointCloudDataset(Dataset):
    """Dataset para carregar arquivos de nuvem de pontos (formato .bin)."""
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_list = [f for f in os.listdir(data_path) if f.endswith('.bin')]
        self.num_points = 78596 #8192

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_list[idx])
        """Lê um arquivo .bin e retorna os pontos como um array numpy."""
        # 1. Carregue sua nuvem de pontos
        # Supondo 4 colunas (XYZI)
        point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        # 2. Pegue apenas XYZ
        point_cloud = point_cloud[:, :3] # Agora é (ex: 117038, 3)

        num_atual_pontos = len(point_cloud)

        # 3. FAÇA A AMOSTRAGEM (SAMPLING)
        # Como N (8192) é sempre < num_atual_pontos (min 78k),
        # este 'if' sempre será verdadeiro.
        if num_atual_pontos > self.num_points:
            # Escolhe 'num_points' índices aleatórios sem reposição
            indices = np.random.choice(
                num_atual_pontos, 
                self.num_points, 
                replace=False
            )
            point_cloud = point_cloud[indices, :]
            
        # O 'elif' para padding (preenchimento) nunca será chamado
        # se você usar o dataset KITTI, mas é bom ter por segurança.
        elif num_atual_pontos < self.num_points:
            padding_size = self.num_points - num_atual_pontos
            padding = np.zeros((padding_size, 3), dtype=point_cloud.dtype)
            point_cloud = np.vstack((point_cloud, padding))
        
        # 4. Retorne o tensor de tamanho fixo [8192, 3]
        # return torch.from_numpy(point_cloud).float()
        return torch.from_numpy(point_cloud)


# =================================================================================
# Argumentos e Loop de Treinamento
# =================================================================================

def get_args_parser():
    parser = argparse.ArgumentParser('Point-MAE Pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05)

    # --- Parâmetros do Modelo ---
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--n_groups', default=64, type=int, help='Número de patches')
    parser.add_argument('--group_size', default=32, type=int, help='Tamanho de cada patch (k do k-NN)')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # --- Parâmetros do Dataset ---
    parser.add_argument('--data_path', default='dummy_point_cloud_data_ply/', type=str,
                        help='Caminho para o dataset')
    parser.add_argument('--num_points', default=1024, type=int, help='Número de pontos por nuvem')

    # --- Outros ---
    parser.add_argument('--device', default='cuda', help='device to use for training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--output_dir', default='./output_dir_pointmae', help='path where to save, empty for no saving')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

def main(args):
    # --- Setup inicial ---
    misc.init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Usando dispositivo: {device}")

    # Cria o diretório de saída
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # # --- Dataset e DataLoader ---
    # create_dummy_dataset(path=args.data_path, num_points=args.num_points)
    dataset_train = PointCloudDataset(args.data_path)
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Modelo e Otimizador ---
    model = PointMAE(
        embed_dim=args.embed_dim,
        n_groups=args.n_groups,
        group_size=args.group_size,
        in_chans=3, # <--- AQUI: Informando ao modelo que a entrada tem 4 canais
        encoder_depth=6,
        encoder_num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=4
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Modelo e otimizador criados.")

    # --- Loop de Treinamento ---
    print(f"Iniciando treinamento por {args.epochs} epochs.")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, point_clouds in enumerate(data_loader_train):
            point_clouds = point_clouds.to(device)

            # Forward pass
            loss, _, _ = model(point_clouds, mask_ratio=args.mask_ratio)

            # Backward pass e otimização
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"  Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(data_loader_train)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(data_loader_train)
        print(f"Epoch {epoch+1} concluída. Perda Média: {avg_loss:.4f}")

        # Salvando o checkpoint do modelo
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-epoch{epoch+1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            }, checkpoint_path)
            print(f"Checkpoint salvo em: {checkpoint_path}")

    print("Treinamento concluído.")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
