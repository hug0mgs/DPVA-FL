import argparse
import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms
from data_utils import separate_data, split_data, save_file

# Adicione o caminho para o diretório 'upload' ao sys.path para importar data_utils
sys.path.append(os.path.join(os.getcwd(), 'upload'))

def load_dataset(dataset_name):
    """Carrega o dataset especificado."""
    if dataset_name == 'Cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Carrega o dataset completo (treino + teste) para particionamento
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Combina os dados para particionamento
        X_combined = np.concatenate([train_set.data, test_set.data], axis=0)
        y_combined = np.concatenate([train_set.targets, test_set.targets], axis=0)
        
        # Ajusta o formato para (N, C, H, W) como esperado pelo PyTorch
        X_combined = np.transpose(X_combined, (0, 3, 1, 2))
        
        num_classes = 10
        
        return (X_combined, y_combined), num_classes
    else:
        raise NotImplementedError(f"Dataset {dataset_name} não suportado para geração de dados.")

def generate_data(dataset_name, num_clients, alpha):
    """Gera e salva os dados particionados."""
    print(f"--- Iniciando geração de dados para {dataset_name} com alpha={alpha} ---")
    
    # 1. Carregar o dataset
    data, num_classes = load_dataset(dataset_name)
    
    # 2. Definir caminhos de salvamento
    base_dir = os.path.join('dataset', dataset_name, f'alpha_{alpha}')
    config_path = os.path.join(base_dir, 'config.json')
    train_path = os.path.join(base_dir, 'train')
    test_path = os.path.join(base_dir, 'test')
    
    # Cria os diretórios se não existirem
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # 3. Particionar os dados
    # Usamos 'dir' para Dirichlet e niid=True para não-IID
    X_clients, y_clients, statistic = separate_data(
        data=data, 
        num_clients=num_clients, 
        num_classes=num_classes, 
        niid=True, 
        balance=False, 
        partition='dir', 
        alpha=alpha
    )
    
    # 4. Dividir em treino/teste
    train_data, test_data = split_data(X_clients, y_clients)
    
    # 5. Salvar os arquivos
    save_file(
        config_path=config_path, 
        train_path=train_path, 
        test_path=test_path, 
        train_data=train_data, 
        test_data=test_data, 
        num_clients=num_clients, 
        num_classes=num_classes, 
        statistic=statistic, 
        niid=True, 
        balance=False, 
        partition='dir', 
        alpha=alpha
    )
    
    print(f"--- Geração de dados concluída para alpha={alpha} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10", help="Nome do dataset.")
    parser.add_argument('-nc', "--num_clients", type=int, default=20, help="Número total de clientes.")
    parser.add_argument('-al', "--alpha", type=float, required=True, help="Parâmetro alpha para distribuição Dirichlet.")
    
    args = parser.parse_args()
    
    # O PyTorch precisa de um diretório 'data' para baixar o dataset
    os.makedirs('./data', exist_ok=True)
    
    generate_data(args.dataset, args.num_clients, args.alpha)
