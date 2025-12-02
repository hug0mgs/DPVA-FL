# dataset/generate_MNIST.py

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms

# Adicionado para poder importar de um diretório pai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 20
# O nome do dataset agora é uma variável para criar a pasta correta.
DATASET_NAME = "MNIST"

def generate_dataset(num_clients, niid, balance, partition, alpha):
    """
    Gera e particiona o dataset MNIST, salvando tudo dentro de uma pasta 'MNIST'.
    """
    # =================================================================================================
    # ================================ AQUI ESTÁ A PRINCIPAL MUDANÇA ==================================
    # =================================================================================================
    # O caminho base agora é a pasta 'dataset'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # O diretório de saída é uma nova pasta com o nome do dataset dentro da pasta 'dataset'
    output_dir = os.path.join(base_dir, DATASET_NAME)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório criado: {output_dir}")
    
    # Todos os caminhos agora apontam para dentro da nova pasta 'MNIST'
    config_path = os.path.join(output_dir, "config.json")
    train_path = os.path.join(output_dir, "train/")
    test_path = os.path.join(output_dir, "test/")
    raw_data_path = os.path.join(output_dir, "rawdata")
    # =================================================================================================

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, alpha):
        return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=raw_data_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=raw_data_path, train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Número de classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid=niid, balance=balance, partition=partition, class_per_client=2, alpha=alpha)
    train_data, test_data = split_data(X, y)
    
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, alpha)

if __name__ == "__main__":
    niid = True if len(sys.argv) > 1 and sys.argv[1] == "noniid" else False
    balance = True if len(sys.argv) > 2 and sys.argv[2] == "balance" else False
    partition = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else 'dir'
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    print(f"Gerando {DATASET_NAME}: niid={niid}, balance={balance}, partition={partition}, alpha={alpha}")
    generate_dataset(num_clients, niid, balance, partition, alpha)
