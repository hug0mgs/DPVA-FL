# utils/data_utils.py

import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import torch
from collections import defaultdict

# Constantes do módulo
batch_size = 10
train_ratio = 0.75

def check(config_path, train_path, test_path, num_clients, niid=False, 
        balance=True, partition=None, alpha=100.0):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config.get('num_clients') == num_clients and \
            config.get('non_iid') == niid and \
            config.get('balance') == balance and \
            config.get('partition') == partition and \
            config.get('alpha') == alpha and \
            config.get('batch_size') == batch_size:
            print("\nDataset com a configuração desejada já existe.\n")
            return True
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return False

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None, alpha=100.0):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    dataset_content, dataset_label = data
    least_samples = int(min(batch_size / (1-train_ratio), len(dataset_label) / num_clients / 2))
    dataidx_map = {}
    if not niid:
        partition = 'pat'
        class_per_client = num_classes
    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])
        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            if len(selected_clients) == 0:
                break
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]
            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))
            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1
    elif partition == "dir":
        print(f"Gerando partição Dirichlet com alpha={alpha}")
        min_size = 0
        K = num_classes
        N = len(dataset_label)
        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Tamanho dos dados do cliente não atende ao requisito mínimo de {least_samples}. Tentando alocar novamente pela {try_cnt}ª vez.')
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                if alpha == 0.0:
                    # Caso especial para alpha=0.0: Alocação concentrada (non-IID extremo)
                    # Para cada classe, apenas um cliente recebe 100% dos dados dessa classe.
                    # Isso é feito de forma cíclica para distribuir as classes entre os clientes.
                    proportions = np.zeros(num_clients)
                    # O cliente que receberá 100% dos dados da classe k é determinado por k % num_clients
                    proportions[k % num_clients] = 1.0
                else:
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch]) if idx_batch and all(idx_batch) else 0
            try_cnt += 1
        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
    del data
    gc.collect()
    for client in range(num_clients):
        print(f"Cliente {client}\t Tamanho dos dados: {len(X[client])}\t Rótulos: ", np.unique(y[client]))
        print(f"\t\t Amostras por rótulo: ", [i for i in statistic[client]])
        print("-" * 50)
    return X, y, statistic

def split_data(X, y):
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}
    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)
        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))
    print("Número total de amostras:", sum(num_samples['train'] + num_samples['test']))
    print("Número de amostras de treino:", num_samples['train'])
    print("Número de amostras de teste:", num_samples['test'])
    print()
    del X, y
    gc.collect()
    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None, alpha=100.0):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha,
        'batch_size': batch_size, 
    }
    gc.collect()
    print("Salvando em disco.\n")
    for idx, train_dict in enumerate(train_data):
        with open(os.path.join(train_path, f"{idx}.npz"), 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(os.path.join(test_path, f"{idx}.npz"), 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)
    print("Geração do dataset finalizada.\n")

# =================================================================================================
# ================================ AQUI ESTÁ A PRINCIPAL MUDANÇA ==================================
# =================================================================================================
def read_data(dataset, idx, is_train=True):
    """
    Lê os dados de um cliente específico, agora procurando dentro da pasta do dataset.
    """
    if is_train:
        data_dir = os.path.join('../dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, 'test/')

    file_path = os.path.join(data_dir, f"{idx}.npz")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {file_path}. Verifique se o dataset '{dataset}' foi gerado.")

    with open(file_path, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data
# =================================================================================================

def read_client_data(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)
    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]

def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['file_name']
        img_label = self.dataframe.iloc[idx]['class']
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_label
