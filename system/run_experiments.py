import subprocess
import sys
import os
import numpy as np
import torch
from torchvision import datasets, transforms

# Adiciona o diretório atual ao sys.path para encontrar módulos locais como data_utils
# Isso é necessário porque o Python pode não procurar no diretório atual automaticamente.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa as funções necessárias do data_utils.py (presumindo que está no mesmo diretório)
from utils.data_utils import separate_data, split_data, save_file

# --- Lógica de generate_data.py ---

def load_dataset(dataset_name):
    """Carrega o dataset especificado."""
    if dataset_name == 'Cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # O PyTorch precisa de um diretório 'data' para baixar o dataset
        os.makedirs('./data', exist_ok=True)
        
        # Carrega o dataset completo (treino + teste) para particionamento
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Garante que targets sejam listas ou arrays antes de concatenar e converte explicitamente
        train_targets = np.array(train_set.targets)
        test_targets = np.array(test_set.targets)
        
        X_combined = np.concatenate([train_set.data, test_set.data], axis=0)
        y_combined = np.concatenate([train_targets, test_targets], axis=0)
        
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

    X_content, y_label = data
    y_label = np.array(y_label) # Força ser array numpy
    data = (X_content, y_label) # Reempacota
    
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

# --- Fim da Lógica de generate_data.py ---

def run_command(command):
    """Executa um comando no shell e verifica se há erros."""
    print(f'--- Executando: {" ".join(command)} ---')
    try:
        # Usamos check=True para garantir que um erro seja levantado se o comando falhar
        subprocess.run(command, check=True, text=True)
        print(f'--- Concluído: {" ".join(command)} ---\n')
    except subprocess.CalledProcessError as e:
        print(f'!!!!!! ERRO ao executar o comando: {" ".join(command)} !!!!!!', file=sys.stderr)
        print(f'Código de saída: {e.returncode}', file=sys.stderr)
        print(f'Saída de erro:\n{e.stderr}', file=sys.stderr)
        sys.exit(1) # Termina o script se um dos comandos falhar
    except FileNotFoundError:
        print(f'!!!!!! ERRO: Comando \'python\' não encontrado. Verifique se o Python está no PATH do sistema.', file=sys.stderr)
        sys.exit(1)

def main():
    """Define e executa a sequência de experimentos para a tese."""
    
    # Valores de alpha para o experimento
    alphas = [0.0, 1.0, 5.0]
    
    # Lista de experimentos a serem executados
    # Cada tupla contém os argumentos para o main.py
    experiments = [
        # --- FedAvg ---
        ("FedAvg", "none"),
        ("FedAvg", "fixed"),
        ("FedAvg", "adaptive"),
        # --- SCAFFOLD ---
        ("SCAFFOLD", "none"),
        ("SCAFFOLD", "fixed"),
        ("SCAFFOLD", "adaptive"),
        # --- FedALA ---
        ("FedALA", "none"),
        ("FedALA", "fixed"),
        ("FedALA", "adaptive"),
    ]

    print("==========================================================")
    print("INICIANDO BATERIA DE EXPERIMENTOS PARA DISSERTAÇÃO DE MESTRADO")
    print("==========================================================")

    total_experiments = len(alphas) * len(experiments)
    current_experiment = 0

    for alpha in alphas:
        # 1. Gerar os dados para o alpha atual (Lógica incorporada)
        generate_data(dataset_name="Cifar10", num_clients=20, alpha=alpha)

        # 2. Executar os experimentos de treinamento
        for algo, dpm in experiments:
            current_experiment += 1
            print(f"\n[ETAPA {current_experiment}/{total_experiments}] Executando {algo} com DP mode: {dpm} e alpha={alpha}...")
            command = [
                "python", "main.py",
                "-algo", algo,
                "-dpm", dpm,
                "-data", "Cifar10",
                "-dal", str(alpha) # Passa o alpha para o main.py (renomeado para evitar conflito)
            ]
            run_command(command)

    print("==========================================================")
    print("TODOS OS TREINAMENTOS FORAM CONCLUÍDOS.")
    print("==========================================================")

    # Executa o script de plotagem
    print("\n[ETAPA FINAL] Gerando gráficos comparativos para o dataset CIFAR10...")
    plot_command = ["python", "plot_tese.py", "-data", "CIFAR10"]
    run_command(plot_command)

    print("\nProcesso finalizado com sucesso!")
    print("Os gráficos foram salvos no formato SVG no diretório 'figs/'.")

if __name__ == "__main__":
    main()
