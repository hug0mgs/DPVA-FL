# utils/result_utils.py

import h5py
import numpy as np
import os

def average_data(algorithm="", dataset="", goal="", times=10, dp_mode='none', dirichlet_alpha=1.0):
    """
    Calcula e imprime a média e o desvio padrão da melhor acurácia entre várias execuções.
    O nome do arquivo agora inclui o modo de DP.
    """
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times, dp_mode, dirichlet_alpha)

    if not test_acc:
        print(f"Nenhum resultado encontrado para {algorithm} com dp_mode={dp_mode}. Pulando média.")
        return

    max_accuracy = []
    for i in range(len(test_acc)):
        if test_acc[i].size > 0:
            max_accuracy.append(test_acc[i].max())

    if not max_accuracy:
        print(f"Nenhuma acurácia registrada para {algorithm} com dp_mode={dp_mode}.")
        return

    print(f"\nResultados Finais para Algoritmo: {algorithm}, Modo DP: {dp_mode}")
    print("Média da melhor acurácia:", np.mean(max_accuracy))
    print("Desvio Padrão da melhor acurácia:", np.std(max_accuracy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, dp_mode='none', dirichlet_alpha=1.0):
    """
    Coleta os resultados de todas as execuções para um único algoritmo e modo de DP.
    """
    test_acc = []
    alpha_str = str(round(dirichlet_alpha, 1)).replace('.', '_')
    for i in range(times):
        file_name = f"{dataset}_{algorithm}_{goal}_{dp_mode}_alpha{alpha_str}_{i}"
        
        try:
            acc_data = read_data_then_delete(file_name, delete=False)
            if acc_data is not None:
                test_acc.append(np.array(acc_data))
        except FileNotFoundError:
            print(f"Arquivo não encontrado, pulando: {file_name}.h5")
            continue
            
    return test_acc


def read_data_then_delete(file_name, delete=False):
    """
    Lê os dados de um arquivo H5. Retorna None se o arquivo não existir.
    """
    # --- CORREÇÃO PRINCIPAL ---
    # Remove o caminho "../results/" para procurar o arquivo no diretório atual,
    # que é onde os servidores estão salvando os resultados.
    file_path = file_name + ".h5"
    # --------------------------

    if not os.path.exists(file_path):
        # Adicionado para evitar confusão: informa o caminho completo que não foi encontrado.
        print(f"Aviso: Arquivo de resultado não encontrado em '{os.path.abspath(file_path)}'")
        return None

    with h5py.File(file_path, 'r') as hf:
        if 'rs_test_acc' in hf:
            rs_test_acc = np.array(hf.get('rs_test_acc'))
        else:
            rs_test_acc = []

    if delete:
        os.remove(file_path)
    
    print(f"Lido arquivo: {file_path}, Comprimento da Acurácia: {len(rs_test_acc)}")

    return rs_test_acc
