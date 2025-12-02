import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Calcula a média e o desvio padrão da melhor acurácia de um arquivo de saída.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Caminho para o arquivo .out contendo os resultados.")
    
    args = parser.parse_args()
    file_path = args.file

    if not os.path.exists(file_path):
        print(f"Erro: Arquivo não encontrado em {file_path}")
        return

    accuracies = []
    with open(file_path, 'r') as f:
        for line in f:
            if 'Best accuracy.' in line:
                try:
                    # A próxima linha deve conter o valor da acurácia
                    accuracy_line = next(f).strip()
                    accuracies.append(float(accuracy_line))
                except (StopIteration, ValueError) as e:
                    print(f"Aviso: Não foi possível extrair a acurácia após 'Best accuracy.' na linha: {line.strip()} - Erro: {e}")

    if accuracies:
        mean_acc = np.mean(accuracies) * 100
        std_acc = np.std(accuracies) * 100
        print(f"Acurácias encontradas: {accuracies}")
        print(f"Média da Acurácia: {mean_acc:.2f}%")
        print(f"Desvio Padrão da Acurácia: {std_acc:.2f}%")
    else:
        print("Nenhuma acurácia 'Best accuracy.' encontrada no arquivo.")

if __name__ == "__main__":
    main()


