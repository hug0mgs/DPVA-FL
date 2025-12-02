import h5py
import numpy as np

# Substitua pelos nomes dos seus arquivos .h5
arquivos = [
    "Cifar10_FedAvg_test_fixed_0.h5",
    "Cifar10_FedAvg_test_adaptive_0.h5",
    "Cifar10_SCAFFOLD_test_fixed_0.h5",
    "Cifar10_SCAFFOLD_test_adaptive_0.h5",
    "Cifar10_FedALA_test_fixed_0.h5",
    "Cifar10_FedALA_test_adaptive_0.h5"
]

for arq in arquivos:
    with h5py.File(arq, "r") as f:
        for k in f.keys():
            print(f"{arq} - {k}:")
            print(np.array(f[k]))


