import time
import torch
import numpy as np
from flcore.clients.clientavg_dp import clientAVG_DP
from flcore.servers.serverbase_dp import ServerDPBase # Importa a nova classe base
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import copy
import h5py
import os

class FedAvg_DP(ServerDPBase):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientAVG_DP)

        # Inicializa privacy_budget_spent após os clientes serem setados
        self.privacy_budget_spent = {client.id: 0.0 for client in self.clients}

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Modo de Privacidade Diferencial: {self.dp_mode.upper()}")
        print("Finished creating server and clients for FedAvg with Differential Privacy.")

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                if self.dp_mode == 'adaptive':
                    print(f"Épsilon Alvo para esta Rodada: {self.current_epsilon_per_round:.4f}")
                self.evaluate()

            total_epsilon_this_round = 0
            uploaded_updates = []
            
            if self.dp_mode != 'none':
                num_rounds = self.global_rounds if self.global_rounds > 0 else 1
                if self.dp_mode == 'adaptive':
                    target_epsilon_for_round = self.current_epsilon_per_round
                else:
                    target_epsilon_for_round = self.dp_epsilon / num_rounds

                for client in self.selected_clients:
                    client.model.train()
                    temp_model = copy.deepcopy(client.model)
                    optimizer = torch.optim.SGD(temp_model.parameters(), lr=self.learning_rate)
                    dataloader = client.load_train_data(batch_size=self.batch_size)
                    
                    privacy_engine = PrivacyEngine(secure_mode=self.args.secure_mode)
                    
                    private_model, private_optimizer, private_dataloader = privacy_engine.make_private_with_epsilon(
                        module=temp_model,
                        optimizer=optimizer,
                        data_loader=dataloader,
                        target_epsilon=target_epsilon_for_round,
                        target_delta=self.dp_delta,
                        epochs=self.local_epochs,
                        max_grad_norm=self.dp_max_grad_norm,
                    )
                    
                    client.train(model=private_model, dataloader=private_dataloader, optimizer=private_optimizer)
                    
                    epsilon_spent = privacy_engine.get_epsilon(self.dp_delta)
                    self.privacy_budget_spent[client.id] += epsilon_spent
                    total_epsilon_this_round += epsilon_spent
                    print(f"Cliente {client.id}: Gasto nesta rodada: ε = {epsilon_spent:.4f} / Orçamento Total Gasto: ε = {self.privacy_budget_spent[client.id]:.4f}")
                    
                    # Calcular a atualização (delta) para o cálculo da variância
                    update = []
                    for global_param, client_param in zip(self.global_model.parameters(), private_model._module.parameters()):
                        update.append((global_param.data - client_param.data).view(-1))
                    uploaded_updates.append(torch.cat(update))

            else:
                for client in self.selected_clients:
                    client.train()

            self.receive_models()
            self.aggregate_parameters()

            if self.dp_mode != 'none' and self.rs_test_acc:
                avg_epsilon_this_round = total_epsilon_this_round / len(self.selected_clients) if self.selected_clients else 0
                self.calculate_and_update_metrics(self.rs_test_acc[-1], avg_epsilon_this_round, uploaded_updates)
            
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        if self.rs_test_acc: print(max(self.rs_test_acc))
        
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1: print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def save_results(self):
        # Garante que o alpha seja formatado como X_0
        alpha_str = str(int(self.args.dirichlet_alpha)) + '_0' if self.args.dirichlet_alpha == int(self.args.dirichlet_alpha) else str(self.args.dirichlet_alpha).replace('.', '_')
        # O nome do arquivo deve ser: Dataset_Algoritmo_test_DPMode_alphaX_0_0.h5
        algo = f"{self.dataset}_{self.algorithm}_{self.goal}_{self.args.dp_mode}_alpha{alpha_str}_0_0"
        result_path = "."
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        file_path = os.path.join(result_path, f"{algo}.h5")
        print(f"Salvando resultados em: {file_path}")

        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            
            if self.dp_mode != 'none':
                hf.create_dataset('rs_variance', data=self.rs_variance)
                hf.create_dataset('rs_epsilon_per_round', data=self.rs_epsilon_per_round)
                hf.create_dataset('rs_cic', data=self.rs_cic)
                hf.create_dataset('rs_cep', data=self.rs_cep)
                hf.create_dataset('rs_ica', data=self.rs_ica)


