# flcore/clients/clientscaffold_dp.py

import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import SCAFFOLDOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager
import copy

class clientSCAFFOLD_DP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Otimizador SCAFFOLD é sempre usado.
        self.optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.args = args

        self.client_c = [torch.zeros_like(param) for param in self.model.parameters()]
        self.global_c = None
        self.global_model_state = None
        self.delta_c = None
        self.delta_y = None

    def set_parameters(self, model, global_c):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        self.global_model_state = [param.data.clone() for param in model.parameters()]
        self.global_c = [c.data.clone() for c in global_c]

    # ====================================================================
    # ======================= CORREÇÃO IMPORTANTE ========================
    # A assinatura do método foi atualizada para aceitar o modelo privado
    def train(self, model=None, dataloader=None, optimizer=None):
    # ====================================================================
        if dataloader is None:
            dataloader = self.load_train_data(batch_size=self.batch_size)
        
        # O modelo passado pelo servidor DP é o que usamos para o treinamento
        train_model = model if model is not None else self.model
        
        train_model.train()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(dataloader):
                if type(x) == type([]): x[0] = x[0].to(self.device)
                else: x = x.to(self.device)
                y = y.to(self.device)
                
                # O otimizador privado (do Opacus) faz o forward/backward/step
                if optimizer:
                    optimizer.zero_grad()
                    output = train_model(x)
                    loss = self.loss(output, y)
                    loss.backward()
                    
                    optimizer.step()

                    # Correção de gradiente do SCAFFOLD APÓS o passo do otimizador
                    for p, c_global, c_local in zip(train_model.parameters(), self.global_c, self.client_c):
                        if p.grad is not None:
                            p.grad.data += c_global.to(self.device) - c_local.to(self.device)
                else: # Modo não-DP
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.loss(output, y)
                    loss.backward()
                    self.optimizer.step(self.global_c, self.client_c)

        # Atualiza o modelo principal do cliente com os pesos do modelo treinado
        if hasattr(train_model, '_module'):
            self.model.load_state_dict(train_model._module.state_dict())
        else:
            self.model.load_state_dict(train_model.state_dict())

        self.update_and_get_deltas(max_local_epochs, len(dataloader))

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def update_and_get_deltas(self, num_epochs, num_batches):
        self.delta_y = []
        self.delta_c = []
        
        for local_param, global_param in zip(self.model.parameters(), self.global_model_state):
            self.delta_y.append(local_param.data - global_param.data)
            
        new_client_c = []
        if num_epochs > 0 and num_batches > 0:
            factor = 1 / (num_epochs * num_batches * self.learning_rate)
            for c_i, c_global, y_i, x in zip(self.client_c, self.global_c, self.model.parameters(), self.global_model_state):
                new_c = c_i.data - c_global.data + (x.data - y_i.data) * factor
                new_client_c.append(new_c)
                self.delta_c.append(new_c - c_i.data)
        else:
            self.delta_c = [torch.zeros_like(c) for c in self.client_c]
            new_client_c = self.client_c

        self.client_c = new_client_c
