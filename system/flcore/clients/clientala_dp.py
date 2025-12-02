# flcore/clients/clientala_dp.py

import time
import torch
import numpy as np
from flcore.clients.clientbase import Client
from opacus.utils.batch_memory_manager import BatchMemoryManager
import copy

class clientALA_DP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.args = args
        self.delta_model = None # Atributo para guardar a atualização

    def train(self, model=None, dataloader=None, optimizer=None):
        if model is None:
            model = self.model
        
        if dataloader is None:
            dataloader = self.load_train_data(batch_size=self.batch_size)
        
        if optimizer is None:
            optimizer = self.optimizer

        # Guardar o estado do modelo ANTES do treinamento
        model_before_train = copy.deepcopy(model)

        model.train()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(dataloader):
                if type(x) == type([]): x[0] = x[0].to(self.device)
                else: x = x.to(self.device)
                y = y.to(self.device)
                
                output = model(x)
                loss = self.loss(output, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
   
        # Calcular e armazenar a atualização (delta) 
        # Cada cliente, de forma independente, calcula a sua própria atualização (delta) após o treino local.
        self.delta_model = copy.deepcopy(model) # Modelo treinado
        for param_delta, param_before in zip(self.delta_model.parameters(), model_before_train.parameters()):
            param_delta.data -= param_before.data

        # Atualizar o modelo principal do cliente
        self.model.load_state_dict(model._module.state_dict())


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
