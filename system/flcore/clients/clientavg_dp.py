# flcore/clients/clientavg_dp.py

import time
import torch
import numpy as np
from flcore.clients.clientbase import Client
from opacus.utils.batch_memory_manager import BatchMemoryManager
import copy

class clientAVG_DP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.args = args

    # ====================================================================
    # ======================= CORREÇÃO IMPORTANTE ========================
    # A assinatura do método foi atualizada para aceitar o argumento 'model'
    def train(self, model=None, dataloader=None, optimizer=None):
    # ====================================================================
        if model is None:
            model = self.model
        
        if dataloader is None:
            dataloader = self.load_train_data(batch_size=self.batch_size)
        
        if optimizer is None:
            optimizer = self.optimizer

        model.train()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            if self.args.dp_mode != 'none' and self.batch_size > 512:
                 with BatchMemoryManager(
                    data_loader=dataloader, 
                    max_physical_batch_size=512, 
                    optimizer=optimizer
                ) as memory_safe_dataloader:
                    for i, (x, y) in enumerate(memory_safe_dataloader):
                        if type(x) == type([]): x[0] = x[0].to(self.device)
                        else: x = x.to(self.device)
                        y = y.to(self.device)
                        
                        output = model(x)
                        loss = self.loss(output, y)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                for i, (x, y) in enumerate(dataloader):
                    if type(x) == type([]): x[0] = x[0].to(self.device)
                    else: x = x.to(self.device)
                    y = y.to(self.device)

                    output = model(x)
                    loss = self.loss(output, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        
        # Acessa o modelo original dentro do wrapper do Opacus e atualiza o modelo principal do cliente
        self.model.load_state_dict(model._module.state_dict())

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
