# flcore/servers/serverbase_dp.py
import time
import torch
import numpy as np
from flcore.servers.serverbase import Server
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import copy
import h5py
import os
import threading
import queue
from collections import defaultdict


class ModelDPPool:
    """Pool de modelos reutilizáveis para Differential Privacy para evitar deepcopy excessivo"""
    def __init__(self, base_model, pool_size=5, device='cpu'):
        self.pool_size = pool_size
        self.device = device
        self.available_models = queue.Queue()
        self.in_use = set()
        self.total_requests = 0
        self.pool_hits = 0

        # Criar pool inicial de modelos
        for _ in range(pool_size):
            model_copy = copy.deepcopy(base_model)
            model_copy.to(device)
            self.available_models.put(model_copy)

    def get_model(self):
        """Obter modelo do pool ou criar novo se pool estiver vazio"""
        self.total_requests += 1
        try:
            # Tentar obter do pool (não-bloqueante)
            model = self.available_models.get_nowait()
            self.pool_hits += 1
            self.in_use.add(id(model))
            return model
        except queue.Empty:
            # Pool vazio, criar novo modelo (fallback)
            print("DP Pool esgotado - criando modelo temporário")
            # Criar modelo base genérico (será substituído pelos parâmetros do cliente)
            base_placeholder = type(list(self.in_use)[0] if self.in_use else torch.nn.Module())()
            return copy.deepcopy(base_placeholder)

    def return_model(self, model, reset_params=True):
        """Retornar modelo ao pool para reutilização"""
        model_id = id(model)
        if model_id in self.in_use:
            self.in_use.remove(model_id)
            if reset_params:
                # Limpar gradientes e estado
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()
                model.eval()

            try:
                self.available_models.put_nowait(model)
            except queue.Full:
                # Pool cheio, descartar modelo
                pass

    def get_stats(self):
        """Obter estatísticas de uso do pool"""
        hit_rate = self.pool_hits / max(self.total_requests, 1)
        return {
            'total_requests': self.total_requests,
            'pool_hits': self.pool_hits,
            'hit_rate': hit_rate,
            'available': self.available_models.qsize(),
            'in_use': len(self.in_use)
        }


class AsyncLogger:
    """Sistema de logging assíncrono para reduzir overhead de prints"""
    def __init__(self, log_interval=10, verbose=False):
        self.log_buffer = []
        self.log_interval = log_interval
        self.verbose = verbose
        self.buffer_lock = threading.Lock()

    def log(self, message, force=False):
        """Adicionar mensagem ao buffer de logging"""
        with self.buffer_lock:
            self.log_buffer.append(f"[{time.time():.2f}] {message}")

        # Flush imediato se forçado ou buffer cheio
        if force or len(self.log_buffer) >= self.log_interval:
            self._flush_logs()

    def _flush_logs(self):
        """Enviar mensagens acumuladas para stdout"""
        with self.buffer_lock:
            if self.verbose or len(self.log_buffer) >= self.log_interval:
                for msg in self.log_buffer:
                    print(msg)
            self.log_buffer.clear()

    def force_flush(self):
        """Forçar flush do buffer"""
        self._flush_logs()


class ServerDPBase(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        if not ModuleValidator.is_valid(self.global_model):
            self.global_model = ModuleValidator.fix(self.global_model)
            print("Modelo corrigido pelo Opacus para ser compatível com DP.")
        
        self.privacy_budget_spent = {}

        # Inicializar pool de modelos para otimização de DP
        self.model_pool = ModelDPPool(self.global_model, pool_size=min(5, self.num_clients), device=self.device)

        # Inicializar logger assíncrono
        self.async_logger = AsyncLogger(log_interval=10, verbose=getattr(args, 'verbose_logging', False))

        self.dp_mode = args.dp_mode
        self.dp_epsilon = args.dp_epsilon
        self.dp_delta = args.dp_delta
        self.dp_max_grad_norm = args.dp_max_grad_norm
        
        num_rounds = args.global_rounds if args.global_rounds > 0 else 1
        self.max_epsilon = args.dp_epsilon_max / num_rounds
        self.min_epsilon = args.dp_epsilon_min / num_rounds
        self.current_epsilon_per_round = args.dp_epsilon / num_rounds
        
        self.rs_variance = []
        self.rs_epsilon_per_round = []
        self.rs_cic = []
        self.rs_cep = []
        self.rs_ica = []

        # Histórico de performance para seleção inteligente de clientes
        self.client_performances = defaultdict(list)
        self.client_data_sizes = {}

        self.async_logger.log(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.async_logger.log(f"Modo de Privacidade Diferencial: {self.dp_mode.upper()}")
        self.async_logger.log("Finished creating server and clients for Differential Privacy.")
        self.Budget = []

    def calculate_and_update_metrics(self, current_acc, epsilon_this_round, uploaded_updates=None):
        variance = 0.0

        if uploaded_updates:
            updates_tensor = torch.stack(uploaded_updates)
            variance = torch.mean(torch.var(updates_tensor, dim=0)).item()
        
        self.rs_variance.append(variance)
        self.async_logger.log(f"Métrica de Análise - Variância: {variance:.6f}")

        if self.dp_mode == 'adaptive' and len(self.rs_test_acc) > 1:
            current_acc_gain = self.rs_test_acc[-1] - self.rs_test_acc[-2]
            current_variance = variance

            acc_gain_threshold = 0.001
            variance_increase_threshold = 0.00001
            
            epsilon_increase_factor = 1.1
            epsilon_decrease_factor = 0.9

            # CORREÇÃO DA LÓGICA:
            # Se a acurácia melhora e a variância é estável, podemos ter MENOS ruído (MAIOR epsilon) para acelerar a convergência.
            if current_acc_gain > acc_gain_threshold and current_variance < np.mean(self.rs_variance[-min(5, len(self.rs_variance)):-1]):
                # Aumentar epsilon (menos privacidade, mais utilidade)
                self.current_epsilon_per_round = min(self.current_epsilon_per_round * epsilon_increase_factor, self.max_epsilon)
                self.async_logger.log("ADAPTIVE DP: Acurácia melhorou, variância estável. Aumentando épsilon (menos ruído).")

            # Se a acurácia piora ou a variância aumenta, precisamos de MAIS ruído (MENOR epsilon) para proteger a privacidade e estabilizar.
            elif current_acc_gain < -acc_gain_threshold or current_variance > (np.mean(self.rs_variance[-min(5, len(self.rs_variance)):-1]) + variance_increase_threshold):
                # Diminuir epsilon (mais privacidade, menos utilidade)
                self.current_epsilon_per_round = max(self.current_epsilon_per_round * epsilon_decrease_factor, self.min_epsilon)
                self.async_logger.log("ADAPTIVE DP: Acurácia piorou ou variância instável. Diminuindo épsilon (mais ruído).")
            else:
                self.async_logger.log("ADAPTIVE DP: Mudança neutra. Mantendo épsilon atual.")
        
        self.rs_epsilon_per_round.append(epsilon_this_round)
        print(f"Métrica de Análise - Epsilon da Rodada: {epsilon_this_round:.4f}")

        cic = 0.0
        if len(self.rs_test_acc) >= 5:
            cic = np.std(self.rs_test_acc[-5:])
        self.rs_cic.append(cic)
        print(f"Métrica de Análise - CIC: {cic:.4f}")

        cep = 0.0
        if len(self.rs_test_acc) >= 2 and epsilon_this_round > 0:
            acc_gain = self.rs_test_acc[-1] - self.rs_test_acc[-2]
            cep = acc_gain / epsilon_this_round
        self.rs_cep.append(cep)
        print(f"Métrica de Análise - CEP: {cep:.4f}")

        max_var = max(self.rs_variance) if self.rs_variance else 1.0
        norm_variance = variance / max_var if max_var > 0 else 0
        ica = current_acc * (1 - norm_variance)
        self.rs_ica.append(ica)
        self.async_logger.log(f"Métrica de Análise - ICA: {ica:.4f}")

    def select_clients_smart(self):
        """Seleção inteligente de clientes baseada em performance e diversidade"""
        available_clients = [c for c in self.clients if c.time_cost <= self.time_threthold]

        if len(available_clients) <= self.num_join_clients:
            return available_clients

        # Atualizar informações dos clientes
        for client in available_clients:
            self.client_data_sizes[client.id] = client.train_samples

        # Se tiver histórico de performance, usar seleção inteligente
        if len(self.client_performances) >= 2:
            # Calcular performance média por cliente
            client_avg_performance = {}
            for client_id, performances in self.client_performances.items():
                if performances:
                    client_avg_performance[client_id] = np.mean(performances[-3:])  # Média das últimas 3 rodadas

            # Selecionar top performers (50% dos slots)
            num_top = max(1, self.num_join_clients // 2)
            sorted_by_performance = sorted(client_avg_performance.items(), key=lambda x: x[1], reverse=True)
            top_clients_ids = [cid for cid, _ in sorted_by_performance[:num_top]]

            # Selecionar clientes diversos (baseado no tamanho do dataset)
            remaining_slots = self.num_join_clients - num_top
            diverse_clients = self._select_diverse_clients(remaining_slots, exclude_ids=top_clients_ids)

            selected_ids = top_clients_ids + diverse_clients
            selected_clients = [c for c in available_clients if c.id in selected_ids]

            self.async_logger.log(f"Seleção inteligente: {len(top_clients_ids)} top performers + {len(diverse_clients)} diversos")
        else:
            # Fallback para seleção aleatória no início
            selected_clients = np.random.choice(available_clients, self.num_join_clients, replace=False).tolist()

        return selected_clients

    def _select_diverse_clients(self, num_clients, exclude_ids=None):
        """Selecionar clientes com base na diversidade de tamanho de dataset"""
        if exclude_ids is None:
            exclude_ids = []

        available_diverse = [c for c in self.clients
                           if c.id not in exclude_ids and c.time_cost <= self.time_threthold]

        if not available_diverse:
            return []

        # Agrupar por tamanho de dataset e selecionar de grupos diferentes
        data_sizes = [(c.id, self.client_data_sizes.get(c.id, 0)) for c in available_diverse]
        data_sizes.sort(key=lambda x: x[1])

        # Selecionar espalhados pela lista (tamanhos diferentes)
        selected_ids = []
        step = max(1, len(data_sizes) // num_clients)
        for i in range(0, len(data_sizes), step):
            if len(selected_ids) < num_clients:
                selected_ids.append(data_sizes[i][0])

        return [c for c in available_diverse if c.id in selected_ids]

    def train_dp_optimized(self, client, target_epsilon_for_round):
        """Treinamento DP otimizado usando pool de modelos e batch size dinâmico"""
        client.model.train()

        # Obter modelo do pool (evita deepcopy)
        temp_model = self.model_pool.get_model()

        # Copiar parâmetros do cliente para o modelo do pool
        with torch.no_grad():
            for client_param, pool_param in zip(client.model.parameters(), temp_model.parameters()):
                pool_param.data.copy_(client_param.data)

        # Batch size dinâmico baseado no tamanho do dataset do cliente
        dynamic_batch_size = self._calculate_optimal_batch_size(client)
        optimizer = torch.optim.SGD(temp_model.parameters(), lr=self.learning_rate)
        dataloader = client.load_train_data(batch_size=dynamic_batch_size)

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

        # Treinar com o modelo privado
        client.train(model=private_model, dataloader=private_dataloader, optimizer=private_optimizer)

        epsilon_spent = privacy_engine.get_epsilon(self.dp_delta)
        self.privacy_budget_spent[client.id] += epsilon_spent

        # Retornar modelo ao pool para reutilização
        self.model_pool.return_model(temp_model)

        return epsilon_spent

    def _calculate_optimal_batch_size(self, client):
        """Calcular batch size ótimo baseado no tamanho do dataset do cliente"""
        base_batch_size = self.batch_size
        client_data_ratio = client.train_samples / getattr(self, 'avg_client_data', 1000)

        # Ajustar batch size baseado na quantidade de dados
        # Clientes com mais dados podem usar batches maiores para eficiência
        optimal_size = min(int(base_batch_size * (1 + np.log(max(client_data_ratio, 0.1)))), 64)
        return max(base_batch_size // 2, optimal_size)  # Limitar entre 50% e 64x o original

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()

            # Usar seleção inteligente de clientes quando disponível
            if hasattr(self, 'select_clients_smart') and i > 0:  # Usar após primeira rodada para ter histórico
                self.selected_clients = self.select_clients_smart()
            else:
                self.selected_clients = self.select_clients()

            self.send_models()

            if i % self.eval_gap == 0:
                self.async_logger.log(f"\n-------------Round number: {i}-------------", force=True)
                if self.dp_mode == 'adaptive':
                    self.async_logger.log(f"Épsilon Alvo para esta Rodada: {self.current_epsilon_per_round:.4f}")
                self.evaluate()
                self.async_logger.force_flush()

            total_epsilon_this_round = 0
            uploaded_updates = []

            if self.dp_mode != 'none':
                num_rounds = self.global_rounds if self.global_rounds > 0 else 1
                if self.dp_mode == 'adaptive':
                    target_epsilon_for_round = self.current_epsilon_per_round
                else:
                    target_epsilon_for_round = self.dp_epsilon / num_rounds

                # Usar treinamento DP otimizado
                for client in self.selected_clients:
                    epsilon_spent = self.train_dp_optimized(client, target_epsilon_for_round)
                    total_epsilon_this_round += epsilon_spent
                    self.async_logger.log(f"Cliente {client.id}: Gasto nesta rodada: ε = {epsilon_spent:.4f} / Orçamento Total Gasto: ε = {self.privacy_budget_spent[client.id]:.4f}")

                    # Coletar atualizações para análise de variância
                    if hasattr(client, 'delta_model') and client.delta_model is not None:
                        update = []
                        for param in client.delta_model.parameters():
                            update.append(param.data.view(-1))
                        uploaded_updates.append(torch.cat(update))

                # Estatísticas do pool de modelos
                pool_stats = self.model_pool.get_stats()
                self.async_logger.log(f"DP Pool Stats: {pool_stats['hit_rate']:.2%} hit rate, {pool_stats['available']} disponíveis")

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
