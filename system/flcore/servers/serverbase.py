import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = [] # Inicializa como lista vazia
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client
        self.alpha = args.dirichlet_alpha # Adicionado para salvar o valor de alpha (renomeado)

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True
        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"] + \
                        client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        # Corrigido para garantir que os resultados sejam salvos corretamente
        # Usa o nome do algoritmo, o modo DP e o valor de alpha para o nome do arquivo
        # Garante que o alpha seja formatado como X_0
        alpha_str = str(self.alpha).replace('.', '_')
        # O nome do arquivo deve ser: Dataset_Algoritmo_test_DPMode_alphaX_0_0.h5
        # O self.goal é 'test' por padrão, e self.times é 1 (o que significa que o índice é 0)
        # O nome do arquivo final deve ser X_0_0.h5
        algo = f"{self.dataset}_{self.algorithm}_{self.goal}_{self.args.dp_mode}_alpha{alpha_str}_0_0"
        
        # Cria o diretório de resultados se não existir
        # CORREÇÃO: Salva no diretório atual, não em ../results/
        result_path = "."
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # CORREÇÃO: Garante que sempre salve os resultados, mesmo que estejam vazios
        file_path = os.path.join(result_path, f"{algo}.h5")
        print(f"Salvando resultados em: {file_path}")

        # Garante que as listas não estejam vazias
        if not self.rs_test_acc:
            print("AVISO: rs_test_acc está vazio. Adicionando valor padrão para evitar arquivo vazio.")
            self.rs_test_acc = [0.0]  # Valor padrão para evitar arquivo vazio
        
        if not self.rs_train_loss:
            print("AVISO: rs_train_loss está vazio. Adicionando valor padrão para evitar arquivo vazio.")
            self.rs_train_loss = [0.0]  # Valor padrão para evitar arquivo vazio
            
        # AVISO: rs_test_auc agora é populado corretamente na função evaluate
        # Não precisamos mais adicionar um valor padrão aqui, a menos que evaluate não seja chamado
        # ou que o AUC seja 0 por algum motivo legítimo.
        # if not self.rs_test_auc:
        #     print("AVISO: rs_test_auc está vazio. Adicionando valor padrão para evitar arquivo vazio.")
        #     self.rs_test_auc = [0.0]  # Valor padrão para evitar arquivo vazio

        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            
        print(f"Resultados salvos com sucesso. Tamanho dos dados: acc={len(self.rs_test_acc)}, auc={len(self.rs_test_auc)}, loss={len(self.rs_train_loss)}")

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])  
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        # CORREÇÃO: Adiciona o valor de test_auc à lista rs_test_auc
        self.rs_test_auc.append(test_auc)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        
        return test_acc, train_loss, test_auc

    def evaluate_personalized_model(self):
        stats = self.test_metrics_personalized()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])  
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        print("Averaged Personalized Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Personalized Test AUC: {:.4f}".format(test_auc))
        print("Std Personalized Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Personalized Test AUC: {:.4f}".format(np.std(aucs)))
        
        return test_acc, test_auc

    def evaluate_one_step(self):
        stats = self.test_metrics_one_step()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])  
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        print("Averaged One-Step Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged One-Step Test AUC: {:.4f}".format(test_auc))
        print("Std One-Step Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std One-Step Test AUC: {:.4f}".format(np.std(aucs)))
        
        return test_acc, test_auc

    def evaluate_new_clients(self, acc=None, loss=None):
        if self.num_new_clients == 0:
            return 0, 0, 0
        if self.eval_new_clients:
            self.fine_tuning_new_clients()
            stats = self.test_metrics_new_clients()
            stats_train = self.train_metrics_new_clients()
            test_acc = sum(stats[2])*1.0 / sum(stats[1])
            test_auc = sum(stats[3])*1.0 / sum(stats[1])  
            train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
            accs = [a / n for a, n in zip(stats[2], stats[1])]
            aucs = [a / n for a, n in zip(stats[3], stats[1])]
            
            if acc == None:
                self.rs_test_acc.append(test_acc)
            else:
                acc.append(test_acc)
            
            if loss == None:
                self.rs_train_loss.append(train_loss)
            else:
                loss.append(train_loss)

            self.rs_test_auc.append(test_auc)

            print("Averaged New Clients Train Loss: {:.4f}".format(train_loss))
            print("Averaged New Clients Test Accuracy: {:.4f}".format(test_acc))
            print("Averaged New Clients Test AUC: {:.4f}".format(test_auc))
            print("Std New Clients Test Accuracy: {:.4f}".format(np.std(accs)))
            print("Std New Clients Test AUC: {:.4f}".format(np.std(aucs)))
            
            return test_acc, train_loss, test_auc
        else:
            return 0, 0, 0

    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
        ids = [c.id for c in self.new_clients]
        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_new_clients(self):
        num_samples = []
        losses = []
        for c in self.new_clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in self.new_clients]
        return ids, num_samples, losses

    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)
            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))
            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
        ids = [c.id for c in self.new_clients]
        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_new_clients(self):
        num_samples = []
        losses = []
        for c in self.new_clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in self.new_clients]
        return ids, num_samples, losses

    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            client.fine_tuning(self.fine_tuning_epoch_new)

    def cprint(self, color, text):
        if color == 'red':
            print('\033[1;31m' + text + '\033[0m')
        elif color == 'green':
            print('\033[1;32m' + text + '\033[0m')
        elif color == 'yellow':
            print('\033[1;33m' + text + '\033[0m')
        elif color == 'blue':
            print('\033[1;34m' + text + '\033[0m')
        elif color == 'purple':
            print('\033[1;35m' + text + '\033[0m')
        elif color == 'cyan':
            print('\033[1;36m' + text + '\033[0m')
        else:
            print(text)

    def train(self):
        for i in range(self.global_rounds):
            s_time = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.evaluate()
            
            e_time = time.time()
            print(f"------------------------- time cost ------------------------- {e_time - s_time}")

        print("Best accuracy.")
        print(max(self.rs_test_acc))
        print("Average time cost per round.")
        print((e_time - s_time) / self.global_rounds)
        self.save_results()
        self.save_global_model()

    def call_dlg(self, round):
        loss = []
        grads = []
        
        for client in self.selected_clients:
            client_loss, client_grad = client.get_grad()
            loss.append(client_loss)
            grads.append(client_grad)
        
        if len(grads) > 0:
            grads = average_data(grads)
            loss = average_data(loss)
            
            dlg = DLG(self.global_model, loss, grads)
            dlg.run(self.dlg_gap)

    def call_dlg_client(self, round):
        loss = []
        grads = []
        
        for client in self.selected_clients:
            client_loss, client_grad = client.get_grad()
            loss.append(client_loss)
            grads.append(client_grad)
        
        if len(grads) > 0:
            grads = average_data(grads)
            loss = average_data(loss)
            
            dlg = DLG(self.global_model, loss, grads)
            dlg.run(self.dlg_gap)
