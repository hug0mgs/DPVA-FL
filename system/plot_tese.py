import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import seaborn as sns

# =====================================================================================
# SEÇÃO 1: FUNÇÕES DE CARREGAMENTO DE DADOS
# =====================================================================================

def load_all_results(dataset, algorithms, goal, times, alpha):
    """
    Carrega todos os dados de resultados (none, fixed, adaptive) dos arquivos H5.
    """
    result_path = "."
    algo_map = {
        "FedAvg": "FedAvg",
        "SCAFFOLD": "SCAFFOLD",
        "FedALA": "FedALA"
    }
    
    results = {algo: {mode: {} for mode in ["none", "fixed", "adaptive"]} for algo in algorithms}

    for algo_display_name in algorithms:
        algo_execution_name = algo_map.get(algo_display_name, algo_display_name)
        for mode in ["none", "fixed", "adaptive"]:
            # Garante que o alpha seja formatado como X_0_0
            alpha_str = str(alpha).replace('.', '_')
            file_name = f"{dataset}_{algo_execution_name}_{goal}_{mode}_alpha{alpha_str}_0_0.h5"
            file_path = os.path.join(result_path, file_name)
            
            if os.path.exists(file_path):
                print(f"Carregando dados gerais: {file_path}")
                with h5py.File(file_path, 'r') as hf:
                    results[algo_display_name][mode]['acc'] = np.array(hf.get('rs_test_acc', []))
                    results[algo_display_name][mode]['loss'] = np.array(hf.get('rs_train_loss', []))
                    results[algo_display_name][mode]['auc'] = np.array(hf.get('rs_test_auc', []))
                    if 'rs_epsilon_per_round' in hf:
                        results[algo_display_name][mode]['epsilon_per_round'] = np.array(hf.get('rs_epsilon_per_round', []))
            else:
                print(f"Aviso: Arquivo de resultado não encontrado para gráficos gerais: {file_path}")
    return results

def load_adaptive_analysis_results(dataset, algorithms, goal, times, alpha):
    """
    Carrega dados de análise científica APENAS do modo adaptativo.
    """
    result_path = "."
    algo_map = {
        "FedAvg": "FedAvg",
        "SCAFFOLD": "SCAFFOLD",
        "FedALA": "FedALA"  
    }
    results = {algo: {} for algo in algorithms}

    for algo_display_name in algorithms:
        algo_execution_name = algo_map.get(algo_display_name, algo_display_name)
        # Garante que o alpha seja formatado como X_0_0
        alpha_str = str(alpha).replace('.', '_')
        file_name = f"{dataset}_{algo_execution_name}_{goal}_adaptive_alpha{alpha_str}_0_0.h5"
        file_path = os.path.join(result_path, file_name)
        
        if os.path.exists(file_path):
            print(f"Carregando dados de análise adaptativa: {file_path}")
            with h5py.File(file_path, 'r') as hf:
                results[algo_display_name]['acc'] = np.array(hf.get('rs_test_acc', []))
                results[algo_display_name]['variance'] = np.array(hf.get('rs_variance', []))
                results[algo_display_name]['cic'] = np.array(hf.get('rs_cic', []))
                results[algo_display_name]['cep'] = np.array(hf.get('rs_cep', []))
                results[algo_display_name]['ica'] = np.array(hf.get('rs_ica', []))
                results[algo_display_name]['epsilon'] = np.array(hf.get('rs_epsilon_per_round', []))
        else:
            print(f"Aviso: Arquivo de resultado não encontrado para análise adaptativa: {file_path}")
    return results


# =====================================================================================
# SEÇÃO 2: FUNÇÕES DE PLOTAGEM GERAIS
# =====================================================================================

def save_plot(fig, filename, output_dir='figs'):
    """Salva a figura em um subdiretório especificado."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, format='svg', bbox_inches='tight')
    print(f"Gráfico salvo em: {filepath}")
    plt.close(fig)

def plot_algorithm_comparison(results, dataset, algorithms, alpha):
    """GRÁFICO 1: Compara os modos de DP para cada algoritmo."""
    dp_modes = ["none", "fixed", "adaptive"]
    styles = {
        'none':     {'label': 'Sem DP (Baseline)', 'linestyle': '-', 'color': 'blue', 'marker': 'o', 'markersize': 4},
        'fixed':    {'label': 'DP Fixo', 'linestyle': '--', 'color': 'red', 'marker': 's', 'markersize': 4},
        'adaptive': {'label': 'DP Adaptativo (Proposto)', 'linestyle': ':', 'color': 'green', 'marker': 'x', 'markersize': 4}
    }
    
    fig_acc, axes_acc = plt.subplots(1, len(algorithms), figsize=(20, 6), sharey=True)
    if len(algorithms) == 1: axes_acc = [axes_acc]
    fig_acc.suptitle(f'Comparativo de Acurácia por Algoritmo no Dataset {dataset} (α={alpha})', fontsize=18, weight='bold')

    for i, algo in enumerate(algorithms):
        ax = axes_acc[i]
        has_data = False
        for mode in dp_modes:
            if 'acc' in results[algo][mode] and len(results[algo][mode]['acc']) > 0:
                ax.plot(np.arange(len(results[algo][mode]['acc'])), results[algo][mode]['acc'], **styles[mode])
                has_data = True
        ax.set_title(f'Algoritmo: {algo}', fontsize=14)
        ax.set_xlabel('Rodadas de Comunicação', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        if has_data:
            ax.legend()
    axes_acc[0].set_ylabel('Acurácia de Teste', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig_acc, f'01_acuracia_por_algoritmo_{dataset}_alpha{str(alpha).replace(".", "_")}.svg')

def plot_dp_mode_comparison(results, dataset, algorithms, alpha):
    """GRÁFICO 2: Compara os algoritmos para cada modo de DP."""
    dp_modes = ["none", "fixed", "adaptive"]
    styles = {
        'FedAvg':   {'label': 'FedAvg', 'linestyle': '-', 'color': 'purple'},
        'SCAFFOLD': {'label': 'SCAFFOLD', 'linestyle': '--', 'color': 'orange'},
        'FedALA':   {'label': 'FedALA', 'linestyle': ':', 'color': 'cyan'}
    }

    fig, axes = plt.subplots(1, len(dp_modes), figsize=(20, 6), sharey=True)
    if len(dp_modes) == 1: axes = [axes]
    fig.suptitle(f'Comparativo de Desempenho de Algoritmos por Modo de DP no Dataset {dataset} (α={alpha})', fontsize=18, weight='bold')

    for i, mode in enumerate(dp_modes):
        ax = axes[i]
        has_data = False
        for algo in algorithms:
            if 'acc' in results[algo][mode] and len(results[algo][mode]['acc']) > 0:
                ax.plot(np.arange(len(results[algo][mode]['acc'])), results[algo][mode]['acc'], **styles.get(algo, {}))
                has_data = True
        mode_title = {'none': 'Sem DP', 'fixed': 'DP Fixo', 'adaptive': 'DP Adaptativo'}[mode]
        ax.set_title(f'Cenário: {mode_title}', fontsize=14)
        ax.set_xlabel('Rodadas de Comunicação', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        if has_data:
            ax.legend()
    axes[0].set_ylabel('Acurácia de Teste', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, f'02_acuracia_por_modo_dp_{dataset}_alpha{str(alpha).replace(".", "_")}.svg')

def plot_summary_accuracy(results, dataset, algorithms, alpha):
    """GRÁFICO 3: Gráfico de barras resumindo a acurácia máxima."""
    max_accuracies = {}
    for algo in algorithms:
        max_accuracies[algo] = []
        for mode in ['none', 'fixed', 'adaptive']:
            if 'acc' in results[algo][mode] and len(results[algo][mode]['acc']) > 0:
                max_accuracies[algo].append(np.max(results[algo][mode]['acc']) * 100)
            else:
                max_accuracies[algo].append(0)

    labels = ['Sem DP', 'DP Fixo', 'DP Adaptativo']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))
    algo_map = {'FedAvg': 'purple', 'SCAFFOLD': 'orange', 'FedALA': 'cyan'}
    
    rects = []
    for i, algo in enumerate(algorithms):
        offset = width * (i - (len(algorithms) - 1) / 2)
        bar = ax.bar(x + offset, max_accuracies.get(algo, [0,0,0]), width, label=algo, color=algo_map.get(algo))
        rects.append(bar)

    ax.set_ylabel('Acurácia Máxima de Teste (%)', fontsize=12)
    ax.set_title(f'Resumo da Acurácia Máxima por Configuração no Dataset {dataset} (α={alpha})', fontsize=16, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 100)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for r in rects:
        autolabel(r)

    fig.tight_layout()
    save_plot(fig, f'03_resumo_acuracia_maxima_{dataset}_alpha{str(alpha).replace(".", "_")}.svg')

# =====================================================================================
# SEÇÃO 3: FUNÇÕES DE PLOTAGEM (ANÁLISE CIENTÍFICA E NOVOS GRÁFICOS)
# =====================================================================================

def plot_adaptive_analysis(results, metric_name, title, ylabel, filename, algorithms):
    """Função genérica para plotar qualquer um dos coeficientes de análise."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    styles = {
        'FedAvg':   {'label': 'FedAvg', 'color': 'purple', 'marker': 'o'},
        'SCAFFOLD': {'label': 'SCAFFOLD', 'color': 'orange', 'marker': 's'},
        'FedALA':   {'label': 'FedALA', 'color': 'cyan', 'marker': '^'}
    }

    has_data = False
    for algo in algorithms:
        if algo in results and metric_name in results[algo] and len(results[algo][metric_name]) > 0:
            style = styles.get(algo, {})
            ax.plot(results[algo][metric_name], linestyle='--', markersize=4, alpha=0.8, **style)
            has_data = True

    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel('Rodadas de Comunicação', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if has_data:
        ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    save_plot(fig, filename, output_dir='figs_analysis')

def plot_privacy_utility_tradeoff(results, dataset, algorithms, alpha):
    """GRÁFICO 10: Acurácia Máxima vs. Orçamento de Privacidade Total (Épsilon)."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    styles = {
        ('FedAvg', 'fixed'):    {'label': 'FedAvg (Fixo)', 'color': 'purple', 'marker': 's'},
        ('FedAvg', 'adaptive'): {'label': 'FedAvg (Adaptativo)', 'color': 'purple', 'marker': 'o', 'linestyle': '--'},
        ('SCAFFOLD', 'fixed'):  {'label': 'SCAFFOLD (Fixo)', 'color': 'orange', 'marker': 's'},
        ('SCAFFOLD', 'adaptive'):{'label': 'SCAFFOLD (Adaptativo)', 'color': 'orange', 'marker': 'o', 'linestyle': '--'},
        ('FedALA', 'fixed'):    {'label': 'FedALA (Fixo)', 'color': 'cyan', 'marker': 's'},
        ('FedALA', 'adaptive'): {'label': 'FedALA (Adaptativo)', 'color': 'cyan', 'marker': 'o', 'linestyle': '--'},
    }
    
    has_data = False
    for algo in algorithms:
        for mode in ['fixed', 'adaptive']:
            if 'acc' in results[algo][mode] and len(results[algo][mode]['acc']) > 0 and 'epsilon_per_round' in results[algo][mode]:
                max_acc = np.max(results[algo][mode]['acc']) * 100
                total_epsilon = np.sum(results[algo][mode]['epsilon_per_round'])
                
                style = styles.get((algo, mode), {})
                ax.scatter(total_epsilon, max_acc, s=150, **style)
                ax.text(total_epsilon, max_acc + 0.5, f'{max_acc:.1f}%', ha='center')
                has_data = True

    ax.set_title(f'Trade-off Privacidade-Utilidade no Dataset {dataset}', fontsize=16, weight='bold')
    ax.set_xlabel('Orçamento de Privacidade Total Gasto (ε)', fontsize=12)
    ax.set_ylabel('Acurácia Máxima de Teste (%)', fontsize=12)
    ax.set_ylim(bottom=0, top=100)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if has_data:
        ax.legend(fontsize=12, title="Configuração")

    plt.tight_layout()
    save_plot(fig, f'10_privacidade_utilidade_tradeoff_{dataset}_alpha{str(round(alpha, 1)).replace(".", "_")}.svg', output_dir='figs_analysis')

def plot_variance_vs_epsilon(adaptive_results, algo, dataset, alpha):
    """NOVO GRÁFICO 11: Compara a Variância e o Gasto de Épsilon ao longo do tempo."""
    if not (algo in adaptive_results and 'variance' in adaptive_results[algo] and 'epsilon' in adaptive_results[algo]):
        print(f"Dados de variância ou épsilon não encontrados para {algo}. Pulando gráfico.")
        return

    variance_data = adaptive_results[algo]['variance']
    epsilon_data = adaptive_results[algo]['epsilon']
    
    if len(variance_data) == 0 or len(epsilon_data) == 0:
        print(f"Dados de variância ou épsilon vazios para {algo}. Pulando gráfico.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Eixo Y esquerdo para a Variância
    color = 'tab:blue'
    ax1.set_xlabel('Rodadas de Comunicação', fontsize=12)
    ax1.set_ylabel('Variância das Atualizações', color=color, fontsize=12)
    ax1.plot(variance_data, color=color, linestyle='-', marker='o', markersize=4, label='Variância')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Eixo Y direito para o Épsilon
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Gasto de Privacidade (ε) por Rodada', color=color, fontsize=12)
    ax2.plot(epsilon_data, color=color, linestyle='--', marker='x', markersize=4, label='Épsilon (ε)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(f'Política Adaptativa em Ação ({algo} no {dataset}, α={alpha})', fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Adiciona uma única legenda para ambos os eixos
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    save_plot(fig, f'11_variance_vs_epsilon_{algo}_{dataset}_alpha{str(round(alpha, 1)).replace(".", "_")}.svg', output_dir='figs_analysis')


# =====================================================================================
# SEÇÃO 4: FUNÇÃO PRINCIPAL (MAIN - ATUALIZADA)
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="Gera gráficos comparativos e de análise para a tese.")
    parser.add_argument('-data', "--dataset", type=str, required=True, help="Dataset usado (ex: Cifar10)")
    parser.add_argument('-go', "--goal", type=str, default="test", help="O objetivo do experimento (geralmente 'test')")
    parser.add_argument('-t', "--times", type=int, default=0, help="O índice da execução (geralmente 0 se times=1)")
    parser.add_argument('-dal', '--dirichlet_alpha', type=float, default=1.0, help='Valor de alpha para o particionamento Dirichlet.')
    
    args = parser.parse_args()
    
    sns.set_theme(style="whitegrid")
    
    algorithms_to_plot = ["FedAvg", "SCAFFOLD", "FedALA"]
    alphas_to_plot = [0.0, 1.0, 5.0] # Valores de alpha que você está usando
    
    for alpha in alphas_to_plot:
        print(f"\n==========================================================")
        print(f"INICIANDO PLOTAGEM PARA ALPHA = {alpha}")
        print(f"==========================================================")
        
        # --- Parte 1: Gráficos Comparativos Gerais ---
        print("--- Gerando Gráficos Comparativos Gerais ---")
        general_results = load_all_results(args.dataset, algorithms_to_plot, args.goal, args.times, alpha)
        
        if any(any(mode_results for mode_results in algo_results.values()) for algo_results in general_results.values()):
            plot_algorithm_comparison(general_results, args.dataset, algorithms_to_plot, alpha)
            plot_dp_mode_comparison(general_results, args.dataset, algorithms_to_plot, alpha)
            plot_summary_accuracy(general_results, args.dataset, algorithms_to_plot, alpha)
            print("\n--- Gráficos Comparativos Gerais Concluídos ---\n")
        else:
            print(f"\n--- Nenhum dado encontrado para os gráficos gerais (alpha={alpha}). Pulando. ---\n")
        
        # --- Parte 2: Análise Científica do Modo Adaptativo ---
        print("--- Gerando Gráficos de Análise Científica (Modo Adaptativo) ---")
        adaptive_results = load_adaptive_analysis_results(args.dataset, algorithms_to_plot, args.goal, args.times, alpha)
        
        if any(adaptive_results.values()):
            alpha_str_plot = str(int(alpha)) + '_0' if alpha == int(alpha) else str(alpha).replace('.', '_')
            
            plot_adaptive_analysis(adaptive_results, 'acc', f'Acurácia de Teste (Modo Adaptativo, α={alpha})', 'Acurácia', f'04_adaptive_accuracy_alpha{alpha_str_plot}.svg', algorithms_to_plot)
            plot_adaptive_analysis(adaptive_results, 'epsilon', f'Épsilon por Rodada (Modo Adaptativo, α={alpha})', 'Valor de Épsilon (ε)', f'05_adaptive_epsilon_alpha{alpha_str_plot}.svg', algorithms_to_plot)
            plot_adaptive_analysis(adaptive_results, 'variance', f'Variância das Atualizações (Modo Adaptativo, α={alpha})', 'Variância Média', f'06_adaptive_variance_alpha{alpha_str_plot}.svg', algorithms_to_plot)
            plot_adaptive_analysis(adaptive_results, 'cic', f'Coeficiente de Instabilidade de Convergência (CIC, α={alpha})', 'CIC', f'07_adaptive_cic_alpha{alpha_str_plot}.svg', algorithms_to_plot)
            plot_adaptive_analysis(adaptive_results, 'cep', f'Coeficiente de Eficiência de Privacidade (CEP, α={alpha})', 'CEP (ΔAcurácia / ε)', f'08_adaptive_cep_alpha{alpha_str_plot}.svg', algorithms_to_plot)
            plot_adaptive_analysis(adaptive_results, 'ica', f'Índice de Convergência Adaptativa (ICA, α={alpha})', 'ICA', f'09_adaptive_ica_alpha{alpha_str_plot}.svg', algorithms_to_plot)
            
            plot_privacy_utility_tradeoff(general_results, args.dataset, algorithms_to_plot, alpha) # Usa general_results para ter todos os modos
            
            for algo in algorithms_to_plot:
                plot_variance_vs_epsilon(adaptive_results, algo, args.dataset, alpha)
            
            print("\n--- Gráficos de Análise Científica Concluídos ---\n")
        else:
            print(f"\n--- Nenhum dado encontrado para os gráficos de análise adaptativa (alpha={alpha}). Pulando. ---\n")
    
    print("Todos os gráficos foram gerados e salvos nas pastas 'figs/' e 'figs_analysis/'.")

if __name__ == "__main__":
    main()
