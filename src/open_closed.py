import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
from collections import Counter
import math
import numpy as np
import pandas as pd
import os

# Set seaborn style for better aesthetics
sns.set(style="whitegrid", context="talk")

def generate_union_closed_family(n, num_sets, bias=None, seed=None):
    """
    Generates a union-closed family of sets with improved balance.
    
    Parameters:
    - n (int): Number of elements in the base set.
    - num_sets (int): Number of sets to generate.
    - bias (dict, optional): Probability bias for adding elements not already in the union.
    - seed (int, optional): Random seed for reproducibility.
    
    Returns:
    - family (list of sets): Generated union-closed family.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    base_universe = set(range(1, n+1))
    # Initialize with the empty set and all singleton sets
    family = [set()] + [{i} for i in range(1, n+1)]
    
    # Initialize element counts
    element_counts = Counter()
    for s in family:
        element_counts.update(s)
    
    for _ in range(num_sets - len(family)):
        s1 = random.choice(family)
        s2 = random.choice(family)
        new_set = s1.union(s2)
        
        if bias is not None:
            for elem in base_universe:
                if elem not in new_set:
                    # Adjust bias based on current frequency
                    current_freq = element_counts[elem] / len(family)
                    adjusted_bias = bias.get(elem, 0.0) * (1 - current_freq)
                    if random.random() < adjusted_bias:
                        new_set.add(elem)
                        element_counts[elem] += 1
        
        if new_set not in family:
            family.append(new_set)
            element_counts.update(new_set)
    
    return family

def element_frequencies(family, n):
    """
    Calculates the frequency of each element in the union-closed family.
    
    Parameters:
    - family (list of sets): Union-closed family.
    - n (int): Number of elements in the base set.
    
    Returns:
    - freq_dict (dict): Frequency of each element.
    """
    counts = Counter()
    total = len(family)
    for s in family:
        counts.update(s)
    return {elem: counts[elem]/total for elem in range(1, n+1)}

def compute_entropy(family, n):
    """
    Computes the entropy of the element distribution in the family.
    
    Parameters:
    - family (list of sets): Union-closed family.
    - n (int): Number of elements in the base set.
    
    Returns:
    - entropy (float): Entropy value.
    """
    freqs = element_frequencies(family, n)
    entropy = 0.0
    for p in freqs.values():
        if p > 0 and p < 1:
            entropy -= p * math.log2(p) + (1-p) * math.log2(1-p)
    return entropy

def build_network(family):
    """
    Builds a network graph from the union-closed family.
    
    Parameters:
    - family (list of sets): Union-closed family.
    
    Returns:
    - G (networkx.Graph): Network graph.
    """
    G = nx.Graph()
    for s in family:
        elements = list(s)
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                G.add_edge(elements[i], elements[j])
    return G

def plot_single_simulation(freqs, G, simulation_id, output_dir):
    """
    Plots frequency distribution and network graph for a single simulation.
    
    Parameters:
    - freqs (dict): Frequency of each element.
    - G (networkx.Graph): Network graph.
    - simulation_id (int): Identifier for the simulation.
    - output_dir (str): Directory to save the plot.
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot Frequency Distribution
    elements = list(freqs.keys())
    frequencies = list(freqs.values())
    
    sns.barplot(ax=axes[0], x=elements, y=frequencies, palette="viridis")
    axes[0].axhline(0.5, color='red', linestyle='--', label='50% Threshold')
    axes[0].set_xlabel('Elements')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Element Frequency Distribution (Sim {simulation_id})')
    axes[0].legend()
    
    # Plot Network Graph
    pos = nx.spring_layout(G, seed=42, k=0.15)  # Adjust 'k' for better spacing
    nx.draw_networkx_nodes(G, pos, ax=axes[1], node_size=300, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=axes[1], alpha=0.5)
    
    if G.number_of_nodes() <= 20:
        nx.draw_networkx_labels(G, pos, ax=axes[1], font_size=10)
    
    axes[1].set_title(f'Network Visualization (Sim {simulation_id})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'simulation_{simulation_id}.png'), dpi=300)
    plt.close()

def plot_combined_simulations(simulation_ids, max_freqs, entropies, output_dir):
    """
    Plots combined max frequencies and entropy over simulations.
    
    Parameters:
    - simulation_ids (list of int): Simulation identifiers.
    - max_freqs (list of float): Maximum frequencies.
    - entropies (list of float): Entropy values.
    - output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(20, 10))
    
    # Plot Max Frequencies
    plt.subplot(1, 2, 1)
    sns.lineplot(x=simulation_ids, y=max_freqs, marker='o', color='green', label='Max Frequency')
    plt.axhline(0.5, color='red', linestyle='--', label='50% Threshold')
    plt.xlabel('Simulation ID')
    plt.ylabel('Max Element Frequency')
    plt.title('Maximum Element Frequency Across Simulations')
    plt.legend()
    
    # Plot Entropy
    plt.subplot(1, 2, 2)
    sns.lineplot(x=simulation_ids, y=entropies, marker='o', color='blue', label='Entropy')
    plt.xlabel('Simulation ID')
    plt.ylabel('Entropy')
    plt.title('Entropy Across Simulations')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_simulations.png'), dpi=300)
    plt.close()

def run_simulation(n, num_sets, bias, simulation_id, output_dir):
    """
    Runs a single simulation and generates corresponding plots.
    
    Parameters:
    - n (int): Number of elements in the base set.
    - num_sets (int): Number of sets in the family.
    - bias (dict): Bias for adding elements.
    - simulation_id (int): Identifier for the simulation.
    - output_dir (str): Directory to save outputs.
    
    Returns:
    - result (dict): Key metrics from the simulation.
    """
    family = generate_union_closed_family(n, num_sets, bias=bias, seed=simulation_id)
    freqs = element_frequencies(family, n)
    max_freq_elem = max(freqs, key=freqs.get)
    max_freq = freqs[max_freq_elem]
    entropy = compute_entropy(family, n)
    family_size = len(family)
    
    # Build network graph
    G = build_network(family)
    
    # Plot frequency distribution and network graph
    plot_single_simulation(freqs, G, simulation_id, output_dir)
    
    # Print frequencies for debugging
    print(f"Simulation {simulation_id}: Max frequency = {max_freq:.2f} (Element {max_freq_elem}), Entropy = {entropy:.2f}")
    
    return {
        'Simulation ID': simulation_id,
        'Max Frequency Element': max_freq_elem,
        'Max Frequency': max_freq,
        'Entropy': entropy,
        'Family Size': family_size
    }

def main():
    # Parameters
    n = 60  # Number of elements in the base set
    num_sets = 500  # Increased number of sets for better distribution
    bias = {i: 0.05 for i in range(1, n+1)}  # Bias for adding elements
    num_simulations = 10  # Total number of simulations
    
    # Directories for outputs
    output_dir = 'simulation_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to store simulation data
    results = []
    entropies = []
    max_freqs = []
    simulation_ids = list(range(1, num_simulations + 1))
    
    # Run simulations
    for sim_id in simulation_ids:
        print(f"Running simulation {sim_id}...")
        result = run_simulation(n, num_sets, bias, sim_id, output_dir)
        results.append(result)
        entropies.append(result['Entropy'])
        max_freqs.append(result['Max Frequency'])
    
    # Save all results to CSV
    df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, 'simulation_results.csv')
    df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
    # Generate summary statistics
    summary = df.describe().transpose()
    summary_csv = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(summary_csv)
    print(f"Summary statistics saved to {summary_csv}")
    
    # Plot combined max frequencies and entropy
    plot_combined_simulations(simulation_ids, max_freqs, entropies, output_dir)
    print(f"Combined simulation plots saved to {os.path.join(output_dir, 'combined_simulations.png')}")
    
    # Optionally, create a multi-panel figure with multiple frequency distributions
    # For brevity, this example will include only 9 simulations
    num_plots = 9
    fig, axes = plt.subplots(3, 3, figsize=(30, 30))
    selected_simulations = simulation_ids[:num_plots]
    
    for idx, sim_id in enumerate(selected_simulations):
        row = idx // 3
        col = idx % 3
        sim_result = df[df['Simulation ID'] == sim_id].iloc[0]
        
        # Regenerate family and frequencies
        family = generate_union_closed_family(n, num_sets, bias=bias, seed=sim_id)
        freqs = element_frequencies(family, n)
        
        elements = list(freqs.keys())
        frequencies = list(freqs.values())
        
        sns.barplot(ax=axes[row, col], x=elements, y=frequencies, palette="viridis")
        axes[row, col].axhline(0.5, color='red', linestyle='--', label='50% Threshold')
        axes[row, col].set_xlabel('Elements')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].set_title(f'Element Frequency (Sim {sim_id})')
        axes[row, col].legend()
    
    plt.tight_layout()
    multi_plot_path = os.path.join(output_dir, 'multiple_frequency_distributions.png')
    plt.savefig(multi_plot_path, dpi=300)
    plt.close()
    print(f"Multiple frequency distributions saved to {multi_plot_path}")

if __name__ == "__main__":
    main()
