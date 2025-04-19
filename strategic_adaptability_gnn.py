#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules from the project
from meta_game_analysis.data_processing import (
    process_all_games,
    get_display_name,
    compute_global_max_values
)
from meta_game_analysis.matrix_creation import (
    create_performance_matrices,
    create_welfare_matrices,
    clean_matrix_names,
    filter_matrices
)
from meta_game_analysis.nash_analysis import (
    run_nash_analysis,
    calculate_acceptance_ratio
)

# Constants
REASONING_STYLES = {
    "reasoning": [
        "o3_mini",
        "sonnet_3.7_reasoning"
    ],
    "non_reasoning": [
    ]
}

class SimpleGNN(nn.Module):
    """Simple GNN for strategic analysis"""
    def __init__(self, num_node_features, hidden_channels=64):
        super().__init__()
        
        print(f"Creating SimpleGNN with {num_node_features} input features and {hidden_channels} hidden channels")
        
        # Edge feature compression
        self.edge_nn = nn.Sequential(
            nn.Linear(8, 1),  
            nn.Sigmoid() 
        )
        
        # GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Output layers
        self.global_pool = global_mean_pool
        self.output = nn.Linear(hidden_channels, 1)  # Binary classification
    
    def forward(self, x, edge_index, edge_attr):
        # Process edge attributes to get scalar edge weights
        edge_weight = self.edge_nn(edge_attr).view(-1)
        
        # First convolution
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second convolution
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        # Global pooling (if needed)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = self.global_pool(x, batch)
        
        # Output
        score = self.output(x)  # Node-level classification
        
        return score

def build_strategic_graph(performance_matrix, nash_welfare_matrix, bootstrap_stats, variance_matrix, reasoning_styles=None):
    """Build a graph representation of the strategic landscape"""
    agents = performance_matrix.index.tolist()
    
    # Try to extract NE regret from bootstrap_stats
    ne_regret_matrix = None
    
    if bootstrap_stats is not None:
        if hasattr(bootstrap_stats, 'columns'):
            for col in bootstrap_stats.columns:
                if 'regret' in col.lower():
                    regret_column = col
                    ne_regret_matrix = bootstrap_stats
                    break
        elif isinstance(bootstrap_stats, dict) and 'regret' in bootstrap_stats:
            regret_values = bootstrap_stats['regret']
            ne_regret_matrix = pd.DataFrame(regret_values, index=agents, columns=['NE Regret'])
    
    # If regret information not found, create dummy values
    if ne_regret_matrix is None:
        print("Warning: Could not find NE regret information. Using random dummy values.")
        dummy_regrets = np.random.uniform(0.001, 0.1, size=len(agents))
        ne_regret_matrix = pd.DataFrame(dummy_regrets, index=agents, columns=['NE Regret'])
    
    # Extract features for each agent (node)
    node_features = []
    for agent in agents:
        # Performance metrics
        perf_i = float(performance_matrix.loc[agent, :].mean())
        max_perf = performance_matrix.values.max()
        norm_perf = perf_i / max_perf if max_perf > 0 else 0.0
        
        # Cooperation metrics
        welfare_i = float(nash_welfare_matrix.loc[agent, :].mean()) if agent in nash_welfare_matrix.index else 0.0
        
        # Exploitability metrics
        regret_i = 0.0
        if agent in ne_regret_matrix.index:
            if 'Mean NE Regret' in ne_regret_matrix.columns:
                regret_i = float(ne_regret_matrix.loc[agent, 'Mean NE Regret'])
            elif 'NE Regret' in ne_regret_matrix.columns:
                regret_i = float(ne_regret_matrix.loc[agent, 'NE Regret'])
        
        # Transform regret to exploitability score
        exploitability_score = np.exp(-10 * regret_i) if regret_i > 0 else 1.0
        
        # Consistency metrics
        variance_i = float(variance_matrix.loc[agent, :].mean()) if agent in variance_matrix.index else 0.0
        consistency_score = 1.0 / (1.0 + variance_i) if variance_i > 0 else 1.0
        
        # Agent features
        features = [
            norm_perf,                # Normalized performance
            welfare_i,                # Nash welfare
            exploitability_score,     # Exploitability score
            float(regret_i),          # Raw regret
            consistency_score,        # Consistency
            float(variance_i)         # Variance
        ]
        
        node_features.append(features)
    
    # Create edge relationships and features
    edge_index = []
    edge_features = []
    
    for i, agent_i in enumerate(agents):
        for j, agent_j in enumerate(agents):
            if i == j:  # Skip self-loops
                continue
            
            edge_index.append([i, j])
            
            try:
                # Performance when agent_i plays against agent_j
                perf_ij = performance_matrix.loc[agent_i, agent_j] if agent_j in performance_matrix.columns else 0.0
                perf_ij = float(perf_ij) if np.isscalar(perf_ij) else 0.0
                
                # Normalize performance
                norm_perf_ij = perf_ij / max_perf if max_perf > 0 else 0.0
                
                # Reciprocal performance
                perf_ji = performance_matrix.loc[agent_j, agent_i] if agent_i in performance_matrix.columns else 0.0
                perf_ji = float(perf_ji) if np.isscalar(perf_ji) else 0.0
                
                # Nash welfare
                welfare_ij = nash_welfare_matrix.loc[agent_i, agent_j] if (agent_i in nash_welfare_matrix.index and agent_j in nash_welfare_matrix.columns) else 0.0
                welfare_ij = float(welfare_ij) if np.isscalar(welfare_ij) else 0.0
                
                # Variance in performance
                variance_ij = 0.0
                if agent_i in variance_matrix.index and agent_j in variance_matrix.columns:
                    variance_ij = variance_matrix.loc[agent_i, agent_j]
                    variance_ij = float(variance_ij) if np.isscalar(variance_ij) else 0.0
                
                # Edge features
                edge_attr = [
                    float(norm_perf_ij),         # Normalized performance i->j
                    float(perf_ij),              # Raw performance i->j
                    float(perf_ji),              # Performance j->i
                    float(welfare_ij),           # Nash welfare i->j
                    float(perf_ij - perf_ji),    # Performance difference
                    float(perf_ij / max(1.0, perf_ji)), # Performance ratio
                    float(variance_ij),          # Variance i->j
                    float(1.0 if perf_ij > perf_ji else 0.0)  # Win indicator
                ]
                
            except (ValueError, TypeError) as e:
                print(f"Warning: Error creating edge features for {agent_i} -> {agent_j}: {e}")
                edge_attr = [0.0] * 8
            
            edge_features.append(edge_attr)
    
    # Convert to PyTorch Geometric Data object
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        agent_names=agents
    )
    
    return data

def classify_agent_reasoning_style(agent_name):
    """Classify agent as reasoning or non-reasoning"""
    if any(pattern in agent_name for pattern in REASONING_STYLES["reasoning"]):
        return "reasoning"
    return "non_reasoning"

def train_model(graph_data, reasoning_styles, num_epochs=200, lr=0.01):
    """Train the simple GNN model using supervised learning"""
    # Create labels - reasoning = 1, non_reasoning = 0
    y = torch.zeros(len(graph_data.agent_names), 1)
    for i, agent in enumerate(graph_data.agent_names):
        if agent in reasoning_styles:
            style = reasoning_styles[agent]
        else:
            style = classify_agent_reasoning_style(agent)
            reasoning_styles[agent] = style
            
        if style == "reasoning":
            y[i] = 1.0
    
    # Split data into train/val
    num_nodes = graph_data.x.shape[0]
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_idx = indices[:int(0.8 * num_nodes)]
    val_idx = indices[int(0.8 * num_nodes):]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    # Initialize model and optimizer
    model = SimpleGNN(graph_data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    model.train()
    progress_bar = tqdm(range(num_epochs), desc="Training epochs")
    
    for epoch in progress_bar:
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation every 10 epochs
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
                val_loss = criterion(val_out[val_mask], y[val_mask])
                
                # Calculate accuracies
                train_preds = (torch.sigmoid(out[train_mask]) > 0.5).float()
                train_acc = (train_preds == y[train_mask]).float().mean()
                
                val_preds = (torch.sigmoid(val_out[val_mask]) > 0.5).float()
                val_acc = (val_preds == y[val_mask]).float().mean()
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'train_acc': f"{train_acc.item():.4f}",
                    'val_loss': f"{val_loss.item():.4f}",
                    'val_acc': f"{val_acc.item():.4f}"
                })
            model.train()
    
    return model

def evaluate_strategic_adaptability(model, graph_data, reasoning_styles, performance_data):
    """Evaluate strategic adaptability and test hypothesis"""
    model.eval()
    
    with torch.no_grad():
        scores = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        probabilities = torch.sigmoid(scores).squeeze()
    
    # Group results by reasoning style
    results = {
        "reasoning": {
            "agents": [],
            "scores": [],
            "probabilities": [],
            "performance": []
        },
        "non_reasoning": {
            "agents": [],
            "scores": [],
            "probabilities": [],
            "performance": []
        }
    }
    
    for i, agent in enumerate(tqdm(graph_data.agent_names, desc="Evaluating agents")):
        style = reasoning_styles[agent]
        results[style]["agents"].append(agent)
        results[style]["scores"].append(scores[i].item())
        results[style]["probabilities"].append(probabilities[i].item())
        results[style]["performance"].append(performance_data.get(agent, 0))
    
    # Calculate average performance by reasoning style
    avg_performance = {
        style: np.mean(data["performance"]) 
        for style, data in results.items() 
        if data["performance"]
    }
    
    # Calculate performance improvement percentage
    if avg_performance["non_reasoning"] > 0:
        improvement_pct = ((avg_performance["reasoning"] / avg_performance["non_reasoning"]) - 1) * 100
    else:
        improvement_pct = 0
    
    # Test if improvement is within hypothesized range (30-50%)
    hypothesis_confirmed = 30 <= improvement_pct <= 50
    
    # Return results
    return {
        "results_by_style": results,
        "avg_performance": avg_performance,
        "improvement_pct": improvement_pct,
        "hypothesis_confirmed": hypothesis_confirmed
    }

def analyze_feature_importance(model, graph_data):
    """Analyze which features are most important for classification"""
    # Create a baseline prediction
    model.eval()
    with torch.no_grad():
        baseline_scores = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        baseline_probs = torch.sigmoid(baseline_scores).detach().cpu().numpy()
    
    # Feature names
    feature_names = [
        "Performance", 
        "Nash Welfare", 
        "Exploitability", 
        "Raw Regret", 
        "Consistency", 
        "Variance"
    ]
    
    # Test importance by zeroing out each feature
    importance_scores = []
    
    for i in range(graph_data.x.shape[1]):
        # Copy features and zero out one dimension
        modified_x = graph_data.x.clone()
        modified_x[:, i] = 0
        
        # Get predictions with modified features
        with torch.no_grad():
            modified_scores = model(modified_x, graph_data.edge_index, graph_data.edge_attr)
            modified_probs = torch.sigmoid(modified_scores).detach().cpu().numpy()
        
        # Calculate change in predictions
        importance = np.mean(np.abs(baseline_probs - modified_probs))
        importance_scores.append(importance)
    
    # Create dictionary of feature importance
    feature_importance = {
        feature_names[i]: importance_scores[i] 
        for i in range(len(feature_names))
    }
    
    return feature_importance

def visualize_results(evaluation_results, feature_importance):
    """Create visualizations of the results"""
    start_time = time.time()
    
    # 1. Performance comparison by reasoning style
    plt.figure(figsize=(10, 6))
    styles = list(evaluation_results["avg_performance"].keys())
    performances = [evaluation_results["avg_performance"][style] for style in styles]
    
    plt.bar(styles, performances)
    plt.ylabel("Average Performance")
    plt.title(f"Performance by Reasoning Style (Improvement: {evaluation_results['improvement_pct']:.1f}%)")
    plt.savefig("performance_comparison.png")
    
    # 2. Feature importance
    plt.figure(figsize=(10, 6))
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    
    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]
    
    plt.bar(sorted_features, sorted_importances)
    plt.ylabel("Importance Score")
    plt.title("Feature Importance for Strategic Classification")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    
    # 3. Agent performance vs probability
    plt.figure(figsize=(10, 6))
    
    for style, data in evaluation_results["results_by_style"].items():
        plt.scatter(
            data["probabilities"],
            data["performance"],
            label=style,
            alpha=0.7
        )
    
    plt.xlabel("Reasoning Probability")
    plt.ylabel("Performance")
    plt.title("Relationship Between Reasoning Score and Performance")
    plt.legend()
    plt.savefig("reasoning_vs_performance.png")
    
    print(f"Visualizations created in {timedelta(seconds=int(time.time() - start_time))}")

def print_diagnostic_info(graph_data, ne_regret_matrix, reasoning_styles):
    """Print diagnostic information about the data"""
    print("\n" + "="*60)
    print(" " * 17 + "DIAGNOSTIC INFORMATION")
    print("="*60 + "\n")
    
    # Print NE regret values by reasoning style
    print("📊 NE Regret Values by Reasoning Style:\n")
    
    reasoning_agents = []
    non_reasoning_agents = []
    
    for i, agent in enumerate(graph_data.agent_names):
        style = reasoning_styles.get(agent) or classify_agent_reasoning_style(agent)
        
        # Get regret values
        regret_value = 0.01  # Default
        exploitability_score = 0.0
        
        if agent in ne_regret_matrix.index:
            if 'Mean NE Regret' in ne_regret_matrix.columns:
                regret_value = ne_regret_matrix.loc[agent, 'Mean NE Regret']
            elif 'NE Regret' in ne_regret_matrix.columns:
                regret_value = ne_regret_matrix.loc[agent, 'NE Regret']
            
            # Calculate exploitability score
            exploitability_score = np.exp(-10 * regret_value) if regret_value > 0 else 1.0
            
            # Format and add to appropriate list
            if style == "reasoning":
                reasoning_agents.append((agent, regret_value, exploitability_score))
            else:
                non_reasoning_agents.append((agent, regret_value, exploitability_score))
    
    # Print reasoning agents
    print("  🤔 REASONING AGENTS:")
    for agent, regret, exploit in reasoning_agents:
        print(f"    {agent}: {regret:.6f} (exploit. score: {exploit:.4f}, raw: {regret})")
    
    # Print non-reasoning agents
    print("\n  🎯 NON-REASONING AGENTS:")
    for agent, regret, exploit in non_reasoning_agents:
        print(f"    {agent}: {regret:.6f} (exploit. score: {exploit:.4f}, raw: {regret})")
    
    # Calculate and print summary statistics
    avg_reasoning_regret = np.mean([r for _, r, _ in reasoning_agents]) if reasoning_agents else 0
    avg_non_reasoning_regret = np.mean([r for _, r, _ in non_reasoning_agents]) if non_reasoning_agents else 0
    
    regret_ratio = avg_reasoning_regret / avg_non_reasoning_regret if avg_non_reasoning_regret > 0 else 0
    
    avg_reasoning_exploit = np.mean([e for _, _, e in reasoning_agents]) if reasoning_agents else 0
    avg_non_reasoning_exploit = np.mean([e for _, _, e in non_reasoning_agents]) if non_reasoning_agents else 0
    
    exploit_diff = 100 * (avg_reasoning_exploit - avg_non_reasoning_exploit) / avg_non_reasoning_exploit if avg_non_reasoning_exploit > 0 else 0
    
    print("\n  📈 REGRET SUMMARY:")
    print(f"    Avg reasoning regret: {avg_reasoning_regret:.6f}")
    print(f"    Avg non-reasoning regret: {avg_non_reasoning_regret:.6f}")
    print(f"    Ratio (reasoning/non-reasoning): {regret_ratio:.4f}")
    print(f"    Exploitability difference: {exploit_diff:.2f}%")
    
    # Print key node features sample
    print("\n📊 Key Node Features Sample:")
    print("    Agent                                          |   Performance   |     Welfare     |  Exploit. Score |   Consistency   |     Variance    |    Raw Regret  ")
    print("    ----------------------------------------------------------------------------------------------------")
    
    # Print features for a sample of agents
    sample_agents = reasoning_agents[:2] + non_reasoning_agents[:2]
    for agent, regret, exploit in sample_agents:
        idx = graph_data.agent_names.index(agent)
        style_indicator = "(reasoning)" if agent in [a for a, _, _ in reasoning_agents] else "(non_reasoning)"
        
        perf = graph_data.x[idx, 0].item()
        welfare = graph_data.x[idx, 1].item()
        exploit_score = graph_data.x[idx, 2].item()
        consistency = graph_data.x[idx, 4].item()
        variance = graph_data.x[idx, 5].item()
        
        print(f"    {agent} {style_indicator} |          {perf:.4f} |        {welfare:.4f} |          {exploit_score:.4f} |          {consistency:.4f} |          {variance:.4f} |          {regret:.4f}")
    
    print("\n" + "="*60 + "\n")

def main():
    start_time = time.time()
    print("Starting strategic adaptability analysis...")
    
    # Step 1: Load and process game data
    # Placeholder for loading performance matrix, welfare matrix, etc.
    # In a real implementation, you would load these from your game data
    
    # For demo purposes, create random matrices
    num_agents = 9
    agent_names = [
        "anthropic_sonnet_3.7_reasoning_circle_0",
        "openai_o3_mini_circle_0",
        "anthropic_3.7_sonnet_circle_5",
        "anthropic_3.7_sonnet_circle_6",
        "gemini_2.0_flash_circle_2",
        "gemini_2.0_flash_circle_5",
        "openai_4o_circle_4",
        "openai_4o_circle_5",
        "openai_4o_circle_6"
    ]
    
    performance_matrix = pd.DataFrame(
        np.random.uniform(0.5, 1.0, size=(num_agents, num_agents)),
        index=agent_names,
        columns=agent_names
    )
    
    welfare_matrix = pd.DataFrame(
        np.random.uniform(300, 700, size=(num_agents, num_agents)),
        index=agent_names,
        columns=agent_names
    )
    
    # Create bootstrap stats with regret information
    regret_values = np.random.uniform(0.01, 0.1, size=num_agents)
    bootstrap_stats = pd.DataFrame(
        regret_values,
        index=agent_names,
        columns=['NE Regret']
    )
    
    # Create variance matrix
    variance_matrix = pd.DataFrame(
        np.random.uniform(0, 1, size=(num_agents, num_agents)),
        index=agent_names,
        columns=agent_names
    )
    
    # Create reasoning styles dictionary
    reasoning_styles = {}
    for agent in agent_names:
        reasoning_styles[agent] = classify_agent_reasoning_style(agent)
    
    # Step 2: Build graph representation
    print("Building strategic graph...")
    graph_data = build_strategic_graph(
        performance_matrix,
        welfare_matrix,
        bootstrap_stats,
        variance_matrix,
        reasoning_styles
    )
    
    # Step 3: Print diagnostic information
    print_diagnostic_info(graph_data, bootstrap_stats, reasoning_styles)
    
    # Step 4: Train the model
    print("Training strategic GNN model...")
    model = train_model(graph_data, reasoning_styles)
    print("Model training completed")
    
    # Step 5: Evaluate strategic adaptability
    print("Evaluating strategic adaptability...")
    
    # Create performance data dictionary
    performance_data = {
        agent: performance_matrix.loc[agent, :].mean()
        for agent in agent_names
    }
    
    evaluation_results = evaluate_strategic_adaptability(
        model,
        graph_data,
        reasoning_styles,
        performance_data
    )
    
    # Step 6: Analyze feature importance
    feature_importance = analyze_feature_importance(model, graph_data)
    
    # Step 7: Print results
    print("\nResults:")
    print(f"Average performance by reasoning style:")
    for style, perf in evaluation_results["avg_performance"].items():
        print(f"  {style}: {perf:.2f}")
    
    print(f"Performance improvement: {evaluation_results['improvement_pct']:.2f}%")
    print(f"Hypothesis confirmed: {evaluation_results['hypothesis_confirmed']}")
    
    print("\nFeature importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    # Step 8: Create visualizations
    print("\nCreating visualizations...")
    visualize_results(evaluation_results, feature_importance)
    
    # Step 9: Print total time
    total_time = time.time() - start_time
    print(f"\nStrategic adaptability analysis complete! Total time: {timedelta(seconds=int(total_time))}")

if __name__ == "__main__":
    main() 