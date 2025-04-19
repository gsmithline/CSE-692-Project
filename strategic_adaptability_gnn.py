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
from torch.optim import * # Ensure optimizers are imported if needed
import torch.optim.lr_scheduler as lr_scheduler # Ensure schedulers are imported

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
    calculate_acceptance_ratio,
    # load_or_create_bootstrap_results # Assuming this helper exists - REMOVED
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
        
        # Edge feature compression (ensure input size matches edge_attr)
        # Original used Linear(8, 1). Check build_strategic_graph's edge_attr size.
        # If edge_attr has 8 features:
        self.edge_nn = nn.Sequential(
            nn.Linear(8, 1),  
            nn.Sigmoid() 
        )
        # If edge_attr has fewer features, adjust Linear layer input size.
        
        # GNN layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Output layers
        # Note: Original used global_mean_pool *before* the output layer,
        # implying graph-level classification. The MoE script did node-level.
        # Assuming node-level classification based on the goal:
        self.output = nn.Linear(hidden_channels, 1)  # Node-level binary classification
    
    def forward(self, x, edge_index, edge_attr):
        # Process edge attributes to get scalar edge weights
        # Check if edge_attr has the expected shape for self.edge_nn
        if edge_attr.shape[1] == 8: # Assuming 8 edge features as per original
            edge_weight = self.edge_nn(edge_attr).view(-1)
        else:
            # Handle case where edge_attr shape is different, e.g., use mean if no edge_nn
            print(f"Warning: edge_attr shape {edge_attr.shape} unexpected. Using uniform weights.")
            edge_weight = None # Or torch.ones(edge_index.shape[1], device=x.device)
        
        # First convolution
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training) # Keep dropout as in original
        
        # Second convolution
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        # Output (Node-level classification score)
        score = self.output(x)
        
        return score

def build_strategic_graph(performance_matrix, nash_welfare_matrix, bootstrap_stats, variance_matrix, reasoning_styles=None):
    """Build a graph representation of the strategic landscape"""
    agents = performance_matrix.index.tolist()
    
    # Try to extract NE regret from bootstrap_stats
    ne_regret_matrix = None
    
    if bootstrap_stats is not None:
        if hasattr(bootstrap_stats, 'columns'):
            for col in bootstrap_stats.columns:
                # Look for mean regret specifically if available
                if 'mean ne regret' in col.lower():
                    regret_column = col
                    ne_regret_matrix = bootstrap_stats
                    break
            # Fallback to any regret column
            if ne_regret_matrix is None:
                 for col in bootstrap_stats.columns:
                    if 'regret' in col.lower():
                        regret_column = col
                        ne_regret_matrix = bootstrap_stats
                    break
        elif isinstance(bootstrap_stats, dict) and 'regret' in bootstrap_stats:
            regret_values = bootstrap_stats['regret']
            ne_regret_matrix = pd.DataFrame(regret_values, index=agents, columns=['NE Regret'])
            regret_column = 'NE Regret'
    
    # If regret information not found, create dummy values
    if ne_regret_matrix is None:
        print("Warning: Could not find NE regret information. Using random dummy values.")
        dummy_regrets = np.random.uniform(0.001, 0.1, size=len(agents))
        ne_regret_matrix = pd.DataFrame(dummy_regrets, index=agents, columns=['NE Regret'])
        regret_column = 'NE Regret'
    elif regret_column is None: # Should have been set if ne_regret_matrix was found
        print("Warning: Regret matrix found but column name unknown. Using first column.")
        regret_column = ne_regret_matrix.columns[0]
    
    # Extract features for each agent (node)
    node_features = []
    max_perf = performance_matrix.values.max() # Calculate once
    max_welfare = nash_welfare_matrix.values.max() # Calculate once
    max_variance = variance_matrix.values.max() # Calculate once
    
    for agent in tqdm(agents, desc="Building Node Features", leave=False):
        # Performance metrics
        agent_perf_series = performance_matrix.loc[agent, :]
        perf_i = float(agent_perf_series.mean())
        norm_perf = perf_i / max_perf if max_perf > 0 else 0.0
        
        # Cooperation metrics
        agent_welfare_series = nash_welfare_matrix.loc[agent, :] if agent in nash_welfare_matrix.index else None
        welfare_i = float(agent_welfare_series.mean()) if agent_welfare_series is not None else 0.0
        norm_welfare = welfare_i / max_welfare if max_welfare > 0 else 0.0 # Normalize welfare
        
        # Exploitability metrics
        regret_i = 0.0
        if agent in ne_regret_matrix.index:
            try:
                 regret_i = float(ne_regret_matrix.loc[agent, regret_column])
            except KeyError:
                 print(f"Warning: Regret column '{regret_column}' not found for agent {agent}. Using 0.")
            except ValueError:
                 print(f"Warning: Non-numeric regret value for agent {agent}. Using 0.")
        
        exploitability_score = np.exp(.1 * regret_i) # k=10 scaling factor (adjust if needed)
        exploitability_score = max(exploitability_score, 1e-9)
        # Consistency metrics
        agent_variance_series = variance_matrix.loc[agent, :] if agent in variance_matrix.index else None
        variance_i = float(agent_variance_series.mean()) if agent_variance_series is not None else 0.0
        consistency_score = 1.0 / (1.0 + variance_i) if variance_i > 0 else 1.0 # Lower variance = higher score
        norm_variance = variance_i / max_variance if max_variance > 0 else 0.0 # Normalize variance
        
        # Agent features - ensure order matches feature_names later
        features = [
            norm_perf,                # Normalized performance
            norm_welfare,             # Normalized Nash welfare
            exploitability_score,     # Exploitability score (derived from regret)
            float(regret_i),          # Raw regret
            consistency_score,        # Consistency score (derived from variance)
            norm_variance             # Normalized Variance
        ]
        
        node_features.append(features)
    
    # Create edge relationships and features
    edge_index = []
    edge_features = []
    
    for i, agent_i in enumerate(tqdm(agents, desc="Building Edge Features", leave=False)):
        for j, agent_j in enumerate(agents):
            if i == j:  # Skip self-loops
                continue
            
            edge_index.append([i, j])
            
            # Initialize default edge features
            edge_attr = [0.0] * 8 # Match original SimpleGNN expectation
            
            try:
                # Performance features
                perf_ij = performance_matrix.loc[agent_i, agent_j] if agent_j in performance_matrix.columns else 0.0
                perf_ij = float(perf_ij)
                norm_perf_ij = perf_ij / max_perf if max_perf > 0 else 0.0
                
                perf_ji = performance_matrix.loc[agent_j, agent_i] if agent_i in performance_matrix.columns else 0.0
                perf_ji = float(perf_ji)
                norm_perf_ji = perf_ji / max_perf if max_perf > 0 else 0.0 # Normalize reciprocal perf
                
                # Welfare features
                welfare_ij = nash_welfare_matrix.loc[agent_i, agent_j] if (agent_i in nash_welfare_matrix.index and agent_j in nash_welfare_matrix.columns) else 0.0
                welfare_ij = float(welfare_ij)
                norm_welfare_ij = welfare_ij / max_welfare if max_welfare > 0 else 0.0 # Normalize
                
                # Variance features
                variance_ij = 0.0
                if agent_i in variance_matrix.index and agent_j in variance_matrix.columns:
                    variance_ij = variance_matrix.loc[agent_i, agent_j]
                    variance_ij = float(variance_ij)
                norm_variance_ij = variance_ij / max_variance if max_variance > 0 else 0.0 # Normalize
                
                # Edge features (ensure order/meaning is consistent)
                edge_attr = [
                    float(norm_perf_ij),         # Normalized performance i->j
                    float(norm_perf_ji),         # Normalized performance j->i
                    float(norm_welfare_ij),      # Normalized Nash welfare i->j
                    float(norm_variance_ij),     # Normalized Variance i->j
                    float(perf_ij - perf_ji),    # Performance difference i-j
                    float(perf_ij / max(1e-6, perf_ji)), # Performance ratio (avoid zero division)
                    float(1.0 if perf_ij > perf_ji else (-1.0 if perf_ji > perf_ij else 0.0)), # Win/Loss/Tie indicator
                    float(1.0) # Placeholder/bias term if needed
                ]
                
            except (ValueError, TypeError, KeyError) as e:
                print(f"Warning: Error creating edge features for {agent_i} -> {agent_j}: {e}. Using defaults.")
                # Keep default edge_attr = [0.0] * 8
            
            edge_features.append(edge_attr)
    
    # Convert to PyTorch Geometric Data object
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Add agent names to data object for easier lookup
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        agent_names=agents # Add agent names
    )
    
    print(f"Graph built: {data}")
    return data

def classify_agent_reasoning_style(agent_name):
    """Classify agent as reasoning or non-reasoning"""
    # Use cleaned display name for classification consistency
    display_name = get_display_name(agent_name) # Assumes get_display_name handles the raw name
    if any(pattern in display_name for pattern in REASONING_STYLES["reasoning"]):
        return "reasoning"
    # Check explicit non-reasoning patterns if needed, otherwise default
    # if any(pattern in display_name for pattern in REASONING_STYLES["non_reasoning"]):
    #     return "non_reasoning"
    return "non_reasoning"

def train_model(graph_data, reasoning_styles, num_epochs=200, lr=0.01, train_idx=None, val_idx=None):
    """Train the simple GNN model using supervised learning, returning the trained model."""
    num_nodes = graph_data.x.shape[0]
    
    # Use provided indices if available, otherwise create split
    if train_idx is None or val_idx is None:
        print("Creating new train/val split for this model instance.") # Should only happen if called directly
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        train_split_idx = int(0.8 * num_nodes) # Standard 80/20 split
        train_idx = indices[:train_split_idx]
        val_idx = indices[train_split_idx:]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
    else:
        # Use existing masks if provided
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        
    # Create labels - reasoning = 1, non_reasoning = 0
    y = torch.zeros(num_nodes, 1) # Target shape for node-level BCEWithLogitsLoss
    for i, agent in enumerate(graph_data.agent_names):
        # Ensure consistent labeling using the classification function
        style = reasoning_styles.get(agent, classify_agent_reasoning_style(agent))
        if style == "reasoning":
            y[i, 0] = 1.0 # Assign 1.0 to the first dimension
    
    # Move graph data and labels to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = graph_data.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    
    # Initialize model and optimizer
    model = SimpleGNN(graph_data.x.shape[1]).to(device) # Ensure model is on device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) # Keep weight decay
    criterion = nn.BCEWithLogitsLoss() # Standard loss for binary classification
    
    # Add scheduler and early stopping like in MoE script
    scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 10), gamma=0.5)
    best_val_acc = 0
    patience_counter = 0
    patience = 20 # Early stopping patience
    
    # Training loop
    model.train()
    progress_bar = tqdm(range(num_epochs), desc="Training SimpleGNN", leave=False)
    
    for epoch in progress_bar:
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr) # Get node scores
        
        # Ensure output and labels have compatible shapes for loss calculation
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Validation step
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
            
            # Early stopping check
        if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Optional: Save best model state
                # torch.save(model.state_dict(), 'best_simple_gnn_model.pth')
        else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        model.train() # Set back to train mode
    
    # Optional: Load best model state if saved
    # if os.path.exists('best_simple_gnn_model.pth'):
    #     model.load_state_dict(torch.load('best_simple_gnn_model.pth'))
    #     print("Loaded best model state for evaluation.")
    
    return model.cpu() # Return model on CPU

def evaluate_ensemble_strategic_adaptability(models, graph_data, reasoning_styles, performance_data):
    """Evaluate strategic adaptability using an ensemble of SimpleGNN models."""
    num_models = len(models)
    num_agents = len(graph_data.agent_names)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = graph_data.to(device)
    
    all_scores = torch.zeros(num_agents, 1, device=device)
    
    for model in tqdm(models, desc="Evaluating Ensemble SimpleGNN", leave=False):
        model.to(device)
        model.eval()
    with torch.no_grad():
        scores = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        all_scores += scores
        model.cpu() # Move back to CPU
    
    avg_scores = all_scores / num_models
    avg_probabilities = torch.sigmoid(avg_scores).cpu().numpy() # Get probabilities from averaged logits
    avg_scores_np = avg_scores.cpu().numpy()
    
    # Group results by reasoning style
    results = {
        "reasoning": {"agents": [], "scores": [], "probabilities": [], "performance": []},
        "non_reasoning": {"agents": [], "scores": [], "probabilities": [], "performance": []}
    }
    
    for i, agent in enumerate(graph_data.agent_names):
        # Use cleaned name for matching performance data and consistent labeling
        cleaned_agent_name = get_display_name(agent)
        style = reasoning_styles.get(cleaned_agent_name, classify_agent_reasoning_style(cleaned_agent_name)) # Use cleaned name for lookup
        
        results[style]["agents"].append(cleaned_agent_name) # Store cleaned name
        results[style]["scores"].append(avg_scores_np[i].item())
        results[style]["probabilities"].append(avg_probabilities[i].item())
        # Performance data should ideally use cleaned names already
        results[style]["performance"].append(performance_data.get(cleaned_agent_name, 0))
    
    # Calculate average performance by reasoning style
    avg_performance = {}
    for style, data in results.items():
        valid_performances = [p for p in data["performance"] if isinstance(p, (int, float)) and not np.isnan(p)]
        if valid_performances:
            avg_performance[style] = np.mean(valid_performances)
        else:
            avg_performance[style] = 0
    
    # Calculate performance improvement percentage
    improvement_pct = 0
    if "reasoning" in avg_performance and "non_reasoning" in avg_performance and avg_performance["non_reasoning"] > 0:
        improvement_pct = ((avg_performance["reasoning"] / avg_performance["non_reasoning"]) - 1) * 100
    else:
        improvement_pct = float('inf') if avg_performance.get("reasoning", 0) > 0 else 0 # Handle zero denominator
    
    # Test if improvement is within hypothesized range (e.g., 30-50%)
    # hypothesis_confirmed = 30 <= improvement_pct <= 50 # Hypothesis might change
    
    return {
        "results_by_style": results,
        "avg_performance": avg_performance,
        "improvement_pct": improvement_pct,
        # "hypothesis_confirmed": hypothesis_confirmed
    }

def analyze_ensemble_feature_importance(models, graph_data, criterion):
    """Analyze feature importance using permutation importance averaged over the ensemble."""
    num_models = len(models)
    num_features = graph_data.x.shape[1]
    num_agents = graph_data.x.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = graph_data.to(device)
    
    # Feature names must match the order in build_strategic_graph
    feature_names = [
        "Perf_Norm",
        "Welfare_Norm", # Changed from Coop_Welfare if normalized
        "Exploit_Score",
        "Exploit_RawRegret",
        "Adapt_Consistency", # Changed from Consistency
        "Variance_Norm" # Changed from Variance
    ]
    if len(feature_names) != num_features:
         print(f"Warning: Mismatch between expected features ({len(feature_names)}) and actual ({num_features}). Using generic names.")
         feature_names = [f"Feature_{i}" for i in range(num_features)]
    
    total_importance_scores = np.zeros(num_features)
    
    # Create dummy labels for loss calculation consistency if needed
    # Better: Use actual labels derived from reasoning_styles if available
    # Recreate labels here based on reasoning_styles for accuracy
    y_labels = torch.zeros(num_agents, 1, dtype=torch.float, device=device)
    temp_reasoning_styles = {name: classify_agent_reasoning_style(name) for name in graph_data.agent_names}
    for i, agent in enumerate(graph_data.agent_names):
        if temp_reasoning_styles.get(agent) == "reasoning":
            y_labels[i, 0] = 1.0
    
    print("Analyzing feature importance across SimpleGNN ensemble...")
    for model in tqdm(models, desc="Feature Importance (SimpleGNN)", leave=False):
        model.to(device)
        model.eval()
        
        # 1. Calculate baseline loss
        with torch.no_grad():
            baseline_scores = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            # Use the actual labels for calculating baseline loss
            baseline_loss = criterion(baseline_scores, y_labels).item()
        
        model_importance = np.zeros(num_features)
        for i in range(num_features):
            original_feature = graph_data.x[:, i].clone()
            permuted_indices = torch.randperm(num_agents, device=device)
            graph_data.x[:, i] = graph_data.x[permuted_indices, i] # Permute feature i
            
            with torch.no_grad():
                permuted_scores = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
                # Use actual labels for permuted loss calculation
                permuted_loss = criterion(permuted_scores, y_labels).item()
            
            # Importance = increase in loss after permutation
            model_importance[i] = permuted_loss - baseline_loss
            
            # Restore original feature
            graph_data.x[:, i] = original_feature
        
        total_importance_scores += model_importance
        model.cpu() # Move back to CPU
    
    # Average importance over models
    avg_importance_scores = total_importance_scores / num_models
    
    feature_importance = {
        feature_names[i]: avg_importance_scores[i]
        for i in range(len(feature_names))
    }
    
    return feature_importance

def visualize_results(evaluation_results, feature_importance):
    """Create visualizations of the SimpleGNN ensemble results."""
    start_time = time.time()
    prefix = "simple_gnn_ensemble_" # Add prefix to filenames
    
    # 1. Performance comparison by reasoning style
    plt.figure(figsize=(10, 6))
    styles = list(evaluation_results["avg_performance"].keys())
    performances = [evaluation_results["avg_performance"][style] for style in styles]
    plt.bar(styles, performances)
    plt.ylabel("Average Performance (Ensemble)")
    plt.title(f"SimpleGNN Ensemble Performance by Reasoning Style (Improvement: {evaluation_results['improvement_pct']:.1f}%)")
    # Add text labels
    for i, v in enumerate(performances):
        plt.text(i, v + (max(performances) * 0.02), f"{v:.1f}", ha='center')
    plt.ylim(0, max(performances) * 1.1)
    plt.savefig(f"{prefix}performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved performance comparison plot.")
    
    # 2. Feature importance
    if feature_importance:
        plt.figure(figsize=(10, 7)) # Adjusted size for potentially longer labels
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        plt.bar(sorted_features, sorted_importances)
        plt.ylabel("Average Importance Score (Ensemble)")
        plt.title("SimpleGNN Ensemble Feature Importance (Permutation Method)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{prefix}feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved feature importance plot.")
    else:
        print("Skipping feature importance plot (no data).")
    
    # 3. Agent performance vs probability score
    plt.figure(figsize=(12, 7)) # Wider figure
    markers = {'reasoning': 'X', 'non_reasoning': 'o'} # Different markers
    colors = {'reasoning': 'red', 'non_reasoning': 'blue'} # Different colors
    
    all_probs = []
    all_perf = []
    
    for style, data in evaluation_results["results_by_style"].items():
        plt.scatter(
            data["probabilities"],
            data["performance"],
            label=style,
            alpha=0.7,
            marker=markers[style],
            color=colors[style],
            s=50 # Slightly larger markers
        )
        all_probs.extend(data["probabilities"])
        all_perf.extend(data["performance"])
        
        # Add agent name annotations
        for i, txt in enumerate(data["agents"]):
             # Shorten long names if necessary
            display_txt = txt if len(txt) < 25 else txt[:22] + "..."
            plt.annotate(display_txt, (data["probabilities"][i], data["performance"][i]), fontsize=8, alpha=0.8)
    
    plt.xlabel("Average Reasoning Probability (Ensemble)")
    plt.ylabel("Agent Average Performance")
    plt.title("SimpleGNN Ensemble: Agent Performance vs. Reasoning Score")
    plt.legend()
    # Optional: Add threshold line if meaningful
    # plt.axvline(x=0.5, color='grey', linestyle='--', label='Threshold (0.5)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{prefix}reasoning_vs_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved reasoning vs performance plot.")
    
    print(f"Visualizations created in {timedelta(seconds=int(time.time() - start_time))}")

def print_diagnostic_info(graph_data, ne_regret_matrix, reasoning_styles):
    """Print diagnostic information about the data"""
    print("\n" + "="*60)
    print(" " * 17 + "DIAGNOSTIC INFORMATION")
    print("="*60 + "\n")
    
    # Determine the correct regret column name
    regret_column = None
    if ne_regret_matrix is not None:
        if 'Mean NE Regret' in ne_regret_matrix.columns:
            regret_column = 'Mean NE Regret'
        elif 'NE Regret' in ne_regret_matrix.columns:
            regret_column = 'NE Regret'
        elif len(ne_regret_matrix.columns) > 0:
            regret_column = ne_regret_matrix.columns[0] # Fallback
    
    # Print NE regret values by reasoning style
    print("📊 NE Regret Values by Reasoning Style:\n")
    
    reasoning_agents_stats = []
    non_reasoning_agents_stats = []
    
    # Use graph_data.agent_names as the definitive list
    for i, agent in enumerate(graph_data.agent_names):
        style = reasoning_styles.get(agent, classify_agent_reasoning_style(agent)) # Use cleaned name
        
        # Get regret values
        regret_value = np.nan # Default to NaN
        exploitability_score_node = np.nan
        raw_regret_node = np.nan
        
        if agent in ne_regret_matrix.index and regret_column:
            try:
                regret_value = float(ne_regret_matrix.loc[agent, regret_column])
            except (ValueError, TypeError):
                 print(f"Warning: Could not convert regret '{ne_regret_matrix.loc[agent, regret_column]}' to float for {agent}")
                 regret_value = np.nan # Ensure it's NaN on conversion error
            
        # Get features directly from graph_data if they exist
        if graph_data.x.shape[1] >= 6: # Ensure features exist
            exploitability_score_node = graph_data.x[i, 2].item() # Feature index 2
            raw_regret_node = graph_data.x[i, 3].item() # Feature index 3
            # consistency_node = graph_data.x[i, 4].item() # Feature index 4
            # variance_node = graph_data.x[i, 5].item() # Feature index 5
            
            # Format and add to appropriate list
            if style == "reasoning":
                reasoning_agents_stats.append((agent, regret_value, exploitability_score_node, raw_regret_node))
            else:
                non_reasoning_agents_stats.append((agent, regret_value, exploitability_score_node, raw_regret_node))
    
    # Print reasoning agents
    print("  🤔 REASONING AGENTS:")
    for agent, regret, exploit_node, raw_node in reasoning_agents_stats:
        print(f"    {agent}: Regret={regret:.6f}, ExploitScore(Node)={exploit_node:.4f}, RawRegret(Node)={raw_node:.4f}")
    
    # Print non-reasoning agents
    print("\n  🎯 NON-REASONING AGENTS:")
    for agent, regret, exploit_node, raw_node in non_reasoning_agents_stats:
        print(f"    {agent}: Regret={regret:.6f}, ExploitScore(Node)={exploit_node:.4f}, RawRegret(Node)={raw_node:.4f}")
    
    # Calculate and print summary statistics (handle NaNs)
    reasoning_regrets = [r for _, r, _, _ in reasoning_agents_stats if not np.isnan(r)]
    non_reasoning_regrets = [r for _, r, _, _ in non_reasoning_agents_stats if not np.isnan(r)]
    
    avg_reasoning_regret = np.mean(reasoning_regrets) if reasoning_regrets else np.nan
    avg_non_reasoning_regret = np.mean(non_reasoning_regrets) if non_reasoning_regrets else np.nan
    
    regret_ratio = avg_reasoning_regret / avg_non_reasoning_regret if avg_non_reasoning_regret > 0 else np.inf
    
    print("\n  📈 REGRET SUMMARY (from NE Analysis):")
    print(f"    Avg reasoning regret: {avg_reasoning_regret:.6f}")
    print(f"    Avg non-reasoning regret: {avg_non_reasoning_regret:.6f}")
    print(f"    Ratio (reasoning/non-reasoning): {regret_ratio:.4f}")
    
    # Print key node features sample
    print("\n📊 Key Node Features Sample (from graph_data.x):")
    feature_headers = ["PerfNorm", "WelfNorm", "ExploitScore", "RawRegret", "Consistency", "VarNorm"]
    print("    Agent".ljust(45) + "".join([f" | {h.center(12)}" for h in feature_headers]))
    print("    " + "-" * (45 + len(feature_headers) * 15))
    
    sample_agents = reasoning_agents_stats[:2] + non_reasoning_agents_stats[:2]
    for agent, _, _, _ in sample_agents:
        try:
            idx = graph_data.agent_names.index(agent)
            style_indicator = "(reas)" if agent in [a for a, _, _, _ in reasoning_agents_stats] else "(non-reas)"
            features = graph_data.x[idx].tolist() # Get features for this agent
            print(f"    {agent[:35].ljust(35)} {style_indicator} " + "".join([f" | {f:12.4f}" for f in features]))
        except ValueError:
             print(f"    Agent {agent} not found in graph_data.agent_names for feature printing.")
    
    print("\n" + "="*60 + "\n")

def load_or_create_bootstrap_results(performance_matrix, num_bootstrap_samples=100, confidence_level=0.95):
    """Loads or runs bootstrap analysis, ensuring the returned DataFrame index uses cleaned agent names."""
    cleaned_agent_names = performance_matrix.index.tolist() # Assume input index is cleaned
    try:
        bootstrap_results, bootstrap_stats, _, ne_strategy_df = run_nash_analysis(
            performance_matrix,
            num_bootstrap_samples=num_bootstrap_samples,
            confidence_level=confidence_level
        )
        # --- FIX: Ensure bootstrap_stats index is correctly set --- 
        if isinstance(bootstrap_stats, pd.DataFrame):
            # If the index isn't already the agent names, try setting it
            if not bootstrap_stats.index.equals(performance_matrix.index):
                if len(bootstrap_stats) == len(cleaned_agent_names):
                    print("Setting bootstrap_stats index to match input performance matrix index.")
                    bootstrap_stats.index = performance_matrix.index
                else:
                    print("Warning: Length mismatch between bootstrap_stats and agent names. Index not set.")
        return bootstrap_results, bootstrap_stats, ne_strategy_df
    except Exception as e:
        print(f"Error running bootstrap analysis: {e}")
        print("Creating simplified bootstrap results instead")

        bootstrap_stats = pd.DataFrame({
            'Mean NE Regret': [0.01] * len(cleaned_agent_names),  # Dummy values
            'Std NE Regret': [0.005] * len(cleaned_agent_names),
            'Mean Expected Utility': performance_matrix.mean(axis=1).values,
            'Std Expected Utility': performance_matrix.std(axis=1).values
        }, index=cleaned_agent_names) # Set index directly

        bootstrap_results = {
            'ne_regret': [[0.01] * len(cleaned_agent_names)] * 10,
            'ne_strategy': [np.ones(len(cleaned_agent_names)) / len(cleaned_agent_names)] * 10,
            'expected_utility': [performance_matrix.mean(axis=1).values] * 10
        }

      
        ne_strategy_df = pd.DataFrame({
            'Nash Probability': np.ones(len(cleaned_agent_names)) / len(cleaned_agent_names)  # Uniform strategy
        }, index=cleaned_agent_names) # Set index directly

        return bootstrap_results, bootstrap_stats, ne_strategy_df

# Modify main function for ensemble and real data
def main(num_ensemble=5, num_epochs=150, lr=1e-3): # Add ensemble parameters, adjust defaults
    """Main function to run the SimpleGNN ensemble analysis"""
    start_time = time.time()
    print(f"Starting strategic adaptability analysis using Ensemble of {num_ensemble} SimpleGNNs")
    
    # Step 1: Load and process real game data (Adapted from MoE script)
    # Configure progress bar if desired
    # main_steps = ["Data Processing", "Matrix Creation", "Welfare Matrices", "Matrix Cleaning",
    #               "Nash Analysis", "Agent Classification", "Graph Building", "Model Training",
    #               "Evaluation", "Visualization"]
    # main_progress = tqdm(main_steps, desc="Analysis Progress", position=0)
    
    print("\nProcessing game data...")
    process_start = time.time()
    # Make sure this path is correct for your setup
    crossplay_dir = "crossplay/game_matrix_1"
    all_results, agent_performance, agent_final_rounds, agent_game_counts, _ = process_all_games(
        crossplay_dir, discount_factor=0.98
    )
    print(f"Game data processing completed in {timedelta(seconds=int(time.time() - process_start))}")
    # main_progress.update(1)
    
    print("\nCreating performance matrices...")
    matrices_start = time.time()
    performance_matrices = create_performance_matrices(all_results, agent_performance, agent_final_rounds)
    all_agents_raw = sorted(list(performance_matrices['overall_agent_performance'].keys()))
    print(f"Performance matrices created in {timedelta(seconds=int(time.time() - matrices_start))}")
    # main_progress.update(1)
    
    print("\nCreating welfare matrices...")
    welfare_start = time.time()
    # Ensure this function provides necessary welfare data for build_strategic_graph
    global_max_nash_welfare, _ = compute_global_max_values(num_samples=1000)
    welfare_matrices = create_welfare_matrices(all_results, all_agents_raw, global_max_nash_welfare)
    print(f"Welfare matrices created in {timedelta(seconds=int(time.time() - welfare_start))}")
    # main_progress.update(1)
    
    print("\nCleaning matrix names...")
    clean_start = time.time()
    cleaned_matrices = {}
    # Define which matrices to clean
    matrices_to_clean = {**performance_matrices, **welfare_matrices}
    # Add overall_agent_performance dict separately as it's not a DataFrame
    cleaned_matrices['overall_agent_performance'] = matrices_to_clean.pop('overall_agent_performance', {})
    
    matrix_cleaning_progress = tqdm(matrices_to_clean.items(), desc="Cleaning matrices", leave=False)
    for name, matrix in matrix_cleaning_progress:
        if isinstance(matrix, pd.DataFrame): # Only clean DataFrames
            cleaned_matrices[name] = clean_matrix_names(matrix, get_display_name)
        else:
            cleaned_matrices[name] = matrix # Keep non-DataFrames as is
    
    # Get cleaned agent names
    if 'performance_matrix' in cleaned_matrices:
        all_agents_cleaned = sorted(cleaned_matrices['performance_matrix'].index.tolist())
        # Ensure the index is also set correctly for the performance matrix used below
        performance_matrix = cleaned_matrices['performance_matrix']
    else:
         # Fallback if performance_matrix isn't cleaned or available
        all_agents_cleaned = sorted([get_display_name(agent) for agent in all_agents_raw])
        print("Warning: Cleaned performance matrix not found. Attempting to use raw performance matrix.")
        # Try to use the original performance matrix if cleaned one is missing
        # This assumes performance_matrices dict exists and has 'performance_matrix'
        performance_matrix = performance_matrices.get('performance_matrix')
        if performance_matrix is None:
             print("Error: Cannot find any performance matrix. Exiting.")
             sys.exit(1)
        # Clean this raw matrix just in case
        performance_matrix = clean_matrix_names(performance_matrix, get_display_name)

    print(f"Matrix names cleaned in {timedelta(seconds=int(time.time() - clean_start))}")
    # main_progress.update(1)
    
    print("\nRunning Nash equilibrium analysis...")
    nash_start = time.time()
    # Use the cleaned performance matrix for Nash analysis
    perf_matrix_for_nash = cleaned_matrices.get('performance_matrix')
    if perf_matrix_for_nash is None:
        print("Error: Cleaned performance matrix not found for Nash analysis. Exiting.")
        sys.exit(1)

    # --- START FIX: Clean the matrix before passing to Nash analysis ---
    print("Validating performance matrix for Nash analysis...")
    # Ensure numeric types, handle potential NaNs or Infs
    perf_matrix_for_nash = perf_matrix_for_nash.apply(pd.to_numeric, errors='coerce') # Convert non-numeric to NaN
    original_nan_count = perf_matrix_for_nash.isnull().sum().sum()
    if original_nan_count > 0:
        print(f"Warning: Found {original_nan_count} NaN values in performance matrix. Attempting to fill with row/column means.")
        # Impute NaNs - more robust strategies exist, this is a basic one
        # Fill with row mean first, then column mean for any remaining NaNs
        perf_matrix_for_nash = perf_matrix_for_nash.apply(lambda row: row.fillna(row.mean()), axis=1)
        perf_matrix_for_nash = perf_matrix_for_nash.apply(lambda col: col.fillna(col.mean()), axis=0)
        # Check if any NaNs remain (e.g., if a whole row/col was NaN)
        remaining_nan_count = perf_matrix_for_nash.isnull().sum().sum()
        if remaining_nan_count > 0:
            print(f"Warning: {remaining_nan_count} NaNs remain after basic imputation. Filling with global mean.")
            global_mean = perf_matrix_for_nash.mean().mean()
            perf_matrix_for_nash.fillna(global_mean, inplace=True)

    # Check for infinite values
    inf_count = np.isinf(perf_matrix_for_nash).sum().sum()
    if inf_count > 0:
        print(f"Warning: Found {inf_count} infinite values. Replacing with large finite number.")
        perf_matrix_for_nash.replace([np.inf, -np.inf], 1e9, inplace=True) # Replace with a large number

    print("Performance matrix validation complete.")
    # --- END FIX ---

    # Use the helper function if available, otherwise run nash_analysis directly
    # Adjust parameters as needed
    if 'load_or_create_bootstrap_results' in globals():
         bootstrap_results, bootstrap_stats, ne_strategy_df = load_or_create_bootstrap_results(perf_matrix_for_nash)
    else:
         # Assuming run_nash_analysis returns these (adjust if needed)
         bootstrap_results, bootstrap_stats, _, ne_strategy_df = run_nash_analysis(
             perf_matrix_for_nash, num_bootstrap_samples=100, confidence_level=0.95 # Reduced samples for speed
         )
    # Ensure bootstrap_stats index uses cleaned names if it's a DataFrame
    if isinstance(bootstrap_stats, pd.DataFrame):
        # --- REVERT CHECK: Remove the isinstance check, assume index is now correct ---
        # The mapping might still be useful if run_nash_analysis slightly mangles names
        # that get_display_name can fix.
        try:
            # Ensure index is strings before mapping (as a safeguard)
            if all(isinstance(idx, str) for idx in bootstrap_stats.index):
                 bootstrap_stats.index = bootstrap_stats.index.map(get_display_name)
            else:
                 print("Warning: bootstrap_stats index still contains non-strings after load_or_create. Cannot map.")
        except TypeError as e:
            print(f"Warning: TypeError during final bootstrap_stats index mapping: {e}. Skipping.")
        except Exception as e:
            print(f"Warning: Unexpected error during final bootstrap_stats index mapping: {e}. Skipping.")
        # --- END REVERT ---

    print(f"Nash equilibrium analysis completed in {timedelta(seconds=int(time.time() - nash_start))}")
    # main_progress.update(1)
    
    # Extract data needed for graph construction using CLEANED matrices
    performance_matrix = cleaned_matrices.get('performance_matrix')
    nash_welfare_matrix = cleaned_matrices.get('nash_welfare_matrix')
    variance_matrix = cleaned_matrices.get('variance_matrix')
    
    if performance_matrix is None or nash_welfare_matrix is None or variance_matrix is None:
        print("Error: Missing cleaned matrices required for graph building. Exiting.")
        sys.exit(1)
    
    # Step 2: Classify agents by reasoning style using CLEANED names
    print("\nClassifying agents by reasoning style...")
    reasoning_styles = {} # Dictionary mapping CLEANED agent name to style
    for agent in tqdm(all_agents_cleaned, desc="Classifying agents", leave=False):
        reasoning_styles[agent] = classify_agent_reasoning_style(agent) # classify expects cleaned name now
    
    # Print reasoning style classification counts
    reasoning_count = sum(1 for style in reasoning_styles.values() if style == "reasoning")
    non_reasoning_count = len(all_agents_cleaned) - reasoning_count
    print("\nAgent reasoning style classification:")
    print(f"  reasoning: {reasoning_count} agents")
    print(f"  non_reasoning: {non_reasoning_count} agents")
    # main_progress.update(1)
    
    # Step 3: Build strategic graph using CLEANED data
    print("\nBuilding strategic graph...")
    graph_start = time.time()
    graph_data = build_strategic_graph(
        performance_matrix,
        nash_welfare_matrix,
        bootstrap_stats, # Ensure this uses cleaned agent names as index if it's a DataFrame
        variance_matrix,
        reasoning_styles # Pass styles dict based on cleaned names
    )
    # Verify agent names in graph_data match reasoning_styles keys
    if not all(agent in reasoning_styles for agent in graph_data.agent_names):
         print("Warning: Mismatch between agent names in graph data and reasoning styles keys!")
         # Attempt to fix reasoning_styles keys if graph_data names are definitive
         reasoning_styles = {agent: classify_agent_reasoning_style(agent) for agent in graph_data.agent_names}
    
    print(f"Strategic graph built in {timedelta(seconds=int(time.time() - graph_start))}")
    # main_progress.update(1)
    
    # Optional: Print diagnostic info
    print_diagnostic_info(graph_data, bootstrap_stats, reasoning_styles)
    
    # Step 4: Train Ensemble of SimpleGNN Models
    print(f"\n--- Training Ensemble of {num_ensemble} SimpleGNN models ---")
    train_start_ensemble = time.time()
    trained_models = []
    num_nodes = graph_data.x.shape[0]
    
    # Create a fixed train/val split for the ensemble
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_split_idx = int(0.8 * num_nodes) # 80% train
    train_idx = indices[:train_split_idx]
    val_idx = indices[train_split_idx:]
    
    for i in range(num_ensemble):
        print(f"\n--- Training Model {i+1}/{num_ensemble} ---")
        # Pass the graph_data (which now contains cleaned agent names)
        # Pass the reasoning_styles dictionary (keyed by cleaned names)
        model = train_model(
            graph_data.clone(), # Pass a clone to avoid in-place modification issues
            reasoning_styles,
            num_epochs=num_epochs,
            lr=lr,
            train_idx=train_idx, # Pass fixed split
            val_idx=val_idx      # Pass fixed split
        )
        trained_models.append(model)
    print(f"\nEnsemble training completed in {timedelta(seconds=int(time.time() - train_start_ensemble))}")
    # main_progress.update(1)
    
    # Step 5: Evaluate Ensemble
    print("\nEvaluating strategic adaptability using the SimpleGNN ensemble...")
    eval_start = time.time()
    # Prepare performance data dictionary using cleaned names
    # The 'overall_agent_performance' dict from processing might still use raw names
    # We need to map it using get_display_name
    overall_perf_raw = cleaned_matrices.get('overall_agent_performance', {})
    performance_data_cleaned = { get_display_name(agent): perf
                                 for agent, perf in overall_perf_raw.items()}
    
    evaluation_results = evaluate_ensemble_strategic_adaptability(
        trained_models,
        graph_data,
        reasoning_styles, # Should be keyed by cleaned names
        performance_data_cleaned # Keyed by cleaned names
    )
    print(f"Ensemble evaluation completed in {timedelta(seconds=int(time.time() - eval_start))}")
    # main_progress.update(1)
    
    # Step 6: Analyze Ensemble Feature Importance
    print("\nAnalyzing ensemble feature importance...")
    analysis_start = time.time()
    criterion = nn.BCEWithLogitsLoss() # Define criterion for importance calculation
    ensemble_feature_importance = analyze_ensemble_feature_importance(
        trained_models, graph_data, criterion
    )
    print(f"Feature importance analysis completed in {timedelta(seconds=int(time.time() - analysis_start))}")
    
    # Step 7: Print Results
    print("\n--- SimpleGNN Ensemble Results ---")
    print(f"Average performance by reasoning style:")
    for style, avg_perf in evaluation_results["avg_performance"].items():
        print(f"  {style}: {avg_perf:.2f}")
    print(f"Performance improvement: {evaluation_results['improvement_pct']:.2f}%")
    # print(f"Hypothesis confirmed: {evaluation_results.get('hypothesis_confirmed', 'N/A')}") # If hypothesis testing is added back
    
    print("\nAverage Feature Importance (Ensemble):")
    if ensemble_feature_importance:
        for feature, importance in sorted(ensemble_feature_importance.items(), key=lambda item: item[1], reverse=True):
            print(f"  {feature}: {importance:.6f}") # More precision for importance
    else:
        print("  No feature importance data generated.")
    
    # Step 8: Create Visualizations
    print("\nCreating SimpleGNN ensemble visualizations...")
    visualize_results(evaluation_results, ensemble_feature_importance)
    # main_progress.update(1)
    
    # Step 9: Print total time
    total_time = time.time() - start_time
    # main_progress.close()
    print(f"\nSimpleGNN Ensemble strategic adaptability analysis complete! Total time: {timedelta(seconds=int(total_time))}")

if __name__ == "__main__":
    # Adjust ensemble size, epochs, lr as needed
    main(num_ensemble=5, num_epochs=200, lr=1e-3) 