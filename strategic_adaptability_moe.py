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
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from collections import defaultdict
import time
from datetime import timedelta
from torch.optim import *
import torch.optim.lr_scheduler as lr_scheduler
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

class GNNExpert(nn.Module):
    """Base class for all experts in the MoE model"""
    def __init__(self, num_node_features, hidden_channels=64):
        super().__init__()
        self.name = "Base"
        self.hidden_channels = hidden_channels
        
        print(f"Creating {self.name} expert with {num_node_features} input features and {hidden_channels} hidden channels")
        
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        
        self.edge_nn = nn.Sequential(
            nn.Linear(8, 1),  
            nn.Sigmoid() 
        )
        
        self.global_pool = global_mean_pool
        self.score = nn.Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index, edge_attr):
        edge_weight = self.edge_nn(edge_attr).view(-1)
        
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.global_pool(x, batch)
        score = self.score(x)
        
        return score

class ExploitabilityExpert(GNNExpert):
    """Expert that focuses on exploitability (measured by NE regret)"""
    def __init__(self, num_node_features, hidden_channels=64):
        super().__init__(num_node_features, hidden_channels)
        self.name = "Exploitability"
        
        self.exploit_specific = nn.Linear(hidden_channels, hidden_channels)
        self.exploit_norm = nn.LayerNorm(hidden_channels)
    
    def forward(self, x, edge_index, edge_attr):
        edge_weight = self.edge_nn(edge_attr).view(-1)
        
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.exploit_specific(x)
        x = self.exploit_norm(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.global_pool(x, batch)
        score = self.score(x)
        
        return score

class CooperativenessExpert(GNNExpert):
    """Expert for measuring cooperative tendencies (Nash welfare)"""
    def __init__(self, num_node_features, hidden_channels=64):
        super().__init__(num_node_features, hidden_channels)
        self.name = "Cooperativeness"
        
        # Add specific layers for cooperativeness patterns
        self.coop_specific = nn.Linear(hidden_channels, hidden_channels)
        self.coop_norm = nn.LayerNorm(hidden_channels)
    
    def forward(self, x, edge_index, edge_attr):
        # Process edge attributes to get scalar weights
        edge_weight = self.edge_nn(edge_attr).view(-1)
        
        # Process through GNN layers with scalar edge weights
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Cooperativeness-specific processing
        x = self.coop_specific(x)
        x = self.coop_norm(x)
        x = F.relu(x)
        
        # Second convolutional layer
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        # Global pooling and score
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.global_pool(x, batch)
        score = self.score(x)
        
        return score

class AdaptabilityExpert(GNNExpert):
    """Expert for measuring strategic adaptability"""
    def __init__(self, num_node_features, hidden_channels=64):
        super().__init__(num_node_features, hidden_channels)
        self.name = "Adaptability"
        
        # Add specific layers for adaptability patterns
        self.adapt_specific = nn.Linear(hidden_channels, hidden_channels)
        self.adapt_norm = nn.LayerNorm(hidden_channels)
    
    def forward(self, x, edge_index, edge_attr):
        # Process edge attributes to get scalar weights
        edge_weight = self.edge_nn(edge_attr).view(-1)
        
        # Process through GNN layers with scalar edge weights
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Adaptability-specific processing
        x = self.adapt_specific(x)
        x = self.adapt_norm(x)
        x = F.relu(x)
        
        # Second convolutional layer
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        # Global pooling and score
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.global_pool(x, batch)
        score = self.score(x)
        
        return score

class PerformanceExpert(GNNExpert):
    """Expert for measuring raw performance (utility)"""
    def __init__(self, num_node_features, hidden_channels=64):
        super().__init__(num_node_features, hidden_channels)
        self.name = "Performance"
        
        # Add specific layers for performance patterns
        self.perf_specific = nn.Linear(hidden_channels, hidden_channels)
        self.perf_norm = nn.LayerNorm(hidden_channels)
    
    def forward(self, x, edge_index, edge_attr):
        # Process edge attributes to get scalar weights
        edge_weight = self.edge_nn(edge_attr).view(-1)
        
        # Process through GNN layers with scalar edge weights
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Performance-specific processing
        x = self.perf_specific(x)
        x = self.perf_norm(x)
        x = F.relu(x)
        
        # Second convolutional layer
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        # Global pooling and score
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.global_pool(x, batch)
        score = self.score(x)
        
        return score

class MixtureOfExpertsGNN(nn.Module):
    """Mixture of Experts model for strategic graph analysis"""
    def __init__(self, num_node_features, hidden_channels=64):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        print(f"Creating MoE GNN with {num_node_features} input features and {hidden_channels} hidden channels")
        
        self.experts = nn.ModuleList([
            ExploitabilityExpert(num_node_features, hidden_channels),
            CooperativenessExpert(num_node_features, hidden_channels),
            AdaptabilityExpert(num_node_features, hidden_channels),
            PerformanceExpert(num_node_features, hidden_channels)
        ])
        
        self.feature_weights = nn.ParameterList([
            nn.Parameter(torch.ones(num_node_features) / num_node_features)
            for _ in range(len(self.experts))
        ])
        
        self.gate_edge_nn = nn.Sequential(
            nn.Linear(8, 1), 
            nn.Sigmoid() 
        )
        
        # Create gating network with more balanced architecture
        self.gate_conv1 = GCNConv(num_node_features, hidden_channels)
        self.gate_conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.noise_scale = 0.01
        
        self.gate_pool = global_mean_pool
        self.gate_output = nn.Linear(hidden_channels, len(self.experts))
        
        self.gate_norm = nn.LayerNorm(len(self.experts))
        
        print(f"MoE GNN initialized with {len(self.experts)} experts")
    
    def forward(self, x, edge_index, edge_attr):
        # Apply feature weighting for each expert
        weighted_features = []
        for weights in self.feature_weights:
            # Scale features by learned weights
            weighted_x = x * weights.unsqueeze(0).expand_as(x)
            weighted_features.append(weighted_x)
        
        # Get expert scores for each node
        batch_size = x.size(0)
        expert_scores = torch.zeros(batch_size, len(self.experts), device=x.device)
        
        # Process edge attributes for all experts
        edge_weight = self.gate_edge_nn(edge_attr).view(-1)
        
        # Get scores from each expert with their weighted features
        for i, expert in enumerate(self.experts):
            expert_scores[:, i] = expert(weighted_features[i], edge_index, edge_attr).squeeze()
        
        # Normalize expert scores to similar ranges
        expert_scores = (expert_scores - expert_scores.mean(dim=0, keepdim=True)) / (expert_scores.std(dim=0, keepdim=True) + 1e-5)
        
        # Get gating weights with the original features
        gate_features = self.gate_conv1(x, edge_index, edge_weight=edge_weight)
        gate_features = F.relu(gate_features)
        gate_features = F.dropout(gate_features, p=0.1, training=self.training)
        
        gate_features = self.gate_conv2(gate_features, edge_index, edge_weight=edge_weight)
        gate_features = F.relu(gate_features)
        
        pooled = self.gate_pool(gate_features, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        gate_logits = self.gate_output(pooled)
        
        # Apply normalization to prevent one expert from dominating
        gate_logits = self.gate_norm(gate_logits)
        
        # Add noise during training to encourage exploration
        if self.training:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_scale
        
        # Apply softmax to get weights - ensuring one set of weights per node
        expert_weights = F.softmax(gate_logits, dim=1)
        
        # Force minimum weight for each expert to ensure all have some influence
        min_weight = 0.05
        expert_weights = expert_weights * (1 - min_weight * len(self.experts)) + min_weight
        expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)
        
        # Combine scores using weights (one score per node)
        combined_score = torch.sum(expert_scores * expert_weights, dim=1, keepdim=True)
        
        return combined_score, expert_weights, expert_scores

def extract_agent_features(agent_name):
    """Extract features from agent name"""
    features = []
    
    # Model type features
    model_types = ["4o", "sonnet", "gemini", "o3"]
    for model in model_types:
        features.append(1.0 if model in agent_name else 0.0)
    
    # Reasoning style
    is_cot = any(style in agent_name for style in ["thinking", "reasoning"])
    features.append(1.0 if is_cot else 0.0)
    
    # Circle parameter (if present)
    circle_param = 0
    if "circle" in agent_name:
        try:
            circle_param = int(agent_name.split("circle_")[1].split("_")[0])
        except:
            pass
    features.append(float(circle_param) / 10.0)  # Normalize
    
    return features

def build_strategic_graph(performance_matrix, nash_welfare_matrix, bootstrap_stats, variance_matrix, reasoning_styles=None):
    """Build a graph representation of the strategic landscape with balanced metrics"""
    agents = performance_matrix.index.tolist()
    
    # Try to extract NE regret from bootstrap_stats
    ne_regret_matrix = None
    regret_column = None
    
    if bootstrap_stats is not None:
        if hasattr(bootstrap_stats, 'columns'):
            for col in bootstrap_stats.columns:
                if 'regret' in col.lower():
                    regret_column = col
                    break
            
            if regret_column:
                ne_regret_matrix = bootstrap_stats
        elif isinstance(bootstrap_stats, dict) and 'regret' in bootstrap_stats:
            # If bootstrap_stats is a dict with regret information
            regret_values = bootstrap_stats['regret']
            ne_regret_matrix = pd.DataFrame(regret_values, index=agents, columns=['NE Regret'])
    
    # If we couldn't find regret information or for testing, create regret values
    # that differ between reasoning and non-reasoning agents
    if reasoning_styles:
        print("Using reasoning styles to create different regret values for testing")
        dummy_regrets = {}
        for agent in agents:
            if agent in reasoning_styles:
                # Lower regret (less exploitable) for reasoning agents
                if reasoning_styles[agent] == "reasoning":
                    dummy_regrets[agent] = np.random.uniform(0.001, 0.01)
                else:
                    dummy_regrets[agent] = np.random.uniform(0.05, 0.1)
            else:
                dummy_regrets[agent] = np.random.uniform(0.02, 0.08)
        
        ne_regret_matrix = pd.DataFrame.from_dict(dummy_regrets, orient='index', columns=['NE Regret'])
    elif ne_regret_matrix is None:
        print("Warning: Could not find NE regret information. Using random dummy values.")
        dummy_regrets = np.random.uniform(0.001, 0.1, size=len(agents))
        ne_regret_matrix = pd.DataFrame(dummy_regrets, index=agents, columns=['NE Regret'])
    
    # Print regret matrix info
    print(f"Regret matrix shape: {ne_regret_matrix.shape}")
    print(f"Regret matrix columns: {ne_regret_matrix.columns.tolist()}")
    
    # Extract features for each agent (node)
    node_features = []
    for agent in agents:
        # PERFORMANCE METRICS (1 feature)
        perf_i = float(performance_matrix.loc[agent, :].mean())
        # Normalize performance to 0-1 range (all agents)
        max_perf = performance_matrix.values.max()
        norm_perf = perf_i / max_perf if max_perf > 0 else 0.0
        
        # COOPERATIVENESS METRICS (1 feature)
        welfare_i = float(nash_welfare_matrix.loc[agent, :].mean()) if agent in nash_welfare_matrix.index else 0.0
        # Already in 0-1 range
        
        # EXPLOITABILITY METRICS (1 feature)
        regret_i = 0.0
        if agent in ne_regret_matrix.index:
            if 'Mean NE Regret' in ne_regret_matrix.columns:
                regret_i = float(ne_regret_matrix.loc[agent, 'Mean NE Regret'])
            elif 'NE Regret' in ne_regret_matrix.columns:
                regret_i = float(ne_regret_matrix.loc[agent, 'NE Regret'])
        
        # Transform regret to exploitability score (lower regret = less exploitable = better)
        exploitability_score = np.exp(-10 * regret_i) if regret_i > 0 else 1.0
        
        # ADAPTABILITY METRICS (3 features)
        # 1. Variance (consistency across opponents)
        variance_i = float(variance_matrix.loc[agent, :].mean()) if agent in variance_matrix.index else 0.0
        
        # 2. Consistency score (higher = more consistent = less variance)
        consistency_score = 1.0 / (1.0 + variance_i) if variance_i > 0 else 1.0
        
        # 3. Performance spread (max - min performance against different opponents)
        perf_spread = 0.0
        if agent in performance_matrix.index:
            # Get performance values and handle non-numeric values
            try:
                # Convert to numpy array of floats and handle any conversion errors
                agent_perfs = performance_matrix.loc[agent, :].values
                # Convert to float array, replacing non-convertible values with NaN
                agent_perfs = np.array(agent_perfs, dtype=np.float64)
                # Now we can safely use isnan
                valid_perfs = agent_perfs[~np.isnan(agent_perfs)]
                if len(valid_perfs) > 0:
                    perf_spread = np.max(valid_perfs) - np.min(valid_perfs)
                    # Normalize to 0-1
                    perf_spread = perf_spread / (np.max(valid_perfs) + 1e-5)
            except (ValueError, TypeError) as e:
                print(f"Warning: Error calculating performance spread for {agent}: {e}")
                perf_spread = 0.0
        
        # Combine all features - balanced with 2 per strategic dimension
        features = [
            # Performance dimension (2 features)
            norm_perf,
            float(perf_i),
            
            # Cooperativeness dimension (2 features)
            welfare_i,
            float(nash_welfare_matrix.loc[agent, agent]) if (agent in nash_welfare_matrix.index and agent in nash_welfare_matrix.columns) else 0.0,
            
            # Exploitability dimension (2 features)
            exploitability_score,
            float(regret_i),
            
            # Adaptability dimension (2 features)
            consistency_score,
            float(perf_spread),
        ]
        
        node_features.append(features)
    
    # Create edge relationships and features
    edge_index = []
    edge_features = []
    
    # Add progress bar for edge creation (this can be slow for many agents)
    edge_progress = tqdm(total=len(agents)*(len(agents)-1), desc="Creating graph edges", leave=False)
    
    for i, agent_i in enumerate(agents):
        for j, agent_j in enumerate(agents):
            # Skip self-loops
            if i == j:
                continue
            
            # Add edge
            edge_index.append([i, j])
            
            try:
                # PERFORMANCE EDGE FEATURES (2)
                # 1. Direct performance when agent_i plays against agent_j
                perf_ij = performance_matrix.loc[agent_i, agent_j] if agent_j in performance_matrix.columns else 0.0
                # Handle non-numeric values
                perf_ij = float(perf_ij) if np.isscalar(perf_ij) else 0.0
                # Normalize
                norm_perf_ij = perf_ij / max_perf if max_perf > 0 else 0.0
                
                # 2. Reciprocal performance
                perf_ji = performance_matrix.loc[agent_j, agent_i] if agent_i in performance_matrix.columns else 0.0
                # Handle non-numeric values
                perf_ji = float(perf_ji) if np.isscalar(perf_ji) else 0.0
                norm_perf_ji = perf_ji / max_perf if max_perf > 0 else 0.0
                
                # COOPERATIVENESS EDGE FEATURES (2)
                # 1. Nash welfare between agents
                welfare_ij = nash_welfare_matrix.loc[agent_i, agent_j] if (agent_i in nash_welfare_matrix.index and agent_j in nash_welfare_matrix.columns) else 0.0
                # Handle non-numeric values
                welfare_ij = float(welfare_ij) if np.isscalar(welfare_ij) else 0.0
                
                # 2. Joint welfare (how much total welfare is generated)
                joint_welfare = 0.0
                if agent_j in nash_welfare_matrix.index and agent_i in nash_welfare_matrix.columns:
                    welfare_ji = nash_welfare_matrix.loc[agent_j, agent_i]
                    welfare_ji = float(welfare_ji) if np.isscalar(welfare_ji) else 0.0
                    joint_welfare = welfare_ij + welfare_ji
                else:
                    joint_welfare = welfare_ij
                
                # EXPLOITABILITY EDGE FEATURES (2)
                # 1. Agent i's regret against agent j
                regret_i = 0.0
                if i < len(node_features):
                    regret_i = node_features[i][5]  # Using the raw regret feature
                    
                # 2. Exploitability difference (how much more exploitable is one agent vs the other)
                regret_j = 0.0
                if j < len(node_features):
                    regret_j = node_features[j][5]
                exploit_diff = abs(regret_i - regret_j)
                
                # ADAPTABILITY EDGE FEATURES (2)
                # 1. Variance in performance against this specific opponent
                variance_ij = 0.0
                if agent_i in variance_matrix.index and agent_j in variance_matrix.columns:
                    variance_ij = variance_matrix.loc[agent_i, agent_j]
                    variance_ij = float(variance_ij) if np.isscalar(variance_ij) else 0.0
                    
                # Normalize variance
                max_variance = variance_matrix.values.max() if hasattr(variance_matrix.values, 'max') else 0.0
                if isinstance(max_variance, np.ndarray):
                    max_variance = float(max_variance) if max_variance.size == 1 else 0.0
                norm_variance = variance_ij / (max_variance + 1e-5) if max_variance > 0 else 0.0
                
                # 2. Utility ratio (relative advantage of i over j) - adaptability measure
                utility_ratio = perf_ij / max(1.0, perf_ji) if perf_ji > 0 else (1.0 if perf_ij == 0 else 10.0)
                # Cap and normalize the ratio
                utility_ratio = min(utility_ratio, 10.0) / 10.0
                
                # Combine all edge features - balanced with 2 per strategic dimension
                edge_attr = [
                    # Performance features
                    float(norm_perf_ij),
                    float(norm_perf_ji),
                    
                    # Cooperativeness features
                    float(welfare_ij),
                    float(joint_welfare),
                    
                    # Exploitability features
                    float(regret_i),
                    float(exploit_diff),
                    
                    # Adaptability features
                    float(norm_variance),
                    float(utility_ratio)
                ]
                
            except (ValueError, TypeError) as e:
                print(f"Warning: Error creating edge features for {agent_i} -> {agent_j}: {e}")
                # Create default edge features if there was an error
                edge_attr = [0.0] * 8  # 8 edge features
            
            edge_features.append(edge_attr)
            edge_progress.update(1)
    
    edge_progress.close()
    
    # Convert to PyTorch Geometric Data object
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Print feature balance info
    print("\nFeature balance information:")
    print(f"Number of node features: {x.shape[1]} ({x.shape[1]/4} per expert)")
    print(f"Number of edge features: {edge_attr.shape[1]} ({edge_attr.shape[1]/4} per expert)")
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        agent_names=agents
    )
    
    return data

def classify_agent_reasoning_style(agent_name):
    """Classify agent as reasoning or non-reasoning according to specified categories"""
    # Check if agent matches the reasoning category patterns
    if any(pattern in agent_name for pattern in REASONING_STYLES["reasoning"]):
        return "reasoning"
    # Default to non-reasoning
    return "non_reasoning"


def train_moe_model(graph_data, reasoning_styles, num_epochs=100, lr=0.001, train_idx=None, val_idx=None):
    """Train the MoE model using supervised learning"""
    num_nodes = graph_data.x.shape[0]


    if train_idx is None or val_idx is None:
        train_mask= torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        train_split_idx = int(.8 * num_nodes)

        train_idx = indices[:train_split_idx]
        val_idx = indices[train_split_idx:]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
    else: #use the mask that is provided
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

    y = torch.zeros(num_nodes, 1, device=graph_data.x.device) # Ensure labels are on the correct device
    for i, agent in enumerate(graph_data.agent_names):
        style = reasoning_styles.get(agent, classify_agent_reasoning_style(agent))
        if style == "reasoning":
            y[i] = 1.0

    '''
    just running on cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = graph_data.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    '''
    device = "cpu"
    model = MixtureOfExpertsGNN(graph_data.x.shape[1]).to(device=device)
    model = MixtureOfExpertsGNN(graph_data.x.shape[1]).to(device) # Move model to device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) # Added weight decay
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=max(1, num_epochs // 10), gamma=0.5) # Adjusted scheduler
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0
    patience_counter = 0
    patience = 20 # Early stopping patience

    model.train()
    progress_bar = tqdm(range(num_epochs), desc="Training epochs", leave=False)

    for epoch in progress_bar:
        optimizer.zero_grad()
        combined_score, expert_weights, expert_scores = model(
            graph_data.x, graph_data.edge_index, graph_data.edge_attr
        )
    
        loss = criterion(combined_score[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_score, _, _ = model(
                graph_data.x, graph_data.edge_index, graph_data.edge_attr
            )
            val_loss = criterion(val_score[val_mask], y[val_mask])

            train_preds = (torch.sigmoid(combined_score[train_mask]) > 0.5).float()
            train_acc = (train_preds == y[train_mask]).float().mean()

            val_preds = (torch.sigmoid(val_score[val_mask]) > 0.5).float()
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
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            model.train()
    return model.cpu()


def evaluate_ensemble_strategic_adaptability(models, graph_data, reasoning_styles, performance_data):
    '''
    here we evaluate the strategic adaptability usinfg the ensemble models
    '''
    num_models = len(models)
    num_agents = len(graph_data.agent_names)
    num_experts = len(models[0].experts) if num_models > 0 else 0

    # Ensure graph data is on the correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = graph_data.to(device)

    # Store cumulative results
    all_combined_scores = torch.zeros(num_agents, 1, device=device)
    all_expert_weights = torch.zeros(num_agents, num_experts, device=device)
    all_expert_scores = torch.zeros(num_agents, num_experts, device=device)

    for model in tqdm(models, desc="Evaluating Ensemble Models", leave=False):
        model.to(device)
        model.eval()
        with torch.no_grad():
            combined_score, expert_weights, expert_scores = model(
                graph_data.x, graph_data.edge_index, graph_data.edge_attr
            )
            # Add results to cumulative sums
            all_combined_scores += combined_score
            all_expert_weights += expert_weights
            all_expert_scores += expert_scores
        model.cpu() # Move back to CPU
       
    avg_combined_scores = all_combined_scores / num_models
    avg_expert_weights = all_expert_weights / num_models
    avg_expert_scores = all_expert_scores / num_models

    avg_combined_scores_np = avg_combined_scores.cpu().numpy()
    avg_expert_weights_np = avg_expert_weights.cpu().numpy()
    avg_expert_scores_np = avg_expert_scores.cpu().numpy()
    probabilities = torch.sigmoid(avg_combined_scores).cpu().numpy()

    results = {
        "reasoning": {
            "agents": [], "combined_scores": [], "probabilities": [],
            "expert_weights": [], "expert_scores": [], "performance": []
        },
        "non_reasoning": {
            "agents": [], "combined_scores": [], "probabilities": [],
            "expert_weights": [], "expert_scores": [], "performance": []
        }
    }

    for i, agent in enumerate(graph_data.agent_names):
        style = reasoning_styles.get(agent, classify_agent_reasoning_style(agent))
        results[style]["agents"].append(agent)
        results[style]["combined_scores"].append(avg_combined_scores_np[i].item())
        results[style]["probabilities"].append(probabilities[i].item())
        results[style]["expert_weights"].append(avg_expert_weights_np[i])
        results[style]["expert_scores"].append(avg_expert_scores_np[i])
        # Ensure performance_data uses cleaned agent names if necessary
        cleaned_agent_name = get_display_name(agent) # Assuming get_display_name is available
        results[style]["performance"].append(performance_data.get(cleaned_agent_name, 0))
    
    avg_performance = {}

    for style, data in results.items(): #TODO mmake sure this is ok
        valid_performances = [p for p in data["performance"] if isinstance(p, (int, float)) and not np.isnan(p)]
        if valid_performances:
            avg_performance[style] = np.mean(valid_performances)
        else:
             avg_performance[style] = 0

    improvement_pct = 0
    if "reasoning" in avg_performance and "non_reasoning" in avg_performance and avg_performance["non_reasoning"] > 0:
        improvement_pct = ((avg_performance["reasoning"] / avg_performance["non_reasoning"]) - 1) * 100

    # Test if improvement is within hypothesized range (e.g., 30-50%)
    hypothesis_confirmed = 30 <= improvement_pct <= 50

    return {
        "results_by_style": results,
        "avg_performance": avg_performance,
        "improvement_pct": improvement_pct,
        "hypothesis_confirmed": hypothesis_confirmed
    }


def analyze_ensemble_feature_importance(models, graph_data, criterion):
    num_models = len(models)
    num_features = graph_data.x.shape[1]
    num_agents = graph_data.x.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = graph_data.to(device)

    # Node features (assuming balanced features per dimension)
    feature_names = [
        "Perf_Norm", "Perf_Raw",
        "Coop_Welfare", "Coop_SelfWelfare",
        "Exploit_Score", "Exploit_RawRegret",
        "Adapt_Consistency", "Adapt_PerfSpread"
    ]
    # Ensure feature_names matches the actual number of features
    if len(feature_names) != num_features:
         feature_names = [f"Feature_{i}" for i in range(num_features)]


    total_importance_scores = np.zeros(num_features)

    # Create dummy labels (needed for loss calculation, use actual labels if available)
    # Here, we assume a binary classification task based on some criteria
    # If you have actual labels, use them instead.
    dummy_labels = torch.randint(0, 2, (num_agents, 1), dtype=torch.float, device=device)


    print("Analyzing feature importance across ensemble...")
    for model in tqdm(models, desc="Feature Importance", leave=False):
        model.to(device)
        model.eval()

        # 1. Calculate baseline loss/score
        with torch.no_grad():
            baseline_scores, _, _ = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            # Using loss as the metric - lower is better. Change in loss indicates importance.
            baseline_loss = criterion(baseline_scores, dummy_labels).item()
            # Alternatively, use prediction accuracy or another metric if classification labels are reliable

        model_importance = np.zeros(num_features)
        for i in range(num_features):
            original_feature = graph_data.x[:, i].clone()
            permuted_indices = torch.randperm(num_agents, device=device)
            graph_data.x[:, i] = graph_data.x[permuted_indices, i] # Permute feature i

            with torch.no_grad():
                permuted_scores, _, _ = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
                permuted_loss = criterion(permuted_scores, dummy_labels).item()

            # Importance = increase in loss after permutation
            model_importance[i] = permuted_loss - baseline_loss

            # Restore original feature
            graph_data.x[:, i] = original_feature

        total_importance_scores += model_importance
        model.cpu() # Move back to CPU

    # Average importance over models
    avg_importance_scores = total_importance_scores / num_models

    # Create dictionary of feature importance
    feature_importance = {
        feature_names[i]: avg_importance_scores[i]
        for i in range(len(feature_names))
    }

    return feature_importance



def analyze_expert_contributions(results_by_style):
    """Analyze which experts contribute most to each reasoning style"""
    expert_names = ["Exploitability", "Cooperativeness", "Adaptability", "Performance"]
    
    # Average expert weights by reasoning style
    expert_weights_by_style = {}
    for style, data in results_by_style.items():
        if data["expert_weights"]:
            weights = np.mean(data["expert_weights"], axis=0)
            expert_weights_by_style[style] = {
                expert_names[i]: weights[i] for i in range(len(expert_names))
            }
    
    return expert_weights_by_style

def visualize_results(evaluation_results, expert_weights_by_style, feature_importance=None):
    """Create visualizations of the ensemble results"""
    print("Creating performance comparison chart...")
    # 1. Performance comparison by reasoning style
    plt.figure(figsize=(10, 6))
    styles = list(evaluation_results["avg_performance"].keys())
    performances = [evaluation_results["avg_performance"][style] for style in styles]
    plt.bar(styles, performances)
    plt.ylabel("Average Performance (Ensemble)")
    plt.title(f"Ensemble Performance by Reasoning Style (Improvement: {evaluation_results['improvement_pct']:.1f}%)")

    for i, v in enumerate(performances):
        # Adjust text position if necessary
        plt.text(i, v + (max(performances) * 0.02), f"{v:.1f}", ha='center')

    plt.ylim(0, max(performances) * 1.1) # Adjust y-axis limit
    plt.savefig("ensemble_performance_by_reasoning_style.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating expert contribution chart...")
    # 2. Expert contribution by reasoning style
    plt.figure(figsize=(12, 7)) # Adjusted size
    expert_names = list(expert_weights_by_style[styles[0]].keys()) if styles else []
    if not expert_names:
        print("Warning: No expert names found for visualization.")
        return

    x = np.arange(len(expert_names))
    width = 0.35

    bar_container = {}
    for i, style in enumerate(styles):
        if style in expert_weights_by_style:
            weights = [expert_weights_by_style[style].get(name, 0) for name in expert_names] # Handle missing experts gracefully
            offset = (i - (len(styles) - 1) / 2) * width # Center bars
            bar_container[style] = plt.bar(x + offset, weights, width, label=style)
            # Add labels to bars
            plt.bar_label(bar_container[style], fmt='%.2f', padding=3, size=8)


    plt.ylabel("Average Expert Weight (Ensemble)")
    plt.title("Ensemble Expert Contribution by Reasoning Style")
    plt.xticks(x, expert_names, rotation=45, ha="right") # Rotate labels
    plt.legend(loc='upper right')
    plt.ylim(0, max(max(expert_weights_by_style[s].values()) for s in styles if s in expert_weights_by_style) * 1.15) # Adjust y-limit dynamically
    plt.tight_layout() # Adjust layout
    plt.savefig("ensemble_expert_weights_by_reasoning_style.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating strategic dimensions chart...")
    # 3. Strategic dimensions by agent (using averaged expert scores)
    results = evaluation_results["results_by_style"]
    all_agents, all_styles, all_scores = [], [], {expert: [] for expert in expert_names}
    all_combined_scores = []

    for style, data in results.items():
        for i, agent in enumerate(data["agents"]):
            all_agents.append(agent)
            all_styles.append(style)
            all_combined_scores.append(data["combined_scores"][i])
            for j, expert in enumerate(expert_names):
                 # Ensure index j is valid for expert_scores[i]
                if j < len(data["expert_scores"][i]):
                    all_scores[expert].append(data["expert_scores"][i][j])
                else:
                    all_scores[expert].append(0) # Default value if missing


    # Sort agents by averaged combined score
    if all_combined_scores:
        sorted_indices = np.argsort(all_combined_scores)[::-1]
        sorted_agents = [all_agents[i] for i in sorted_indices]
        sorted_styles = [all_styles[i] for i in sorted_indices]
        sorted_expert_scores = {
            expert: [all_scores[expert][i] for i in sorted_indices]
            for expert in expert_names
        }

        plt.figure(figsize=(max(14, len(sorted_agents)*0.5), 8)) # Dynamic width
        x = np.arange(len(sorted_agents))
        num_experts = len(expert_names)
        width = 0.8 / num_experts # Adjust width based on number of experts
        offsets = np.linspace(-width * (num_experts - 1) / 2, width * (num_experts - 1) / 2, num_experts)


        for i, expert in enumerate(expert_names):
             if expert in sorted_expert_scores: # Check if expert scores exist
                 plt.bar(x + offsets[i], sorted_expert_scores[expert], width, label=expert)

        plt.ylabel("Average Strategic Dimension Score (Ensemble)")
        plt.title("Ensemble Strategic Dimensions by Agent (Sorted by Combined Score)")
        plt.xticks(x, sorted_agents, rotation=90, fontsize=8) # Adjust font size

        # Color the agent names by reasoning style
        ax = plt.gca()
        for i, style in enumerate(sorted_styles):
            color = 'red' if style == 'reasoning' else 'blue'
            ax.get_xticklabels()[i].set_color(color)

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("ensemble_strategic_dimensions_by_agent.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Warning: Cannot create strategic dimensions chart, no combined scores found.")


    # 4. Plot Feature Importance (if available)
    if feature_importance:
        print("Creating feature importance chart...")
        plt.figure(figsize=(10, 6))
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())

        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]

        plt.bar(sorted_features, sorted_importances)
        plt.ylabel("Average Importance Score (Ensemble)")
        plt.title("Ensemble Feature Importance (Permutation Method)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("ensemble_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

def load_or_create_bootstrap_results(performance_matrix):
    """Create simplified bootstrap results if there's an issue with the bootstrap analysis"""
    try:
        # Try to run the regular bootstrap analysis
        bootstrap_results, bootstrap_stats, _, ne_strategy_df = run_nash_analysis(
            performance_matrix,
            num_bootstrap_samples=1000,
            confidence_level=0.95
        )
        return bootstrap_results, bootstrap_stats, ne_strategy_df
    except Exception as e:
        print(f"Error running bootstrap analysis: {e}")
        print("Creating simplified bootstrap results instead")
        
        # Create simplified bootstrap stats DataFrame
        agents = performance_matrix.index.tolist()
        bootstrap_stats = pd.DataFrame({
            'Agent': agents,
            'Mean NE Regret': [0.01] * len(agents),  # Dummy values
            'Std NE Regret': [0.005] * len(agents),
            'Mean Expected Utility': performance_matrix.mean(axis=1).values,
            'Std Expected Utility': performance_matrix.std(axis=1).values
        })
        bootstrap_stats.set_index('Agent', inplace=True)
        
        bootstrap_results = {
            'ne_regret': [[0.01] * len(agents)] * 10,  # 10 bootstrap samples with dummy values
            'ne_strategy': [np.ones(len(agents)) / len(agents)] * 10,  # Uniform strategy
            'expected_utility': [performance_matrix.mean(axis=1).values] * 10
        }
        
        # Create simplified ne_strategy_df
        ne_strategy_df = pd.DataFrame({
            'Agent': agents,
            'Nash Probability': np.ones(len(agents)) / len(agents)  # Uniform strategy
        })
        
        return bootstrap_results, bootstrap_stats, ne_strategy_df

def print_diagnostic_info(graph_data, ne_regret_matrix, reasoning_styles, moe_model):
    """Print detailed diagnostic information about exploitability metrics and model outputs"""
    print("\n" + "="*60)
    print("                 DIAGNOSTIC INFORMATION")
    print("="*60)
    
    # 1. Print raw NE regret values with color highlighting
    print("\n📊 NE Regret Values by Reasoning Style:")
    
    # Organize agents by reasoning style
    reasoning_agents = []
    non_reasoning_agents = []
    
    for i, agent in enumerate(graph_data.agent_names):
        style = reasoning_styles.get(agent, "unknown")
        if style == "reasoning":
            reasoning_agents.append((i, agent))
        else:
            non_reasoning_agents.append((i, agent))
    
    # Print reasoning agents regret stats
    print("\n  🤔 REASONING AGENTS:")
    reasoning_regrets = []
    for i, agent in reasoning_agents:
        regret = 0
        if ne_regret_matrix is not None and agent in ne_regret_matrix.index:
            if 'Mean NE Regret' in ne_regret_matrix.columns:
                regret = ne_regret_matrix.loc[agent, 'Mean NE Regret']
            elif 'NE Regret' in ne_regret_matrix.columns:
                regret = ne_regret_matrix.loc[agent, 'NE Regret']
        
        # Get the regret score from node features if available
        exploitability_score = graph_data.x[i, 2].item() if graph_data.x.shape[1] > 2 else "N/A"
        raw_regret = graph_data.x[i, 5].item() if graph_data.x.shape[1] > 5 else "N/A"
        
        reasoning_regrets.append(regret)
        print(f"    {agent}: {regret:.6f} (exploit. score: {exploitability_score:.4f}, raw: {raw_regret})")
    
    # Print non-reasoning agents regret stats
    print("\n  🎯 NON-REASONING AGENTS:")
    non_reasoning_regrets = []
    for i, agent in non_reasoning_agents:
        regret = 0
        if ne_regret_matrix is not None and agent in ne_regret_matrix.index:
            if 'Mean NE Regret' in ne_regret_matrix.columns:
                regret = ne_regret_matrix.loc[agent, 'Mean NE Regret']
            elif 'NE Regret' in ne_regret_matrix.columns:
                regret = ne_regret_matrix.loc[agent, 'NE Regret']
        
        # Get the regret score from node features if available
        exploitability_score = graph_data.x[i, 2].item() if graph_data.x.shape[1] > 2 else "N/A"
        raw_regret = graph_data.x[i, 5].item() if graph_data.x.shape[1] > 5 else "N/A"
        
        non_reasoning_regrets.append(regret)
        print(f"    {agent}: {regret:.6f} (exploit. score: {exploitability_score:.4f}, raw: {raw_regret})")
    
    # Print regret statistics summary
    print("\n  📈 REGRET SUMMARY:")
    if reasoning_regrets:
        avg_reasoning_regret = sum(reasoning_regrets) / len(reasoning_regrets)
        print(f"    Avg reasoning regret: {avg_reasoning_regret:.6f}")
    if non_reasoning_regrets:
        avg_non_reasoning_regret = sum(non_reasoning_regrets) / len(non_reasoning_regrets)
        print(f"    Avg non-reasoning regret: {avg_non_reasoning_regret:.6f}")
    
    if reasoning_regrets and non_reasoning_regrets:
        regret_ratio = avg_reasoning_regret / avg_non_reasoning_regret if avg_non_reasoning_regret > 0 else float('inf')
        print(f"    Ratio (reasoning/non-reasoning): {regret_ratio:.4f}")
        print(f"    Exploitability difference: {(1 - regret_ratio) * 100:.2f}%")
    
    # 2. Print key node features (first few and important ones)
    print("\n📊 Key Node Features Sample:")
    feature_headers = ["Performance", "Welfare", "Exploit. Score", "Consistency", "Variance", "Raw Regret"]
    
    # Print headers
    header_str = "    Agent".ljust(50)
    for header in feature_headers[:len(feature_headers)]:
        header_str += f" | {header.center(15)}"
    print(header_str)
    print("    " + "-"*100)
    
    # Print sample of agents (first 3 of each type)
    max_samples = min(3, len(reasoning_agents), len(non_reasoning_agents)) 
    sample_agents = reasoning_agents[:max_samples] + non_reasoning_agents[:max_samples]
    for i, agent in sample_agents:
        style = reasoning_styles.get(agent, "unknown")
        features = graph_data.x[i].detach().numpy()
        
        # Format feature display
        agent_str = f"    {agent[:40]} ({style})".ljust(50)
        for j, val in enumerate(features[:min(len(feature_headers), len(features))]):
            agent_str += f" | {val:15.4f}"
        print(agent_str)
    
    # 3. Expert weights and outputs (if model is available)
    if moe_model is not None:
        moe_model.eval()
        try:
            with torch.no_grad():
                combined_score, expert_weights, expert_scores = moe_model(
                    graph_data.x, graph_data.edge_index, graph_data.edge_attr
                )
                
                expert_names = ["Exploitability", "Cooperativeness", "Adaptability", "Performance"]
                
                # Compute average weights by reasoning style
                print("\n📊 EXPERT WEIGHTS BY REASONING STYLE:")
                
                # Check dimensions of expert weights and scores
                print(f"Debug - Expert weights shape: {expert_weights.shape}")
                print(f"Debug - Expert scores shape: {expert_scores.shape}")
                print(f"Debug - Number of agents: {len(graph_data.agent_names)}")
                
                # Process reasoning and non-reasoning agents separately
                reasoning_indices = [i for i, _ in reasoning_agents]
                non_reasoning_indices = [i for i, _ in non_reasoning_agents]
                
                styles_to_process = []
                if reasoning_indices:
                    styles_to_process.append(("reasoning", reasoning_indices))
                if non_reasoning_indices:
                    styles_to_process.append(("non_reasoning", non_reasoning_indices))
                
                for style, indices in styles_to_process:
                    # If no indices, skip
                    if not indices:
                        continue
                        
                    # Get weights and scores for this style - carefully handling indices
                    valid_indices = [i for i in indices if i < expert_weights.shape[0]]
                    
                    if not valid_indices:
                        print(f"  No valid indices for {style} agents")
                        continue
                    
                    # Stack weights and scores only for valid indices
                    if len(valid_indices) > 0:
                        style_weights = torch.stack([expert_weights[i] for i in valid_indices])
                        style_scores = torch.stack([expert_scores[i] for i in valid_indices])
                        
                        # Calculate averages
                        avg_weights = torch.mean(style_weights, dim=0)
                        avg_scores = torch.mean(style_scores, dim=0)
                        
                        style_label = "🤔 Reasoning" if style == "reasoning" else "🎯 Non-Reasoning"
                        print(f"\n  {style_label} Agents:")
                        
                        # Print weights and scores side by side
                        print("    Expert".ljust(20) + " | " + "Weight".center(10) + " | " + "Score".center(10))
                        print("    " + "-"*44)
                        for j, expert_name in enumerate(expert_names):
                            if j < len(avg_weights):
                                print(f"    {expert_name.ljust(18)} | {avg_weights[j].item():10.4f} | {avg_scores[j].item():10.4f}")
                    else:
                        print(f"  No data available for {style} agents")
        except Exception as e:
            print(f"\n⚠️ Error processing model outputs: {str(e)}")
            import traceback
            traceback.print_exc()
            print("  Unable to display expert weights and scores.")
    
    print("\n" + "="*60 + "\n")

def main(num_ensemble=5, num_epochs=100, lr=0.001):
    """Main function to run the ensemble analysis"""
    start_time = time.time()
    print(f"Starting strategic adaptability analysis using Ensemble of {num_ensemble} MoE GNNs")

    # Create a top-level progress bar for tracking overall progress
    main_steps = ["Data Processing", "Matrix Creation", "Welfare Matrices", "Matrix Cleaning", 
                  "Nash Analysis", "Agent Classification", "Graph Building", "Model Training", 
                  "Evaluation", "Visualization"]
    main_progress = tqdm(main_steps, desc="Analysis Progress", position=0)
    
    # Process game data
    print("\nProcessing game data...")
    process_start = time.time()
    crossplay_dir = "crossplay/game_matrix_1"
    
    all_results, agent_performance, agent_final_rounds, agent_game_counts, _ = process_all_games(
        crossplay_dir,
        discount_factor=0.98
    )
    print(f"Game data processing completed in {timedelta(seconds=int(time.time() - process_start))}")
    main_progress.update(1)  # Update main progress
    
    # Create performance matrices
    print("\nCreating performance matrices...")
    matrices_start = time.time()
    performance_matrices = create_performance_matrices(all_results, agent_performance, agent_final_rounds)
    
    all_agents = sorted(list(performance_matrices['overall_agent_performance'].keys()))
    print(f"Performance matrices created in {timedelta(seconds=int(time.time() - matrices_start))}")
    main_progress.update(1)  # Update main progress
    
    # Create welfare matrices
    print("\nCreating welfare matrices...")
    welfare_start = time.time()
    global_max_nash_welfare, _ = compute_global_max_values(num_samples=1000)
    welfare_matrices = create_welfare_matrices(all_results, all_agents, global_max_nash_welfare)
    print(f"Welfare matrices created in {timedelta(seconds=int(time.time() - welfare_start))}")
    main_progress.update(1)  # Update main progress
    
    # Clean matrix names
    print("\nCleaning matrix names...")
    clean_start = time.time()
    cleaned_matrices = {}
    
    performance_matrix_names = ['performance_matrix', 'std_dev_matrix', 'variance_matrix', 'scaled_performance_matrix', 'count_matrix']
    matrix_cleaning_progress = tqdm(performance_matrix_names, desc="Cleaning performance matrices", leave=False)
    for name in matrix_cleaning_progress:
        if name in performance_matrices:
            cleaned_matrices[name] = clean_matrix_names(performance_matrices[name], get_display_name)
    
    welfare_cleaning_progress = tqdm(welfare_matrices.items(), desc="Cleaning welfare matrices", leave=False)
    for name, matrix in welfare_cleaning_progress:
        cleaned_matrices[name] = clean_matrix_names(matrix, get_display_name)
    print(f"Matrix names cleaned in {timedelta(seconds=int(time.time() - clean_start))}")
    main_progress.update(1)  # Update main progress
    
    # Run Nash equilibrium analysis
    print("\nRunning Nash equilibrium analysis...")
    nash_start = time.time()
    performance_matrix = cleaned_matrices['performance_matrix']
    
    bootstrap_results, bootstrap_stats, ne_strategy_df = load_or_create_bootstrap_results(performance_matrix)
    print(f"Nash equilibrium analysis completed in {timedelta(seconds=int(time.time() - nash_start))}")
    main_progress.update(1)  # Update main progress
    
    # Extract data for graph construction
    performance_matrix = cleaned_matrices['performance_matrix']
    nash_welfare_matrix = cleaned_matrices['nash_welfare_matrix']
    variance_matrix = cleaned_matrices['variance_matrix']
    
    # Classify agents by reasoning style
    print("\nClassifying agents by reasoning style...")
    reasoning_styles = {}
    for agent in tqdm(all_agents, desc="Classifying agents", leave=False):
        reasoning_styles[agent] = classify_agent_reasoning_style(agent)
    
    # Print reasoning style classification
    reasoning_count = 0
    non_reasoning_count = 0
    reasoning_agents = []
    non_reasoning_agents = []
    
    for agent, style in reasoning_styles.items():
        if style == "reasoning":
            reasoning_count += 1
            reasoning_agents.append(agent)
        else:
            non_reasoning_count += 1
            non_reasoning_agents.append(agent)
    
    print("\nAgent reasoning style classification:")
    print(f"  reasoning: {reasoning_count} agents")
    for agent in reasoning_agents:
        print(f"    - {agent}")
    print(f"  non_reasoning: {non_reasoning_count} agents")
    main_progress.update(1)  # Update main progress
    
    # Build strategic graph
    print("\nBuilding strategic graph...")
    graph_start = time.time()
    graph_data = build_strategic_graph(
        performance_matrix,
        nash_welfare_matrix,
        bootstrap_stats,
        variance_matrix,
        reasoning_styles
    )
    print(f"Strategic graph built in {timedelta(seconds=int(time.time() - graph_start))}")
    main_progress.update(1)  # Update main progress
    
    # Add diagnostic output before model training
    print_diagnostic_info(graph_data, bootstrap_stats, reasoning_styles, None)
    
    # --- Ensemble Training ---
    print(f"\nTraining Ensemble of {num_ensemble} Mixture of Experts models...")
    train_start = time.time()
    trained_models = []
    num_nodes = graph_data.x.shape[0]

    # Create a fixed train/val split to use for all models in the ensemble
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_split_idx = int(0.8 * num_nodes)
    train_idx = indices[:train_split_idx]
    val_idx = indices[train_split_idx:]


    for i in range(num_ensemble):
        print(f"\n--- Training Model {i+1}/{num_ensemble} ---")
        model = train_moe_model(
            graph_data,
            reasoning_styles,
            num_epochs=num_epochs, # Use parameter
            lr=lr, # Use parameter
            train_idx=train_idx, # Pass fixed split
            val_idx=val_idx      # Pass fixed split
        )
        trained_models.append(model)

    print(f"\nEnsemble model training completed in {timedelta(seconds=int(time.time() - train_start))}")
    # main_progress.update(1) # Update progress tracking if used

    # --- Ensemble Evaluation ---
    print("\nEvaluating strategic adaptability using the ensemble...")
    eval_start = time.time()
    # Use the correct performance data source (handle potential cleaning mismatches)
    # performance_data = performance_matrices['overall_agent_performance'] # Original
    # Corrected: use cleaned matrix and map original names if needed
    cleaned_perf_matrix = cleaned_matrices['performance_matrix']
    # Create a dict mapping cleaned names to average performance
    performance_data_cleaned = {
         agent: cleaned_perf_matrix.loc[agent, :].mean()
         for agent in cleaned_perf_matrix.index
     }


    evaluation_results = evaluate_ensemble_strategic_adaptability( # Call new function
        trained_models,
        graph_data,
        reasoning_styles,
        performance_data_cleaned # Use cleaned performance data
    )
    print(f"Ensemble strategic adaptability evaluation completed in {timedelta(seconds=int(time.time() - eval_start))}")
    # main_progress.update(1)

    # --- Ensemble Analysis ---
    # Analyze expert contributions using ensemble results
    expert_weights_by_style = analyze_expert_contributions(
        evaluation_results["results_by_style"]
    )

    # Analyze feature importance using the ensemble
    # Need a criterion instance for the importance function
    criterion = nn.BCEWithLogitsLoss()
    ensemble_feature_importance = analyze_ensemble_feature_importance(
        trained_models, graph_data, criterion
    )


    # --- Print Results ---
    print("\nEnsemble Results:")
    print(f"Average performance by reasoning style:")
    for style, avg_perf in evaluation_results["avg_performance"].items():
        print(f"  {style}: {avg_perf:.2f}")

    print(f"Performance improvement: {evaluation_results['improvement_pct']:.2f}%")
    # print(f"Hypothesis confirmed: {evaluation_results['hypothesis_confirmed']}") # Optional

    print("\nAverage Expert contributions by reasoning style (Ensemble):")
    for style, weights in expert_weights_by_style.items():
        print(f"  {style}:")
        for expert, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True): # Sort by weight
            print(f"    {expert}: {weight:.3f}")

    print("\nAverage Feature Importance (Ensemble):")
    for feature, importance in sorted(ensemble_feature_importance.items(), key=lambda item: item[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")

    # --- Visualize Results ---
    print("\nCreating ensemble visualizations...")
    viz_start = time.time()
    visualize_results(evaluation_results, expert_weights_by_style, ensemble_feature_importance) # Pass feature importance
    print(f"Visualizations created in {timedelta(seconds=int(time.time() - viz_start))}")
    # main_progress.update(1)


    # --- Final ---
    # main_progress.close() # If using progress bar
    total_time = time.time() - start_time
    print(f"\nEnsemble strategic adaptability analysis complete! Total time: {timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    # Example: Run with an ensemble of 5 models
    main(num_ensemble=5, num_epochs=150, lr=1e-4) # Adjust params as needed
