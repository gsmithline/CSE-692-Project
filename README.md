# Graph Learning for LLM Multi-agent Bargaining

This repository contains the code for the paper "Graph Learning for LLM Multi-agent Bargaining". It implements a framework for analyzing the strategic behavior of Large Language Model (LLM) agents in bargaining games using Graph Neural Networks (GNNs).

## Overview

The project investigates LLM agent behavior in a controlled negotiation game. It evaluates agents based on fairness (EF1, Pareto efficiency), welfare (utilitarian, normalized Nash), and realized utility. Agent interactions are modeled as a best-response graph, where nodes represent agent configurations and edges capture empirical performance differences.

Graph Neural Networks are trained on this graph for two main tasks:
1.  **Link Regression:** Predicting the outcome (utility difference) of bargaining games between pairs of agents.
2.  **Node Classification (Exploratory):** Distinguishing between "reasoning" and "non-reasoning" agents based on behavioral features.

The framework utilizes both handcrafted statistical features (e.g., exploitability, fairness consistency, utility variance) and high-dimensional embeddings derived from LLM-generated summaries of agent behavior, fusing them via a learned projection layer for improved performance.

## Key Features

*   **Bargaining Game Environment:** Implementation of the mixed-motive bargaining game described in the paper.
*   **Agent Implementations:** Various LLM-based agent configurations.
*   **Evaluation Metrics:** Code for calculating fairness (EF1, Pareto), welfare (Nash, Utilitarian), and utility-based performance metrics.
*   **Best Response Graph Construction:** Scripts to generate the interaction graph from gameplay data.
*   **Feature Engineering:** Extraction of handcrafted features and LLM summary embeddings.
*   **GNN Models:** Implementations of GNNs for link regression and node classification tasks, including the feature fusion mechanism.
*   **Analysis & Visualization:** Notebooks and scripts for running experiments, analyzing results, and generating figures (like those in the paper).

## File Structure

```
.
├── agents/                 # Agent implementations
├── best_response_graph_graphviz_/ # Visualization outputs for BR graphs
├── bootstrap_analysis/     # Code related to bootstrap analysis
├── crossplay/              # Cross-play interaction data/scripts
├── eval/                   # Evaluation scripts
├── experiment_scripts/     # Scripts to run experiments
├── experiments/            # Experiment configurations or results
├── meta_game_analysis/     # Analysis related to the meta-game
├── metrics/                # Implementations of evaluation metrics
├── nash_equilibrium/       # Code related to Nash equilibrium computation
├── prompts/                # Prompts used for LLM agents
├── strategic_adaptability/ # Core strategic adaptability analysis code
├── test_matrices_performance_matrices/ # Saved performance matrices
├── test_results/           # Test results storage
├── utils/                  # Utility functions
├── .git/
├── .vscode/
├── __pycache__/
├── analyze_single_game.py  # Script for analyzing a single game instance
├── agent.py                # Base agent class or definition
├── best_response_graph     # Data for the best response graph
├── best_response_graph.png # Visualization of the best response graph
├── bootstrap_paper.pdf     # Related paper PDF
├── bootstrap_paper.txt     # Related paper text
├── egta_process.py         # Empirical Game Theoretic Analysis process script
├── eps_nash_finder.py      # Epsilon-Nash equilibrium finder
├── eq_chek.ipynb           # Notebook for equilibrium checking
├── game.py                 # Core bargaining game logic
├── game_runner.py          # Script to run bargaining games between agents
├── LLM_summary_vector_dict.pkl # Saved LLM summary embeddings
├── README.md               # This file
├── requirements_strategic_analysis.txt # Python dependencies
├── results.ipynb           # Jupyter notebook for results analysis/visualization
├── simple_gnn_ensemble*.png # Output figures from GNN analysis
├── solutions.py            # Solutions or equilibrium computation code
├── strategic_adaptability_gnn.py # GNN model for strategic adaptability
├── strategic_adaptability_moe.py # Mixture-of-Experts models experiments & training
├── strategic_adaptability_README.md # Specific README for strategic adaptability
├── test_game_eval.ipynb    # Notebook for testing game evaluation
├── test_game_eval.py       # Script for testing game evaluation
├── visualize_rd_trace.py   # Script to visualize Replicator Dynamics traces (?)
└── ... (other config/data files)
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements_strategic_analysis.txt
    ```
    *(Note: You might need compatible versions of PyTorch depending on your CUDA setup if using GPUs)*

## Usage

The core logic for running experiments and analyses can likely be found in:

*   `game_runner.py`: To simulate games between specified agents.
*   `strategic_adaptability_gnn.py` / `strategic_adaptability_moe.py`: To train and evaluate the GNN models. This contains the ensemble and MOE methods.
*   `results.ipynb`: For reproducing analysis and figures from the paper.
*   `experiment_scripts/`: Contains specific scripts used for the paper's experiments.

Please refer to the specific scripts and notebooks for detailed usage instructions and command-line arguments.

## Citation

If you use this code or framework in your research, please cite the original paper:

```bibtex
@inproceedings{ma2024graph,
  title={Graph Learning for LLM Multi-agent Bargaining},
  author={Ma, Jiaji and Smithline, Gabe},
  booktitle={CSE 692 Class Project},
  year={2024}
}
```
