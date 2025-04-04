# Strategic Adaptability Analysis with Mixture of Experts GNN

This project implements a Graph Neural Network (GNN) based Mixture of Experts (MoE) approach to analyze the strategic adaptability of different LLM agents in negotiation scenarios.

## Hypothesis

**LLM strategic adaptability is primarily determined by reasoning style rather than model size or architecture, with chain-of-thought reasoners demonstrating 30-50% higher performance in mixed-motive scenarios compared to direct responders of equivalent size. This relationship will be most pronounced in games requiring transitions between cooperative and competitive phases.**

## Project Structure

- `strategic_adaptability_moe.py`: Main implementation of the GNN-based Mixture of Experts
- `requirements_strategic_analysis.txt`: Required dependencies

## Setup

Install the required dependencies:

```bash
pip install -r requirements_strategic_analysis.txt
```

## Usage

Run the analysis:

```bash
python strategic_adaptability_moe.py
```

This will:
1. Process the crossplay data
2. Create performance and welfare matrices
3. Run Nash equilibrium analysis
4. Classify agents by reasoning style
5. Build a strategic graph
6. Train the Mixture of Experts GNN model
7. Evaluate strategic adaptability
8. Generate visualizations to test the hypothesis

## GNN Expert Architecture

The Mixture of Experts consists of four specialized GNN experts:

1. **Exploitability Expert**: Measures resistance to exploitation (Nash regret)
2. **Cooperativeness Expert**: Measures cooperative tendencies (Nash welfare)
3. **Adaptability Expert**: Measures strategic adaptability (consistency across opponents)
4. **Performance Expert**: Measures overall performance

A gating network determines which expert to trust for different agent types and scenarios.

## Results

The analysis generates several visualizations:

1. **Performance by Reasoning Style**: Compares performance between chain-of-thought and direct response agents
2. **Expert Contribution by Reasoning Style**: Shows which strategic dimensions are most important for each reasoning style
3. **Strategic Dimensions by Agent**: Detailed breakdown of strategic capabilities by agent

## Interpretation

The results test whether chain-of-thought reasoners have a 30-50% performance advantage over direct responders, confirming or rejecting our hypothesis about the importance of reasoning style in strategic adaptability. 