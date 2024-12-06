# Kuhn Poker MCTS Implementation

This repository contains various Monte Carlo Tree Search (MCTS) implementations for the Kuhn Poker environment. The implementations include different variants of MCTS algorithms optimized for the game's decision-making process.

## Implementations

### Available MCTS Variants
1. Classic MCTS
   - Standard implementation of Monte Carlo Tree Search
   - Uses UCT (Upper Confidence Bound for Trees) for selection

2. Fixed-Width MCTS
   - Limits the number of children nodes explored at each state
   - Helps manage computational resources in complex game states

3. Progressive Widening MCTS
   - Gradually increases the number of explored actions based on visit count
   - Formula: k * N^α, where N is the visit count and α is the widening factor

4. Human-Crafted MCTS
   - Incorporates domain knowledge and expert-designed heuristics
   - Uses hand-crafted rules to guide the selection process

5. Smart Widening MCTS
   - Adaptive action selection based on state characteristics
   - Combines statistical learning with progressive widening

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kuhn-poker-mcts.git
cd kuhn-poker-mcts

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Example use of the code is in the testing.ipynb

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{kuhn-poker-mcts,
  author = {Your Name},
  title = {Kuhn Poker MCTS Implementation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/kuhn-poker-mcts}
}
```
