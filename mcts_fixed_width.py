from environment import *
from mcts import *

class FixedWidthMCTSPlayer(HistoryMCTSPlayer):
    def __init__(self, num_simulations: int, exploration_constant: float, fixed_width: int):
        super().__init__(num_simulations, exploration_constant)
        self.fixed_width = fixed_width
        self.history_to_actions = {}  # Maps history to sampled legal actions

    def get_sampled_actions(self, history: KuhnPokerHistory) -> list[int]:
        """
        Returns a consistent subset of legal actions for a given history.
        If the history is encountered for the first time, sample and store the actions.
        """
        if history not in self.history_to_actions:
            legal_actions = history.get_legal_actions()
            if len(legal_actions) > self.fixed_width:
                self.history_to_actions[history] = random.sample(legal_actions, self.fixed_width)
            else:
                self.history_to_actions[history] = legal_actions
        return self.history_to_actions[history]

    def explore(self, history: KuhnPokerHistory) -> int:
        """
        Selects an action to explore using UCB with a fixed-width constraint for the current history and player.
        """
        current_player = history.get_current_player()
        total_visits = sum(self.visit_counts.get((history, a), 0) for a in history.get_legal_actions())
        legal_actions = self.get_sampled_actions(history)

        best_action = None
        best_value = float('-inf')
        for action in legal_actions:
            q_value = self.action_value_estimates.get((history, action), defaultdict(float))[current_player]
            if self.visit_counts.get((history, action), 0) == 0:
                return action
            ucb_value = q_value + self.exploration_constant * math.sqrt(
                math.log(total_visits + 1) / (self.visit_counts[(history, action)] + 1)
            )
            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action
        assert best_action is not None, "No best action found"
        return best_action

if __name__ == '__main__':
    # Test the FixedWidthMCTSPlayer
    fixed_width_player = FixedWidthMCTSPlayer(num_simulations=100, exploration_constant=1.0, fixed_width=2)
    simulator = Simulator([fixed_width_player, RandomPlayer()])
    results = simulator.simulate_episodes(10)
    print(results)
