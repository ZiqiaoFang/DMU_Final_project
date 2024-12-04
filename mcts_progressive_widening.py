from environment import *
from mcts import *

class ProgressiveWideningMCTSPlayer(HistoryMCTSPlayer):
    def __init__(self, num_simulations: int, exploration_constant: float, theta_1: float, theta_2: float):
        super().__init__(num_simulations, exploration_constant)
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.history_to_actions = {}  # Maps history to shuffled legal actions

    def get_progressively_widened_actions(self, history: KuhnPokerHistory) -> list[int]:
        """
        Returns the subset of legal actions based on progressive widening for the given history.
        """
        if history not in self.history_to_actions:
            self.history_to_actions[history] = random.sample(history.get_legal_actions(), len(history.get_legal_actions()))

        total_visits = self.history_to_visits[history]
        max_actions = int(self.theta_1 * (total_visits ** self.theta_2))
        legal_actions = self.history_to_actions[history][:max(1, min(max_actions, len(self.history_to_actions[history])))]
        return legal_actions

    def explore(self, history: KuhnPokerHistory) -> int:
        """
        Selects an action to explore using UCB with progressive widening.
        """
        current_player = history.get_current_player()
        total_visits = self.history_to_visits[history]
        legal_actions = self.get_progressively_widened_actions(history)

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
    # Test the ProgressiveWideningMCTSPlayer
    progressive_widening_player = ProgressiveWideningMCTSPlayer(
        num_simulations=100,
        exploration_constant=1.0,
        theta_1=1.5,  # Adjust as needed
        theta_2=0.5   # Adjust as needed
    )
    simulator = Simulator([progressive_widening_player, RandomPlayer()])
    results = simulator.simulate_episodes(10)
    print(results)
