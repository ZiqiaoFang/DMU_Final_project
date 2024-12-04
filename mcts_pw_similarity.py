from environment import *
from mcts import *

class PWSimilarityMCTSPlayer(HistoryMCTSPlayer):
    def __init__(self, num_simulations: int, exploration_constant: float, theta_1: float, theta_2: float):
        super().__init__(num_simulations, exploration_constant)
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.history_to_actions = {}  # Maps history to actions already added

    def get_progressively_widened_actions(self, history: KuhnPokerHistory) -> list[int]:
        """
        Returns the subset of legal actions based on progressive widening for the given history.
        """
        if history not in self.history_to_actions:
            # add highest and lowest actions from the legal actions
            self.history_to_actions[history] = [min(history.get_legal_actions()), max(history.get_legal_actions())]

        total_visits = self.history_to_visits[history]
        max_actions = int(self.theta_1 * (total_visits ** self.theta_2))
        
        if len(self.history_to_actions[history]) + 1 <= max_actions:
            # time to add a new action
            legal_actions = history.get_legal_actions()
            new_action = self.add_new_action(self.history_to_actions[history], history, legal_actions)
            self.history_to_actions[history].append(new_action)

        return self.history_to_actions[history]
    
    def add_new_action(self, already_added_actions: list[int], history: KuhnPokerHistory, legal_actions: list[int]) -> int:
        '''
        First sort the legal actions.
        First find index of action in already_added_actions with the highest Q-value. 
        The find two closest actions to this action in the already_added_actions.
        Then find the action in legal_actions that is closest to the average of the two actions in already_added_actions.
        '''
        legal_actions.sort()
        q_values = [self.action_value_estimates.get((history, action), defaultdict(float))[history.get_current_player()] for action in already_added_actions]
        max_q_value_index = q_values.index(max(q_values))
        if max_q_value_index == 0:
            closest_actions = already_added_actions[:2]
        elif max_q_value_index == len(already_added_actions) - 1:
            closest_actions = already_added_actions[-2:]
        else:
            closest_actions = already_added_actions[max_q_value_index - 1: max_q_value_index + 2]
        average = sum(closest_actions) / 2
        best_action = None
        best_distance = float('inf')
        for action in legal_actions:
            distance = abs(action - average)
            if distance < best_distance:
                best_distance = distance
                best_action = action
        assert best_action is not None, "No best action found"
        return best_action


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
    progressive_widening_player = PWSimilarityMCTSPlayer(
        num_simulations=100,
        exploration_constant=1.0,
        theta_1=1.5,  # Adjust as needed
        theta_2=0.5   # Adjust as needed
    )
    simulator = Simulator([progressive_widening_player, RandomPlayer()])
    results = simulator.simulate_episodes(10)
    print(results)
