from environment import *

class HistoryMCTSPlayer(Player):
    def __init__(self, num_simulations: int, exploration_constant: float):
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.visit_counts = {}  # Maps (history, action) -> visit count
        self.action_value_estimates = {}  # Maps (history, action) -> Q value estimates for each player
        self.beliefs = {card: 1/len(KuhnPokerState.DECK) for card in KuhnPokerState.DECK}  # Maps history -> belief distribution over opponent cards
        self.history_to_visits = {} # Maps history to visit count

    def estimate_values(self, state: KuhnPokerState,) -> dict[int, float]:
        '''
        Estimates the values of a state using random rollouts.
        '''
        print(f'Estimating values for state:\n {state}')
        state = state.copy()
        while not state.is_terminal():
            legal_actions = state.get_legal_actions()
            print(f'The legal actions are {legal_actions} at state:\n {state}')
            action = random.choice(legal_actions)
            print(f'Player {state.current_player()} chooses action {action} in state {state}')
            state.apply_action(action, state.current_player())
            print(f'The state after the action is:\n {state}')
        returns = state.get_returns()
        return returns
    
    def explore(self, history: KuhnPokerHistory) -> int:
        '''
        Selects an action to explore using UCB for the current history and current player.
        '''
        current_player = history.get_current_player()
        # calculate total visits to this history
        total_visits = self.history_to_visits[history]
        best_action = None
        best_value = float('-inf')
        for action in history.get_legal_actions():
            q_value = self.action_value_estimates.get((history, action), defaultdict(float))[current_player]
            # if action has not been visited, return it
            if self.visit_counts.get((history, action), 0) == 0:
                return action
            # calculate UCB value
            ucb_value = q_value + self.exploration_constant * math.sqrt(math.log(total_visits) / self.visit_counts[(history, action)])
            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action
        assert best_action is not None, "No best action found"
        return best_action
        

    def simulate(self, history: KuhnPokerHistory, state: KuhnPokerState,) -> dict[int, float]:
        '''
        Runs a Monte Carlo Tree Search simulation from the given state.

        Returns the simulated values of the state (one for each player).
        '''
        # if state is terminal, return the returns
        if state.is_terminal():
            return state.get_returns()
        
        # if history has not been visited at all, return the estimated values
        if history not in self.history_to_visits.keys():
            self.history_to_visits[history] = 0
            # set q values and visit counts to 0 for each action
            for action in history.get_legal_actions():
                self.action_value_estimates[(history, action)] = {0: 0, 1: 0}
                self.visit_counts[(history, action)] = 0
            
            return self.estimate_values(state)
        
        # if state is not terminal and history has been visited, select action to explore
        action = self.explore(history)

        # apply the selected action to the state
        next_state = state.copy()
        next_state.apply_action(action, state.current_player())
        # simulate from the new state and new history
        new_history = history.switch_perspective(state.current_player(), state.players_hands[state.current_player()])
        new_history.observations.append(next_state.get_observation(next_state.current_player()))
        new_q_values = self.simulate(new_history, next_state)
        # update the q values and visit counts
        self.visit_counts[(history, action)] += 1
        self.history_to_visits[history] += 1
        for player, value in new_q_values.items():
            self.action_value_estimates[(history, action)][player] += (value - self.action_value_estimates[(history, action)][player]) / (self.visit_counts[(history, action)])
        return new_q_values

    def select_random_state(self, history: KuhnPokerHistory, beliefs: Dict[int, float]) -> KuhnPokerState:
        '''
        Chooses a random opponent card according to the beliefs and reconstructs the state from the history.
        '''
        opponent_card = random.choices(list(beliefs.keys()), weights=beliefs.values())[0] # type: ignore
        last_observation = history.get_last_observation()
        return KuhnPokerState.init_from_observation(last_observation, opponent_card)

    def choose_action(self, history: KuhnPokerHistory, player_id: int) -> int:
        '''
        Runs MCTS and chooses the best action based on action value estimates.
        '''
        for _ in range(self.num_simulations):
            # select random state according to beliefs
            state = self.select_random_state(history, self.beliefs)
            # simulate from the selected state
            returns = self.simulate(history, state)
        
        # get best action for the current history by returning action with highest q value
        best_action = None
        best_value = float('-inf')
        for action in history.get_legal_actions():
            q_value = self.action_value_estimates.get((history, action), defaultdict(float))[player_id]
            if q_value > best_value:
                best_value = q_value
                best_action = action
        assert best_action is not None, "No best action found"
        return best_action

if __name__ == '__main__':
    # Now we can simulate the game with the MCTS player
    mcts_player = HistoryMCTSPlayer(num_simulations=100, exploration_constant=1.0)
    simulator = Simulator([mcts_player, RandomPlayer()])
    results = simulator.simulate_episodes(10)
    print(results)