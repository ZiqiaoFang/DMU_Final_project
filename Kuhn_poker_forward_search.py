class ForwardSearchPlayer(Player):
    def __init__(self, max_depth=3, node_budget=1000, discount_factor=0.95):
        self.max_depth = max_depth
        self.node_budget = node_budget
        self.nodes_visited = 0
        self.discount_factor = discount_factor
        # Initialize belief as uniform distribution over opponent cards
        self.belief_state = {card: 1/3 for card in KuhnPokerState.DECK}
        
    def reset_belief(self):
        """Reset belief to uniform distribution"""
        self.belief_state = {card: 1/3 for card in KuhnPokerState.DECK}
        
    def update_belief(self, history: KuhnPokerHistory, player_id: int):
        """Update beliefs based on observed actions"""
        last_obs = history.get_last_observation()
        
        if len(history.observations) < 2:
            return  # No actions to update on yet
            
        prev_obs = history.observations[-2]
        if prev_obs.current_player == 1 - player_id:  # If opponent just acted
            action = last_obs.bets[-1] - prev_obs.bets[-1]  # Get last action's bet amount
            
            # Update probabilities based on opponent's action
            new_belief = {}
            total_weight = 0
            
            for card in KuhnPokerState.DECK:
                if card != last_obs.player_hand:  # Skip our own card
                    # Calculate likelihood of action given card
                    if action > 0:  # Bet/Call
                        likelihood = 0.7 if card > last_obs.player_hand else 0.3
                    else:  # Check/Fold
                        likelihood = 0.3 if card > last_obs.player_hand else 0.7
                        
                    new_belief[card] = self.belief_state[card] * likelihood
                    total_weight += new_belief[card]
            
            # Normalize beliefs
            if total_weight > 0:
                for card in new_belief:
                    new_belief[card] /= total_weight
                self.belief_state = new_belief

    def value_function(self, observation: KuhnPokerObservation) -> float:
        """Base value function for leaf nodes"""
        if observation.is_terminal():
            # Use the actual game payoff for terminal states
            if observation.winner == observation.player_index:
                return float(observation.bets[1 - observation.player_index])
            else:
                return -float(observation.bets[observation.player_index])
            
        # Simple heuristic based on card strength and pot size
        pot_size = sum(observation.bets)
        relative_card_strength = observation.player_hand / len(KuhnPokerState.DECK)
        return 0.1 * relative_card_strength * pot_size

    def forward_search(self, history: KuhnPokerHistory, depth: int, player_id: int):
        """Recursive forward search with belief updates"""
        self.nodes_visited += 1
        current_obs = history.get_last_observation()
        
        if self.nodes_visited >= self.node_budget or depth <= 0 or current_obs.is_terminal():
            return {'action': None, 'value': self.value_function(current_obs)}
        
        legal_actions = history.get_legal_actions()
        if not legal_actions:
            return {'action': None, 'value': self.value_function(current_obs)}
        
        best_action = None
        best_value = float('-inf')
        
        for action in legal_actions:
            value = self.q_value(history, action, depth, player_id)
            if value > best_value:
                best_value = value
                best_action = action
                
        return {'action': best_action, 'value': best_value}
    
    def q_value(self, history: KuhnPokerHistory, action: int, depth: int, player_id: int) -> float:
        """Calculate Q-value for a history-action pair"""
        current_obs = history.get_last_observation()
        expected_value = 0.0
        
        # For each possible opponent card
        for opp_card, prob in self.belief_state.items():
            if prob > 0 and opp_card != current_obs.player_hand:
                # Create a hypothetical state
                state = KuhnPokerState.init_from_observation(current_obs, opp_card)
                
                # Simulate action
                if action in state.get_legal_actions():
                    state.apply_action(action, player_id)
                    
                    # Create new history with the new observation
                    new_history = history.copy()
                    new_history.observations.append(state.get_observation(player_id))
                    
                    if state.is_terminal():
                        returns = state.get_returns()
                        expected_value += prob * returns[player_id]
                    else:
                        # Recursive search
                        result = self.forward_search(new_history, depth-1, player_id)
                        expected_value += prob * result['value']
        
        return expected_value
    
    def choose_action(self, history: KuhnPokerHistory, player_id: int) -> int:
        """Main method to select an action"""
        self.nodes_visited = 0  # Reset node count
        self.update_belief(history, player_id)
        
        result = self.forward_search(history, self.max_depth, player_id)
        return result['action']
    
    def get_policy(self) -> Dict:
        """Return the current policy (empty as policy is computed online)"""
        return {}