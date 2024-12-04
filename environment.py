import enum
import random
import copy
import math
import numpy as np
from typing import Hashable, List, Dict, Optional, Tuple, Callable
from collections import defaultdict
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC
from dataclasses import dataclass



    
class State(ABC):
    @abstractmethod
    def apply_action(self, action: int, player: int):
        pass

    @abstractmethod
    def get_observation(self, player: int) -> Hashable:
        pass

    @abstractmethod
    def current_player(self) -> int:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def legal_actions(self) -> List[int]:
        pass

    @abstractmethod
    def get_returns(self) -> dict[int, float]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

class History(ABC):

    @abstractmethod
    def get_legal_actions(self) -> List[int]:
        pass

class ActionType(enum.IntEnum):
    FOLD = -1
    CHECK = 0
    # BET actions will be numbers from 1 to 100


class PlayerType(enum.IntEnum):
    CHANCE = -2  # For card dealing
    TERMINAL = -1  # Game is over
    # Regular players are 0, 1, 2, ...


@dataclass
class KuhnPokerObservation:
    player_hand: int
    player_index: int
    bets: List[int]
    current_player: int
    folded: List[bool]
    bet_amount: Optional[int]
    winner: Optional[int] = None

    def get_legal_actions(self) -> List[int]:
        if self.is_terminal():
            return []
        elif self.bet_amount is None:
            return [ActionType.CHECK] + list(range(1, KuhnPokerState.MAX_BET + 1))
        else:
            return [ActionType.FOLD, self.bet_amount]
        
    def is_terminal(self) -> bool:
        return self.current_player == PlayerType.TERMINAL
    
    def __hash__(self) -> int:
        return hash((self.player_hand, self.player_index, tuple(self.bets), self.current_player, tuple(self.folded), self.bet_amount, self.winner))
    
    def copy(self) -> 'KuhnPokerObservation':
        return copy.deepcopy(self)

@dataclass
class KuhnPokerHistory:
    observations: List[KuhnPokerObservation]

    def get_legal_actions(self) -> List[int]:
        return self.observations[-1].get_legal_actions()
    
    def copy(self) -> 'KuhnPokerHistory':
        return copy.deepcopy(self)
    
    def __hash__(self) -> int:
        return hash(tuple(self.observations))
    
    def get_current_player(self) -> int:
        return self.observations[-1].current_player
    
    def get_last_observation(self) -> KuhnPokerObservation:
        return self.observations[-1]
    
    def switch_perspective(self, player_id: int, player_card: int) -> 'KuhnPokerHistory':
        '''
        Switches the player_card and player_id in all observations in the history. Returns new history (deep copy).
        '''
        new_observations = []
        for observation in self.observations:
            new_observation = observation.copy()
            new_observation.player_index = player_id
            new_observation.player_hand = player_card
            new_observations.append(new_observation)
        return KuhnPokerHistory(observations=new_observations)


class KuhnPokerState:
    DECK = tuple([0, 1, 2])  # Cards are dealt from this deck
    MAX_BET = 100  # Maximum bet amount

    def __init__(self):
        self.players_hands = random.sample(KuhnPokerState.DECK, 2)  # Deal two cards to two players
        self.bets = [1, 1]  # Individual player bets. Ante is 1
        self.folded = [False, False]  # Track if players have folded
        self.current_player_index = random.randint(0, 1)  # Random starting player
        self.bet_amount = None  # Amount of the bet. None if no bet has been made
        self.winner = None  # Winner of the game

    @staticmethod
    def init_from_observation(observation: KuhnPokerObservation, opponent_card: int) -> 'KuhnPokerState':
        state = KuhnPokerState()
        state.players_hands = [0, 0]
        state.players_hands[observation.player_index] = observation.player_hand
        state.players_hands[1 - observation.player_index] = opponent_card
        state.bets = observation.bets
        state.folded = observation.folded
        state.current_player_index = observation.current_player
        state.bet_amount = observation.bet_amount
        state.winner = observation.winner
        return state

    def apply_action(self, action: int, player: int):
        if self.current_player_index != player:
            raise ValueError("Not this player's turn!")

        if action not in self.get_legal_actions():
            raise ValueError("Illegal action!")

        if action == ActionType.FOLD:
            self.folded[player] = True
            self.current_player_index = PlayerType.TERMINAL
            self.winner = 1 - player  # The other player wins
        elif action == ActionType.CHECK:
            if self.bet_amount is None:  # First check
                self.bet_amount = 0
                self.current_player_index = 1 - player
            else:  # Both players checked
                self.current_player_index = PlayerType.TERMINAL
                self.determine_winner()
        else:  # BET or CALL
            if self.bet_amount is None:  # First bet
                self.bet_amount = action
                self.bets[player] += action
                self.current_player_index = 1 - player
            else:  # CALL
                self.bets[player] += self.bet_amount
                self.current_player_index = PlayerType.TERMINAL
                self.determine_winner()

    def determine_winner(self):
        if self.folded[0]:
            self.winner = 1
        elif self.folded[1]:
            self.winner = 0
        else:  # Compare hands
            if self.players_hands[0] > self.players_hands[1]:
                self.winner = 0
            else:
                self.winner = 1

    def get_observation(self, player: int) -> KuhnPokerObservation:
        return KuhnPokerObservation(
            player_hand=self.players_hands[player],
            player_index=player,
            bets=self.bets,
            current_player=self.current_player_index,
            folded=self.folded,
            bet_amount=self.bet_amount,
            winner=self.winner
        )

    def current_player(self) -> int:
        return self.current_player_index

    def is_terminal(self) -> bool:
        return self.current_player_index == PlayerType.TERMINAL

    def get_legal_actions(self) -> List[int]:
        if self.is_terminal():
            return []
        elif self.bet_amount is None:
            return [ActionType.CHECK] + list(range(1, KuhnPokerState.MAX_BET + 1))
        else:
            return [ActionType.FOLD, self.bet_amount]

    def get_returns(self) -> dict[int, float]:
        if not self.is_terminal():
            return {0: 0.0, 1: 0.0}  # No payoff if the game is not over

        if self.winner == 0:
            out = {0: float(self.bets[1]), 1: float(self.bets[1])}
            assert out[0] <= KuhnPokerState.MAX_BET + 1, f"Player 0 bet {self.bets[1]}"
            return out
        else:
            out = {0: -float(self.bets[0]), 1: float(self.bets[0])}
            assert out[1] <= KuhnPokerState.MAX_BET + 1, f"Player 1 bet {self.bets[0]}"
            return out

    def __str__(self) -> str:
        return (f"Hands: {self.players_hands}, Bets: {self.bets}, Folded: {self.folded}, "
                f"Current Player: {self.current_player_index}, Bet Amount: {self.bet_amount}, "
                f"Winner: {self.winner}")

    def __hash__(self) -> int:
        return hash((tuple(self.players_hands), tuple(self.bets), tuple(self.folded), self.current_player_index, self.bet_amount))

    def copy(self) -> 'KuhnPokerState':
        return copy.deepcopy(self)
    
    def get_pot(self) -> int:
        return sum(self.bets)

    

class Player:

    @abstractmethod
    def choose_action(self, history: KuhnPokerHistory, player_id: int) -> int:
        pass

    @abstractmethod
    def get_policy(self) -> Dict:
        pass

class RandomPlayer(Player):
    
    def choose_action(self, history: KuhnPokerHistory, player_id: int) -> int:
        return random.choice(history.get_legal_actions())

    def get_policy(self) -> Dict:
        return {}
    
class HumanPlayer(Player):

    def choose_action(self, history: KuhnPokerHistory, player_id: int) -> int:
        print(f"Legal actions:\n {history.get_legal_actions()} \n\n Observation: \n {history.get_last_observation()}")
        human_input = input(f"Player {player_id}, choose an action:\n")
        return int(human_input)
    
    def get_policy(self) -> Dict:
        return {}

@dataclass
class SimulatorResults:
    player_0_wins: int
    player_1_wins: int
    draws: int
    average_pot: float
    player_0_episodes_by_card: Dict[int, int]
    player_1_episodes_by_card: Dict[int, int]
    player_0_conditional_winrate_by_card: Dict[int, float]
    player_1_conditional_winrate_by_card: Dict[int, float]
    player_0_average_profit: float
    player_1_average_profit: float
    player_0_average_profit_by_card: Dict[int, float]
    player_1_average_profit_by_card: Dict[int, float]
    total_episodes: int


class Simulator:
    def __init__(self, players: List[Player]):
        self.players = players

    def simulate_episodes(self, num_episodes: int) -> SimulatorResults:
        player_0_wins = 0
        player_1_wins = 0
        draws = 0
        total_pot = 0
        player_0_episodes_by_card = defaultdict(int)
        player_1_episodes_by_card = defaultdict(int)
        player_0_wins_by_card = defaultdict(int)
        player_1_wins_by_card = defaultdict(int)
        player_0_total_profit = 0
        player_1_total_profit = 0
        player_0_total_profit_by_card = defaultdict(float)
        player_1_total_profit_by_card = defaultdict(float)

        for _ in range(num_episodes):
            # Initialize the state and histories
            state = KuhnPokerState()
            player_histories = [
                KuhnPokerHistory(observations=[state.get_observation(player_id)])
                for player_id in range(len(self.players))
            ]

            # Track the current game
            while not state.is_terminal():
                current_player = state.current_player()
                current_history = player_histories[current_player]

                print('state', state)

                # Current player chooses an action
                action = self.players[current_player].choose_action(current_history, current_player)
                assert action in state.get_legal_actions()
                print(f'Player {current_player} chooses action {action} in state {state}')


                # Apply the action to the state
                state.apply_action(action, current_player)

                # Update all players' histories
                for player_id in range(len(self.players)):
                    observation = state.get_observation(player_id)
                    player_histories[player_id].observations.append(observation)

            # Calculate returns and update metrics
            returns = state.get_returns()
            pot = state.get_pot()
            total_pot += pot

            # Update per-player metrics
            player_0_card = state.players_hands[0]
            player_1_card = state.players_hands[1]
            player_0_episodes_by_card[player_0_card] += 1
            player_1_episodes_by_card[player_1_card] += 1
            player_0_total_profit += returns[0]
            player_1_total_profit += returns[1]
            player_0_total_profit_by_card[player_0_card] += returns[0]
            player_1_total_profit_by_card[player_1_card] += returns[1]

            if returns[0] > 0:
                player_0_wins += 1
                player_0_wins_by_card[player_0_card] += 1
            elif returns[1] > 0:
                player_1_wins += 1
                player_1_wins_by_card[player_1_card] += 1
            else:
                draws += 1

        # Calculate conditional win rates and average profits
        player_0_conditional_winrate_by_card = {
            card: player_0_wins_by_card[card] / player_0_episodes_by_card[card]
            if player_0_episodes_by_card[card] > 0 else 0.0
            for card in KuhnPokerState.DECK
        }
        player_1_conditional_winrate_by_card = {
            card: player_1_wins_by_card[card] / player_1_episodes_by_card[card]
            if player_1_episodes_by_card[card] > 0 else 0.0
            for card in KuhnPokerState.DECK
        }
        player_0_average_profit_by_card = {
            card: player_0_total_profit_by_card[card] / player_0_episodes_by_card[card]
            if player_0_episodes_by_card[card] > 0 else 0.0
            for card in KuhnPokerState.DECK
        }
        player_1_average_profit_by_card = {
            card: player_1_total_profit_by_card[card] / player_1_episodes_by_card[card]
            if player_1_episodes_by_card[card] > 0 else 0.0
            for card in KuhnPokerState.DECK
        }

        return SimulatorResults(
            player_0_wins=player_0_wins,
            player_1_wins=player_1_wins,
            draws=draws,
            average_pot=total_pot / num_episodes,
            player_0_episodes_by_card=player_0_episodes_by_card,
            player_1_episodes_by_card=player_1_episodes_by_card,
            player_0_conditional_winrate_by_card=player_0_conditional_winrate_by_card,
            player_1_conditional_winrate_by_card=player_1_conditional_winrate_by_card,
            player_0_average_profit=player_0_total_profit / num_episodes,
            player_1_average_profit=player_1_total_profit / num_episodes,
            player_0_average_profit_by_card=player_0_average_profit_by_card,
            player_1_average_profit_by_card=player_1_average_profit_by_card,
            total_episodes=num_episodes
        )

if __name__ == '__main__':
    simulator = Simulator([RandomPlayer(), RandomPlayer()]) 
    results = simulator.simulate_episodes(10)
    print(results)

    # simulate random player vs human player
    simulator = Simulator([RandomPlayer(), HumanPlayer()])
    results = simulator.simulate_episodes(1)
    print(results)