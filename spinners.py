import numpy as np
import copy

class SpinnerGame():
    # Game Startup
    def __init__(self, board : list[int], initial : int, target : int, update_func, score_history=None):
        # Initial Data
        self.start_board = board
        self.initial = initial

        # Game Data
        self.board = copy.deepcopy(board)
        self.score = initial
        self.target = target
        self.update = update_func
        self.rng = np.random.default_rng()
        self.game_over = initial == target

        if score_history:
            self.score_history = score_history
        else:
            self.score_history = [initial]

    # Spin & Update Score
    def spin(self):
        spun_index = self.rng.choice(len(self.board))
        self.score += self.board[spun_index]
        self.score_history.append(self.score)
        return spun_index
    
    # Update Board 
    def update_board(self, spun_index):
        delta = self.update(self.board, self.score, self.target, spun_index)
        self.board[spun_index] += delta

    # Combines Spin & Update
    def forward(self):

        # Testing if Game Over
        if self.game_over:
            return True

        # Spin & Test Again / Update
        spun_index = self.spin()
        if self.score == self.target:
            self.game_over = True
        else:
            self.update_board(spun_index)

        return self.game_over
    
    # Reset Game to Starting Position
    def reset(self):
        self.board = self.start_board
        self.score = self.initial
        self.score_history = [self.initial]
        self.game_over = self.score == self.target

    # Makes Copy of Game
    def copy(self, keep_score_history=True):

        # Writing Score History
        if keep_score_history:
            passed_history = copy.deepcopy(self.score_history)
        else:
            passed_history = None

        # Create/Return Copy of Game
        new_game = SpinnerGame(copy.deepcopy(self.board), self.score, self.target, self.update, passed_history)
        return new_game


class MassTester():
    # Create Game Tester
    def __init__(self, game : SpinnerGame, num_trials : int):
        # Tester Settings
        self.num_trials = num_trials
        self.unfinished_games = []
        self.finished_games = []

        # Creating Games
        temp = []
        for i in range(num_trials):
            temp.append(game.copy())

        if game.game_over == True:
            self.finished_games = temp
            self.all_games_over = True
        else:
            self.unfinished_games = temp
            self.all_games_over = False

    # One Forward Pass on all Unfinished Games
    def forward(self):
        # Looping through Unfinished Games
        for i, x in enumerate(self.unfinished_games):
            x_game_over = x.forward()
            # Moving to Finished Games if Necessary
            if x_game_over:
                finished_game = self.unfinished_games.pop(i)
                self.finished_games.append(finished_game)
        
        # Checking if all games are finished
        if not self.unfinished_games:
            self.all_games_over = True

        return self.all_games_over
    
    # Full Pass through Experiment (All Games Finished or Max Game Length Reached)
    def full_pass(self, max_pass_length=None):
        # Pass Variables
        pass_flag = True
        if max_pass_length:
            pass_counter = 0

        # Pass Loop
        while pass_flag and not self.all_games_over:
            self.forward()
            if max_pass_length:
                pass_counter += 1
                pass_flag = pass_counter < max_pass_length

    # Return list of (finished & unfinished) game histories
    def test_histories(self, seperate_histories=False):
        finished_histories = []
        unfinished_histories = []

        for x in self.finished_games:
            finished_histories.append(x.score_history)

        for x in self.unfinished_games:
            unfinished_histories.append(x.score_history)
        
        if not seperate_histories:
            return finished_histories + unfinished_histories
        else:
            return {"Finished": finished_histories, "Unfinished": unfinished_histories}
