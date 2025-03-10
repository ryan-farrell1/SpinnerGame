import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Union, List
import random
import copy

# Piecewise Linear Function for smoothing experiment data   
def piecewise_linear(data, t):
    num_steps = len(data)
    x = divmod(t, num_steps)[1]
    int_x, frac_x = divmod(x, 1)
    int_x = int(int_x)
    pt0 = data[int_x]
    pt1 = data[divmod(int_x + 1, num_steps)[1]]
    func_val = (1 - frac_x)*pt0 + frac_x*pt1
    return func_val

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
    def __init__(self, num_trials : int, board_length_rand : bool, board_val_rand : bool, target_rand : bool, update_func):
        # Tester Settings
        self.num_trials = num_trials
        self.board_length_rand = board_length_rand
        self.board_val_rand = board_val_rand
        self.target_rand = target_rand
        self.board_update = update_func
        self.unfinished_games = []
        self.finished_games = []

    def register(self, board=[-1,0,1], target=10, board_range=[-10,10], board_len=3, board_len_range=[2,5], target_range=[0,100]):
        
        # Save important settings
        def rand_range_str(low, high):
            return f"Rand({low}, {high})"

        if self.board_length_rand:
            len_str = rand_range_str(board_len_range[0], board_len_range[1])
        else:
            len_str = str(board_len)

        if self.board_val_rand:
            val_str = rand_range_str(board_range[0], board_range[1])
        else:
            val_str = str(board)

        if self.target_rand:
            tar_str = rand_range_str(target_range[0], target_range[1])
        else:
            tar_str = str(target)
        
        self.game_stats = {"Board Length": len_str, "Board Value": val_str, "Target": tar_str}

        # Creating Games
        for i in range(self.num_trials):
            # Creating Board
            if self.board_length_rand:
                board_len = np.random.randint(low=board_len_range[0], high=board_len_range[1]+1)
            if self.board_val_rand:
                board = np.random.randint(low=board_range[0], high=board_range[1]+1, size=board_len).tolist()

            # Creating Target
            if self.target_rand:
                target = np.random.randint(low=target_range[0], high=target_range[1])

            new_game = SpinnerGame(board, 0, target, self.board_update)

            # Adding Game to Records
            if target == 0:
                self.finished_games.append(new_game)
            else:
                self.unfinished_games.append(new_game)

            self.all_games_over = len(self.unfinished_games) == 0

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
    def test_histories(self, seperate_histories=False, centralized=False):
        finished_histories = []
        unfinished_histories = []

        for x in self.finished_games:
            if centralized:
                finished_histories.append([t - x.target for t in x.score_history])
            else:
                finished_histories.append(x.score_history)

        for x in self.unfinished_games:
            if centralized:
                unfinished_histories.append([t - x.target for t in x.score_history])
            else:
                unfinished_histories.append(x.score_history)
        
        if not seperate_histories:
            return finished_histories + unfinished_histories
        else:
            return {"Finished": finished_histories, "Unfinished": unfinished_histories}
        
    # Create Plot for Game Scores
    def score_plot(self, target=0, include_target=False, centralized=False, partitions_per_turn=20):

        if target:
            include_target = True
        if centralized:
            include_target = False

        # Formatting Data
        game_scores = self.test_histories(centralized=centralized)
        starts = [game[0] for game in game_scores]
        avg_start = np.mean(starts).item()
        avg_start = round(avg_start)
        game_lengths = [len(x) for x in game_scores]
        longest_game = max(game_lengths)
        max_scores = [max(x) for x in game_scores]
        high_score = max(max_scores)
        min_scores = [min(x) for x in game_scores]
        low_score = min(min_scores)
        full_history = [x + [target] * (longest_game - len(x)) for x in game_scores]
        x_tick_spacing = max(int(longest_game / 15),1)
        y_tick_spacing = max(int((high_score - low_score) / 10),1)

        # Smoothing Data
        num_partitions = int(longest_game * partitions_per_turn)
        a = 0
        b = longest_game - 1
        t = np.linspace(a, b, num_partitions)
        if include_target:
            target_goal = np.full(num_partitions, target)
        smoothed_history = []
        for x in full_history:
            curr_hist = []
            for i in t:
                curr_hist.append(piecewise_linear(x, i))
            smoothed_history.append(curr_hist)

        # Coverting to Pandas df
        np_hist = pd.DataFrame(smoothed_history).T
        np_hist['avg_score'] = np_hist.mean(axis=1)
        t_col = pd.DataFrame(t)
        t_col.columns = ['frame_num']
        np_hist = pd.concat([np_hist, t_col], axis=1)

        # Creating Figure
        fig = go.Figure()
        for x in np_hist.columns:
            if x == 'frame_num' or x == 'avg_score':
                continue
            fig.add_trace(go.Scatter(x=np_hist['frame_num'], y=np_hist[x], mode='lines', line=dict(color="rgba(150,150,150,0.2)")))
        fig.add_trace(go.Scatter(x=np_hist['frame_num'], y=np_hist['avg_score'], mode='lines', line=dict(color='rgba(255,0,0,1)')))
        if include_target:
            fig.add_trace(go.Scatter(x=np_hist['frame_num'], y=target_goal, mode='lines', line=dict(color='rgba(255, 155, 0, 1)', dash='dash')))
        if centralized:
            y_title = "Displacement from Target"
        else:
            y_title = "Game Score"
        fig.update_layout(title='Spinner Game Simulation',
                        xaxis_title='Round Number',
                        yaxis_title=y_title,
                        showlegend=False,
                        xaxis=dict(tick0=0, dtick=x_tick_spacing),
                        yaxis=dict(tick0=avg_start, dtick=y_tick_spacing),
                        height=500
                        )

        return fig
    
    # Create Game Length Plot
    def game_length_plot(self):

        # Formatting Data
        game_scores = self.test_histories(seperate_histories=True)
        unfin_game_lengths = [len(x)-1 for x in game_scores['Unfinished']]
        fin_game_lengths = [len(x)-1 for x in game_scores['Finished']]
        game_lengths = unfin_game_lengths + fin_game_lengths
        longest_game = max(game_lengths)
        shortest_game = min(game_lengths)
        plot_buffer = max(int(0.2 * (longest_game - shortest_game)), 2)
        game_lengths = np.array(game_lengths)
        avg_length = np.mean(game_lengths).item()
        avg_length = round(avg_length, 2)

        # Creating Game Stats
        if len(game_lengths) > 1:
            var_length = np.var(game_lengths).item()
            var_length = round(var_length, 2)
        else:
            var_length = 0

        self.game_stats.update({'Mean': str(avg_length), "Variance": str(var_length)})
        
        # Constructing DataFrame
        df = []
        for x in fin_game_lengths:
            df.append([x, "Finished"])
        for x in unfin_game_lengths:
            df.append([x, "Unfinished"])
        df = pd.DataFrame(df, columns=["Game Length", "Game Status"])

        # Creating plot
        fig = px.histogram(df, x='Game Length', color='Game Status', color_discrete_map={'Finished': '#95d2ec', 'Unfinished': '#ff4242'}, range_x=[shortest_game - plot_buffer, longest_game + plot_buffer])
        fig.update_layout(title='Game Length Histogram',
                          xaxis_title='Game Length',
                          yaxis_title='Frequency',
                          height=500
                        )
        fig.update_traces(xbins_size=1)
        
        # Return Figure and Game Statistics
        return fig, self.game_stats
        


