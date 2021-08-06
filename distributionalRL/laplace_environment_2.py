import matplotlib.pyplot as plt
import numpy as np
import random

class gridworld2_final:
    
    def __init__(self):
        self.dim = [1, 4]
        self.goal_one = [0, 0]
        self.goal_two = [0,1]
        #self.big_mag = 2
        self.goal_three = [0, 2]
        self.goal_four = [0,3]
        #self.small_mag = 1
        # Define starting position
        self.start = [0, 0]
        self.time = 0
        self.s = self.start[:]
        self.complete = False
        self.mag_prob = [0.5, 0.5]
        # Step count
        #self.n = 0
        self.action_space = [0, 1]
        self.action_dict = {'Left': 0,
                           'Right': 1}
        self.action_prob = [0.5, 0.5]
        self.action_list = []
        self.goal_one_val = 2
        self.goal_two_val = 2
        self.goal_three_val = 1
        self.goal_four_val = 1
        self.reward = 0
    
    # Show empty selfironment
    def show_grid(self):
        # print rows
        for i in range(self.dim[0]):
            #print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.goal_one[0] and j == self.goal_two[1]:
                    row.append("| +/- 2 ")
                elif i == self.goal_two[0] and j == self.goal_two[1]:
                    row.append("| +/- 2")
                elif i == self.goal_three[0] and j == self.goal_three[1]:
                    row.append("| +/- 1")
                elif i == self.goal_four[0] and j == self.goal_four[1]:
                    row.append("| +/- 1")
                elif i == self.start[0] and j == self.start[1]:
                    row.append("| A ")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        #print("-" * (self.dim[1] * 5 + 1))
    
    def show_mags(self):
        print(self.goal_one_val, self.goal_two_val, self.goal_three_val, self.goal_four_val)
        
    # Give the agent an action
    def step(self, a):
        self.reward = 0
        if a not in self.action_space:
            return "Error: Invalid action submission"
        # Check for special terminal states
        if self.s == self.goal_one:
            self.reward = self.goal_one_val
            self.s[1] += 1
        elif self.s == self.goal_two:
            self.reward = self.goal_two_val
            self.s[1] += 1
        elif self.s == self.goal_three:
            self.reward = self.goal_three_val
            self.s[1] += 1
        elif self.s == self.goal_four:
            self.reward = self.goal_four_val
            self.complete = True

        #if self.s == self.goal_two:
            #self.reward = self.goal_two_val
            #self.cum_reward += self.goal_two_val
        #elif self.s == self.goal_three:
            #self.reward = self.goal_three_val
            #self.cum_reward += self.goal_three_val
        #elif self.s == self.goal_four:
            #self.reward = self.goal_four_val
            #self.complete = True
            #self.reward = self.goal_four_val
            #self.cum_reward += self.goal_four_val

        return self.s, self.reward, self.complete
    
            
    def reset(self):
        self.s = self.start[:]
        self.complete = False
        self.n = 0
        self.reward = 0
        self.cum_reward = 0
        self.action_list = []
        self.goal_one_val = np.random.choice([-2,2])
        self.goal_two_val = np.random.choice([-2,2])
        self.goal_three_val = np.random.choice([-1,1])
        self.goal_four_val = np.random.choice([-1,1])
        if self.s == self.goal_one:
            #self.complete = True
            self.reward += self.goal_one_val
        return self.s