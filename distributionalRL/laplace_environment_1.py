import matplotlib.pyplot as plt
import numpy as np

class gridworld1_final:
    
    def __init__(self):
        self.dim = [1, 7]
        self.big_goal = [0, 6]
        self.big_val = 1
        self.small_goal = [0, 0]
        self.small_val = 0.6
        # Define starting position
        self.start = [0, 1]
        self.time = 0
        self.s = self.start[:]
        self.complete = False
            
        # Step count
        self.n = 0
        self.action_space = [0, 1]
        self.action_dict = {'Left': 0,
                           'Right': 1}
        self.action_prob = [0.5, 0.5]
        self.action_list = []
        self.reward = 0
    # Show empty selfironment
    def show_grid(self):
        # print rows
        for i in range(self.dim[0]):
            print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.big_goal[0] and j == self.big_goal[1]:
                    row.append("| 1 ")
                elif i == self.small_goal[0] and j == self.small_goal[1]:
                    row.append("| 0.6 ")
                elif i == self.start[0] and j == self.start[1]:
                    row.append("| A ")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        print("-" * (self.dim[1] * 5 + 1))
    
    # Show state
    def show_state(self):
        # print rows
        for i in range(self.dim[0]):
            print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.s[0] and j == self.s[1]:
                    row.append("| A ")
                elif i == self.big_goal[0] and j == self.big_goal[1]:
                    row.append("| 1 ")
                elif i == self.small_goal[0] and j == self.small_goal[1]:
                    row.append("| 0.5 ")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        print("-" * (self.dim[1] * 5 + 1))
    
    # Give the agent an action
    def step(self, a):
        self.reward = 0
        if a not in self.action_space:
            return "Error: Invalid action submission"
        # Check for special terminal states
        if self.s == self.big_goal:
            self.complete = True
            self.reward = self.big_val
        elif self.s == self.small_goal:
            self.complete = True
            self.reward = self.small_val
        else:
            # Move left once
            if a == 0:
                self.s[1] -= 1
                self.action_list.append(0)
            else:
                self.s[1] += 1
                self.action_list.append(1)
        self.n+=1
        return self.s, self.reward, self.complete
            
    def reset(self):
        self.s = self.start[:]
        self.complete = False
        self.n = 0
        self.action_list = []
        return self.s

class gridworld1:
    
    def __init__(self):
        self.dim = [1, 7]
        self.big_goal = [0, 6]
        self.big_val = 1
        self.small_goal = [0, 0]
        self.small_val = 0.6
        # Define starting position
        self.start = [0, 1]
        self.time = 0
        self.s = self.start[:]
        self.complete = False
            
        # Step count
        self.n = 0
        self.action_space = [0, 1]
        self.action_dict = {'Left': 0,
                           'Right': 1}
        self.action_prob = [0.5, 0.5]
        self.action_list = []
        self.reward = 0
    # Show empty selfironment
    def show_grid(self):
        # print rows
        for i in range(self.dim[0]):
            print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.big_goal[0] and j == self.big_goal[1]:
                    row.append("| 1 ")
                elif i == self.small_goal[0] and j == self.small_goal[1]:
                    row.append("| 0.6 ")
                elif i == self.start[0] and j == self.start[1]:
                    row.append("| A ")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        print("-" * (self.dim[1] * 5 + 1))
    
    # Show state
    def show_state(self):
        # print rows
        for i in range(self.dim[0]):
            print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.s[0] and j == self.s[1]:
                    row.append("| A ")
                elif i == self.big_goal[0] and j == self.big_goal[1]:
                    row.append("| 1 ")
                elif i == self.small_goal[0] and j == self.small_goal[1]:
                    row.append("| 0.6 ")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        print("-" * (self.dim[1] * 5 + 1))
    
    # Give the agent an action
    def step(self, a):
        self.reward = 0
        if a not in self.action_space:
            return "Error: Invalid action submission"
        if a == 0:
            self.s[1] -= 1
            self.action_list.append(0)
            if self.s == self.small_goal:
                self.complete = True
                self.reward = self.small_val
        else:
            self.s[1] += 1
            self.action_list.append(1)
            if self.s == self.big_goal:
                self.complete = True
                self.reward = self.big_val
        self.n+=1
        return self.s, self.reward, self.complete
            
    def reset(self):
        self.s = self.start[:]
        self.complete = False
        self.n = 0
        self.action_list = []
        return self.s

class gridworld2:
    
    def __init__(self):
        self.dim = [1, 7]
        self.big_goal = [0, 6]
        self.big_val = 1
        self.small_goal = [0, 0]
        self.small_val = 0.6
        # Define starting position
        self.start = [0, 1]
        self.time = 0
        self.s = self.start[:]
        self.complete = False
            
        # Step count
        self.n = 0
        self.action_space = [0, 1]
        self.action_dict = {'Left': 0,
                           'Right': 1}
        self.action_prob = [0.5, 0.5]
        self.action_list = []
        self.reward = 0
        self.branch = 1
    # Show empty selfironment
    def show_grid(self):
        # print rows
        for i in range(self.dim[0]):
            print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.big_goal[0] and j == self.big_goal[1]:
                    row.append("| 1 ")
                elif i == self.small_goal[0] and j == self.small_goal[1]:
                    row.append("| 0.6 ")
                elif i == self.start[0] and j == self.start[1]:
                    row.append("| A ")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        print("-" * (self.dim[1] * 5 + 1))
    
    # Show state
    def show_state(self):
        # print rows
        for i in range(self.dim[0]):
            print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.s[0] and j == self.s[1]:
                    row.append("| A ")
                elif i == self.big_goal[0] and j == self.big_goal[1]:
                    row.append("| 1 ")
                elif i == self.small_goal[0] and j == self.small_goal[1]:
                    row.append("| 0.5 ")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        print("-" * (self.dim[1] * 5 + 1))
    
    # Give the agent an action
    def step(self, a):
        self.reward = 0
        if a not in self.action_space:
            return "Error: Invalid action submission"
        # Check for special terminal states
        if self.s == self.big_goal:
            self.complete = True
            self.reward = self.big_val
        elif self.s == self.small_goal:
            self.complete = True
            self.reward = self.small_val
        else:
            # Move left once
            if a == 0:
                self.s[1] -= 1
                self.action_list.append(0)
            else:
                self.s[1] += 1
                self.action_list.append(1)
        self.n+=1
        return self.s, self.reward, self.complete
            
    def reset(self):
        self.s = self.start[:]
        self.branch = 1+np.random.binomial(1, 0.5)
        self.complete = False
        self.n = 0
        self.action_list = []
        return self.s