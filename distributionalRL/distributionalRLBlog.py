import os
import sys
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from laplace_environment_1 import gridworld1_final
from laplace_environment_2 import gridworld2_final
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tabulate import tabulate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import *
import scipy as sp
import plotly.express as px
from scipy.signal import savgol_filter

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(page_title = "Distributional RL Blog")
    local_css("style.css")
    st.markdown("<img src = 'https://fb-logo-images.s3-us-west-2.amazonaws.com/fatbrain-logo-color-h.png' style='width:210px;height:50px;'>",unsafe_allow_html=True)

    st.title("Distributional Reinforcement Learning")
    st.markdown("Version 1.0 | Author: Josh |  Build Date: 08/01/20")

    st.sidebar.header('Configuration')
    environment = st.sidebar.number_input("Which environment?", min_value=1, max_value=2)

    if st.button('Run'):
      if environment == 1:
        env = gridworld1_final()
      else:
        env = gridworld2_final()
      env.reset()
      env.show_grid()
      sigmoid = lambda x,c,a: 1/(1 + np.exp(-a*(x-c))) #reward sensitivities

      dim=env.dim[1]
      num_states=dim-1 #number of timesteps in the MP
      alpha=0.01 #learning rate 
      num_gammas=100 #number of gammas

      # Use gammas equally separated in the linear space:
      #range_g=np.linspace(0,1,num_gammas)

      # Use gammas equally separated in the 1/log(gamma) space:
      taus=np.linspace(0.01,num_states,num_gammas)
      range_g=np.exp(-1/taus)

      num_trials=1000  #number of trials used to learn the TD code
      width=25 #steepness of the reward sensitivities
      num_h=150 #number of h's
      range_h=np.linspace(-0.25,1.25,num_h) #theta_h range

      V_gamma=np.zeros((len(range_g),dim,num_h)) #initialize V_{h,\gamma}
      i_g=0 #counter for the gammas
      for gamma in tqdm(range_g[0:-1]):
          i_g=i_g+1
          V=np.zeros((dim,num_h))
          for t in range(1,num_trials):
              state = env.reset()
              done = False
              rewards = 0
              while done == False:
                  old_pos = state[1]
                  new_state, reward, done = env.step(1)
                  new_pos = new_state[1]
                  R=sigmoid(reward,range_h, width) #Apply reward sensitivity to the reward r
                  if not done:
                      delta= R + gamma*V[new_pos,:] - V[old_pos,:]
                  else:
                      delta = R - V[old_pos,:]
                  V[old_pos,:]=V[old_pos,:]+alpha*delta
          V_gamma[i_g,:,:]=V
      if environment == 1:
        V_gamma2=V_gamma[:,1,:]
      else:
        V_gamma2=V_gamma[4:-1,0,:]
      Z=V_gamma2[:,0:-2]-V_gamma2[:,1:-1]
      if environment == 1:
        delta_h_fig = plt.figure(dpi=150)
        delta_h_ax = delta_h_fig.gca(projection='3d')
        X,Y=np.meshgrid(range_h[0:-2],range_g);

        # for h in range(0,num_h-2):
        #     Z[:,h]=savgol_filter(Z[:,h], 11, 1)  

        delta_h_surf = delta_h_ax.plot_surface(X, Y, Z, cmap='summer'
                              , edgecolor='none',vmin=0,vmax=0.07)

        delta_h_ax.view_init(50, -45)
        #delta_h_ax.set_zlabel('Convergence value')
        #delta_h_ax.set_ylabel('Temporal Discount')
        #delta_h_ax.set_xlabel('Reward')
      else:
        X,Y=np.meshgrid(range_h[0:-2],range_g[4:-1]);

        for h in range(0,num_h-2):
            Z[:,h]=savgol_filter(Z[:,h], 15, 1)  

                
        delta_h_fig = plt.figure(dpi=150)
        delta_h_ax = delta_h_fig.gca(projection='3d')
        delta_h_surf = delta_h_ax.plot_surface(X, Y, Z, cmap='summer'
                              , edgecolor='none',alpha=1)
        delta_h_ax.view_init(60, -45)
        # ax.set_zlim(0,0.15)
        delta_h_ax.grid(b=None)
        plt.yticks([0,0.5,1])
        plt.xticks([-2,-1,0,1,2])
      delta_h_ax.set_zlabel('Convergence value')
      delta_h_ax.set_ylabel('Temporal Discount')
      delta_h_ax.set_xlabel('Reward')
      delta_h_ax.set_title('Convergence Value for Given Reward Sensitivity and Future Discount Factor')
      st.write(delta_h_fig)

      