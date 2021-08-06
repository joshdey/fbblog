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
        env.reset()
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
        for gamma in tqdm(range_g[0:-2]):
            i_g=i_g+1
            V=np.zeros((dim,num_h))
            for t in range(1,num_trials):
                state = env.reset() #reset environment
                done = False #for each trial through environment, set done to false and end when true
                rewards = 0 #set rewards per trial to 0
                while done == False:
                    old_pos = state[1]
                    if len(env.action_list) == 0:
                        action = np.random.choice(env.action_space)
                    elif len(env.action_list) > 0 and env.action_list[-1] == 1:
                        action = env.action_space[1]
                    else:
                        action = env.action_space[0]
                    new_state, reward, done = env.step(action)
                    new_pos = new_state[1]
                    R=sigmoid(reward,range_h, width) #Apply reward sensitivity to the reward r
                    if not done:
                        delta = R + gamma*V[new_pos,:] - V[old_pos,:]
                    else:
                        delta = R - V[old_pos,:]
                    V[old_pos,:]=V[old_pos,:]+alpha*delta
            V_gamma[i_g,:,:]=V
        V_gamma2=V_gamma[:,1,:]
        Z=V_gamma2[:,0:-2]-V_gamma2[:,1:-1]
        #plt.clf()
        fig1 = plt.figure(dpi=150)
        ax1 = fig1.gca(projection='3d')
        X,Y=np.meshgrid(range_h[0:-2],range_g);

        # for h in range(0,num_h-2):
        #     Z[:,h]=savgol_filter(Z[:,h], 11, 1)  

        surf = ax1.plot_surface(X, Y, Z, cmap='summer'
                              , edgecolor='none',vmin=0,vmax=0.07)

        ax1.view_init(50, -45)
        ax1.set_zlabel('Convergence value')
        ax1.set_ylabel('Temporal Discount')
        ax1.set_xlabel('Reward')
        st.write(fig1)
        alpha_reg=0.6 #regularization parameter


        K=num_states-1 #Temporal horizon of tau-space
        delta_t=1/4 #Temporal resolution. Delta t=0.25 is chosen here. The small Delta t helps with regularization.
        # If temporal resolution is changed, alpha_reg needs to change accordingly.
        K=int(K/delta_t) #K is now the total length of the tau_space

        #Set up matrix F:
        F=np.zeros((len(range_g),K))
        for i_g in range(0,len(range_g)):
            for i_t in range(0,K):
                F[i_g,i_t]=range_g[i_g]**(i_t*delta_t)

        # SVD of F:
        U, lam, V = sp.linalg.svd(F)


        #set up gamma-space:
        V_gamma2=V_gamma[:,1,:]
        Z=V_gamma2[:,0:-2]-V_gamma2[:,1:-1]

        #smooth gamma space (not necessary):
        #for h in range(0,num_h-2):
            #Z[:,h]= savgol_filter(Z[:,h], 7, 1)

        #Linearly recover tau-space from eigenspace of F:
        tau_space=np.zeros((K,num_h-2))
        for h in range(0,num_h-2):
            term=np.zeros((1,K))
            for i in range(0,len(lam)):
                fi=lam[i]**2/(alpha_reg**2+lam[i]**2)
                new=fi*(((U[:,i]@Z[:,h])*V[i,:] )/lam[i])
                term=term+new
            tau_space[:,h]=term

        #Smooth tau-space (not necessary):
        #for h in range(0,num_h-2):
            #tau_space[:,h]=savgol_filter(tau_space[:,h], 7, 1)


        #Normalization. It is crucial when K is large (this is, if a small Delta t is chosen)
        tau_space[tau_space<0]=0 #Probabilities are positive
        for i in range(0,len(tau_space)): #normalize
            if np.nansum(tau_space[i,:])>0.001:
                tau_space[i,:]=tau_space[i,:]/np.nansum(tau_space[i,:])
      
        fig2=plt.figure(dpi=150)
        ax2 = fig2.gca(projection='3d')

        X,Y=np.meshgrid(range_h[0:-2],delta_t*np.linspace(0,K,K))

        surf = ax2.plot_surface(X, Y, tau_space, cmap='summer'
                                  , edgecolor='none')


        ax2.view_init(50, -45)
        # plt.ylim(0, 3.1)
        ax2.set_zlim(0,0.06)
        fig2.canvas.draw_idle()
        ax2.set_zlabel('Probability')
        ax2.set_ylabel('Future Time')
        ax2.set_xlabel('Reward')
        st.write(fig2)
      else:
        env = gridworld2_final()
        env.reset()
        dim=env.dim[1]
        num_states=dim #number of timesteps in the MP
        alpha=0.01 #learning rate 
        num_gammas=100 #number of gammas

        # Use gammas equally separated in the 1/log(gamma) space:
        taus=np.linspace(0.01,3,num_gammas)
        range_g=np.exp(-1/taus)

        # Use gammas equally separated in the linear space:
        # range_g=np.linspace(0.0,1,num_gammas)


        num_trials=1000 #number of trials used to learn the TD code

        width=13 #steepness of the reward sensitivities
        num_h=250 #number of h's
        range_h=np.linspace(-3,3,num_h) #theta_h range
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
        V_gamma2=V_gamma[4:-1,0,:]
        Z=V_gamma2[:,0:-2]-V_gamma2[:,1:-1]
        X,Y=np.meshgrid(range_h[0:-2],range_g[4:-1]);

        for h in range(0,num_h-2):
            Z[:,h]=savgol_filter(Z[:,h], 15, 1)  

                
        fig1 = plt.figure(dpi=150)
        ax1 = fig1.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='summer'
                              , edgecolor='none',alpha=1)
        ax1.view_init(60, -45)
        # ax.set_zlim(0,0.15)
        ax1.grid(b=None)
        plt.yticks([0,0.5,1])
        plt.xticks([-2,-1,0,1,2])
        ax1.set_zlabel('Convergence value')
        ax1.set_ylabel('Temporal Discount')
        ax1.set_xlabel('Reward')
        st.write(fig1)
        alpha_reg=0.2 #Regularization parameter

        K=num_states #Temporal horizon
        delta_t=1 #Length of timestep

        #define matrix F:
        F=np.zeros((len(range_g),K))
        for i_g in range(0,len(range_g)):
            for i_t in range(0,K):
                F[i_g,i_t]=range_g[i_g]**(i_t*delta_t)

                
        U, lam, V = sp.linalg.svd(F) #SVD decomposition of F

        #set up gamma-space:
        V_gamma2=V_gamma[:,0,:]
        Z=V_gamma2[:,0:-2]-V_gamma2[:,1:-1]


        #smooth gamma-space (it might not be necessary, it helps if the input is *very* noisy):
        #for h in range(0,num_h-2):
            #Z[:,h]=savgol_filter(Z[:,h], 5, 1)

        #Linearly recover tau-space from eigenspace of F:
        tau_space=np.zeros((K,num_h-2))
        for h in range(0,num_h-2):
            term=np.zeros((1,K))
            for i in range(0,len(lam)):
                fi=lam[i]**2/(alpha_reg**2+lam[i]**2)
                new=fi*(((U[:,i]@Z[:,h])*V[i,:] )/lam[i])
                term=term+new
            tau_space[:,h]=term

            
        #smooth gamma-space (it might not be necessary, use for a smoother visualization):
        #for h in range(0,num_h-2):
            #tau_space[:,h]=savgol_filter(tau_space[:,h], 11, 1)


        #Normalization (it is not necessary for this very short temporal horizon T=4):
        tau_space[tau_space<0]=0 #make all probabilities positive
        for i in range(0,len(tau_space)): #normalize
            if np.nansum(tau_space[i,:])>0.0:
                tau_space[i,:]=tau_space[i,:]/np.nansum(tau_space[i,:])

        fig2=plt.figure(dpi=150)
        ax2 = fig2.gca(projection='3d')
        X,Y=np.meshgrid(range_h[0:-2],delta_t*np.linspace(0,K-1,K)) #grid to plot

        surf = ax.plot_surface(X, Y, tau_space, cmap='summer'
                                  , edgecolor='none',alpha=1)


        ax2.grid()

        ax2.view_init(60, -45)
        plt.yticks([0,1,2,3])
        plt.xticks([-2,-1,0,1,2])
        #ax.set_zlim(0,0.085)
        ax2.set_zlabel('Probability')
        ax2.set_ylabel('Future Time')
        ax2.set_xlabel('Reward')
        fig2.canvas.draw_idle()
        st.write(fig2)
      