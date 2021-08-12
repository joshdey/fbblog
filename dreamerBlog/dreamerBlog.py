import os
import sys
import streamlit as st

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(page_title = "Dreamerv2")
    local_css("style.css")
    st.markdown("<img src = 'https://fb-logo-images.s3-us-west-2.amazonaws.com/fatbrain-logo-color-h.png' style='width:210px;height:50px;'>",unsafe_allow_html=True)

    st.title("Dreamerv2")
    st.markdown("Version 0.3 | Author: Josh |  Build Date: 08/01/20")

    st.write("[Dreamerv2 Paper](https://arxiv.org/abs/2010.02193)")
    st.write("[Dreamerv2 Blog Post](https://ai.googleblog.com/2021/02/mastering-atari-with-discrete-world.html?m=1)")
    st.write("[Dreamerv2 GitHub](https://github.com/danijar/dreamerv2)")

    st.markdown("## Background")
    st.write('The paper, *Mastering Atari with Discrete World Models* by Danijar Hafner et al., is the first reinforcement learning agent to achieve human-level performance on Atari games. Dreamerv2 is a model based reinforcement learning method in which the Dreamer agent, using some level of past experience, will "dream" out a sequence of the game for various actions. Model-based reinforcement learning involves the system learning some representation of the world or environment the agent lives in and then either coming up the best action given a set of actions and associated rewards. In model-free reinforcement learning, the agent learns a given policy that maximizes rewards as opposed to learning a representation of the world. In the case of Dreamerv2, this learned world model is a discrete one, representing the latent space as 32 distributions over 32 classes of categorical variables. Within its "dream", the dreamer agent uses an actor-critic model. In general, there are value based methods and policy based methods to train an agent, and actor-critic aims to merge both of these methods. The actor outputs the best action given its current learned policy, and the critic will score this action by computing a value function. Over time through a feedback loop, both the actor and critic will improve in their role of generating a policy or estimating a value function, and generally speaking the agent will act out in the environment better and faster compared to either a policy-based method or value-based method on its own.')

    st.markdown('## Architecture')
    st.markdown('### World Model')
    st.write('We can break down the architecture of the Dreamerv2 Model into 3 main parts, the first being the world model generation. A recurrent state-space model encapsulates the entire architecture. A convolutional encoder maps an input image to a stochastic representation which is then stored and kept track of in a recurrent state. Using a convolutional decoder, the model then tries to recreate the input image and predict the reward at each image step. The RSSM aims to also predict the latent space of the input images so that when dreaming, no input image is required but rather uses what it\'s learned about the latent space at various points in time. The original Dreamerv1 agent used continuous variables while the updated architecture represents each image with categorical variables with a multimodal distribution, hence the name "*Discrete* World Model". The aforementioned encoder converts the input image to a representation of 32 distributions over 32 classes and then one-hot encoded for a sparse representation and also kept track of in the recurrent state. The updated dreamer architecture uses straight through gradients to backpropogate through samples. The second key difference between Dreamerv1 and Dreamerv2 is the use of a loss function, KL balancing to train the prediction of the input and regularize how much information the stochastic representation keeps from the input image so as not to overtrain to information from early on in the training process while still using some past information to predict rewards and image reconstruction to learn long-term dependencies.')
    st.markdown('### Actor-Critic')
    st.write('Once the system is effectively "dreaming", the architecture uses an actor-critic model to better learn which actions to take in which states. Here, an input image is passed through our convolutional encoder to create our stochastic representation. From this representation, our actor (in the form of a multilayer perceptron, MLP) implements an action to receive some reward, and our critic, also an MLP, aims to accurately estimate a given value for a sum of future rewards given said action. The actor aims to maximize the critic output (such that the sum of future rewards is maximized) and these two networks act out a given number of steps within our environment to act in a symbiotic relationship to aid in maximizing the other\'s output.')
    st.markdown('### Increase Experience')
    st.write('After the actor-critic system dreams for a specified number of steps, the agent stops *dreaming* to return to the actual environment and use the learned policy to gain even more experience. The actor-critic system learns a policy up to a certain number of steps, and after returning to the actual environment the agent takes in more image inputs to further learn future stochastic representations to then go back to the dream to determine an optimal policy for the new future dataset.')

    st.markdown('## Atari Performance')
    st.write('Below you can see some plots comparing the performance of Dreamerv2 on various Atari games to show that it does outperform various model-free and model-based methods. **Input Images - still have to do**')

    st.markdown('## An Open Question')
    st.write('It\'s clear the Dreamerv2 architecture does indeed work a lot better than previous approaches, including the Dreamerv1 approach, indicating this performance boost is due to one or both of the two significant changes between the two methods: the new discrete world model representation of the environment and the introduction of the new KL Loss function optimizing between past and current inputs. Yet it\'s still unclear exactly how and why these would lead to a significant boost in performance for Atari games. We can investigate this by creating our own environment with images for which latent representations won\'t be so complex, handwritten MNIST digit images. An image of a grid world MNIST digit environment is shown below.')
    st.image('mnist_env.png', caption='The agent starts in State 0 (S0) and moves along either the upper or lower path, hopefully learning to choose the upper path more often and receive the higher reward.')
    st.write('After modifying Dreamerv2 Code to accept the new environment and preprocessing for the MNIST images rather than Atari images, we can analyze some key metrics from the new run, namely the latent space representation, the rewards over time, and the value of being in any given state. Below is an image of the moving average of performance over time in the new MNIST environment.')
    st.image('DreamerMNISTRun1.100.png', caption='The moving average of the reward at the beginning of the run lies around 7.5, indicating roughly a 50/50 split between moving down either of the two paths and as the training time increases it slowly approaches 9.5 indicating a policy of choosing the path leading to a higher reward and increasing number of times.')

