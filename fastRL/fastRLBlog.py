import streamlit as st

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


if __name__ == '__main__':
    st.set_page_config(page_title = "Fast RL Blog")
    local_css("style.css")
    st.markdown("<img src = 'https://fb-logo-images.s3-us-west-2.amazonaws.com/fatbrain-logo-color-h.png' style='width:210px;height:50px;'>",unsafe_allow_html=True)

    st.title("Fast Reinforcement Learning")
    st.markdown("Version 0.2 | Author: Josh |  Build Date: 08/01/20")

    st.markdown('## Background')
    st.write('When a person who\'s been learning piano for a while is tasked with learning a new difficult piece, they can typically learn it pretty fast. When a person who\'s never touched a piano is tasked with learning a new piece, even of roughly easy difficuilty, the learning curve may be relatively steep. Similarly, when a reinforcement learning agent is tasked with learning an atari game, it typically learns from scratch resulting in a steep learning curve. What fast reinforcement learning aims to accomplish is leveraging knowledge from prior tasks to learn newer ones faster.')
    st.markdown('### Model Free and Model Based Reinforcement Learning')
    st.write('As their names suggest, at their essence, a model based RL algorithm will try to model the environment and different paths with information as to quality of rewards at different locations in the environment and a model free RL algorithm creates a much more succinct representation of the environment, typically just expected values of different paths an agent can take. Since the model free algorithm\'s representation of the world is more succinct, it makes sense that they would be much faster to train; however, if an agent changes preferences as to the type of reward it wants to receive, the algorithm won\'t be able to take this account as easily. A model based representation takes much more computational power to train but can adapt a lot more easily. This trade-off makes choosing and training an RL agent on a task difficult and not very neurally plausible.')
    st.markdown('### Successor Features')
    st.write('As a middle ground to speed up the RL training process while still being relatively flexible in terms of preferences, successor features are introduced - rather than a single expected value in a model free algorithm, multiple features are quantified and represented in the model, even if it doesn\'t necessarily immediately impact the agent\'s preferences. This allows for different policies to be evaluated under different preferences, in a process known as generalized policy evaluation, or GPE. More information regarding successor features in references')

    st.markdown('## References')
    st.markdown('<sup>2</sup> [Universal Successor Features Approximators] (https://arxiv.org/pdf/1812.07626v1.pdf)')
    st.markdown(<sup>'3'</sup> '[The Successor Representation: Its Computational Logic and Neural Substrates:] (https://www.jneurosci.org/content/38/33/7193)')
