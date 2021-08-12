import os
import sys
import streamlit as st
from PIL import Image

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
if __name__ == '__main__':
    st.set_page_config(page_title = "FB Blog - Josh")
    local_css("style.css")
    st.markdown("<img src = 'https://fb-logo-images.s3-us-west-2.amazonaws.com/fatbrain-logo-color-h.png' style='width:210px;height:50px;'>",unsafe_allow_html=True)

    st.title("Recap Blog Main Page")
    st.markdown("Version 1.0 | Author: Josh |  Build Date: 08/01/20")


    st.markdown("## Grid Cells")
    st.markdown("## Attention")
    st.markdown("### Transformer Reinforcement Learning")
    st.markdown("## Reinforcement Learning")
    st.markdown("### [Fast Reinforcement Learning] (https://share.streamlit.io/joshdey/fbblog/main/fastRL/fastRLBlog.py)")
    st.markdown("### [Distributional Reinforcement Learning] (https://share.streamlit.io/joshdey/fbblog/main/distributionalRL/distributionalRLBlog.py)")
    st.markdown("### [Dreamerv2 & Latent Spaces] (https://share.streamlit.io/joshdey/fbblog/main/dreamerBlog/dreamerBlog.py)")