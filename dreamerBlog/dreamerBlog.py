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
    st.markdown("Version 1.0 | Author: Josh |  Build Date: 08/01/20")

    st.write("[Dreamerv2 Paper](https://arxiv.org/abs/2010.02193)")
    st.write("[Dreamerv2 Blog Post](https://ai.googleblog.com/2021/02/mastering-atari-with-discrete-world.html?m=1)")
    st.write("[Dreamerv2 GitHub](https://github.com/danijar/dreamerv2)")

