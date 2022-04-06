"""
A Multi-page manager
"""


import streamlit as st
import numpy as np
from PIL import Image


class ApxMultiPage:

    def __init__(self, image_path, version_str) -> None:
        self.pages = []
        self.image_path = image_path
        self.version_str = version_str
    
    def add_page(self, title, func) -> None: 
        self.pages.append(
            {
                "title": title, 
                "function": func
            }
        )

    def run(self):
        display = Image.open(self.image_path)
        display_arr = np.asarray(display)
        st.sidebar.image(display_arr, width=200)
        st.sidebar.container()
        st.sidebar.markdown('   Version: {0}'.format(self.version_str))
        page = st.sidebar.selectbox(
            'App Navigation', 
            self.pages, 
            format_func=lambda page: page['title']
        )
        page['function']()
