import streamlit as st
import numpy as np
from PIL import Image

# Custom imports 
from multipage import MultiPage
from pages import feature_drift, label_drift

app = MultiPage()

display = Image.open('res/deepchecks.png')
display_arr = np.array(display)
col1, col2 = st.columns(2)
col1.image(display_arr, width=200)
# col2.markdown("Deepchecks demo")

app.add_page("Feature drift", feature_drift.app)
app.add_page("Label drift", label_drift.app)

app.run()
