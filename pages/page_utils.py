

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """


def hide_footer(st):
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
