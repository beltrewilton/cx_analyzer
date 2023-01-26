from st_on_hover_tabs import on_hover_tabs
import streamlit as st


class View(object):
    def __init__(self, component):
        self.component = component
        self.tabs = {}
        self.icons = []

    def add_tab(self, name, func, icon_name='token'):
        self.tabs[name] = func
        self.icons.append(icon_name)

    def render(self):
        # st.header("(aquí va el header).")
        # st.image('./img/head.png')

        import base64

        LOGO_IMAGE = "./img/logo.png"

        st.markdown(
            """
            <style>
            .logo-text {
                display: flex;
                margin-top: -80px !important;
            }
            .logo-img {
                float:right;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="logo-text">
                <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<style>' + open('./css/style.css').read() + '</style>', unsafe_allow_html=True)
        with st.sidebar:
            # st.image('./img/luzIA.png', width=100)
            tab = on_hover_tabs(tabName=list(self.tabs),
                                     iconName=self.icons, default_choice=0)

        self.tabs[tab](tab) #here the magic!
