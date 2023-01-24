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
        st.header("Aplicación en progreso (aquí va el header).")
        st.markdown('<style>' + open('./css/style.css').read() + '</style>', unsafe_allow_html=True)
        with st.sidebar:
            # st.image('./img/logo.png', width=100)
            tab = on_hover_tabs(tabName=list(self.tabs),
                                     iconName=self.icons, default_choice=0)

        self.tabs[tab](tab) #here the magic!
