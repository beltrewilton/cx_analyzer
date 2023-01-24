import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.figure_factory as ff
from components.dataloader import DataLoader
from view.render import View

from pretrain.pretrained_models import models

st.set_page_config(layout="wide", page_icon="🦈")

dataframe = pd.read_csv('./dataset/Cosecha_OTONO_Final.csv', names=['edad', 'genero', 'departamento', 'tema'])
dataframe.dropna(axis=0, inplace=True)


def home(tab):
    st.title("Landing page")
    st.markdown("For companies that genuinelly cares about their customers, `luzia` customer experience and voice of customer "
                "translates your customers voice in a structured, easy to read and valuable informacion because "
                "is realiable fast and cost saving.")
    st.write('La opción seleccionada  {}'.format(tab))

def magic(tab):
    st.title("Aqui un poco de magia")
    st.write('La opción seleccionada  {}'.format(tab))

    st.warning('El gráfico mostrado es con propósito de pruebas.', icon="⚠️")

    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    # Group data together
    hist_data = [x1, x2, x3]

    group_labels = ['Group 1', 'Group 2', 'Group 3']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])

    # Plot!
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(dataframe, width=1200, height=500)



def ous(tab):
    st.title("Conocé")
    st.write('La opción seleccionada  {}'.format(tab))
    st.image('https://proyectorfantasma.com.ar/wp-content/uploads/2018/02/Fantastic-Four-reboot.jpg', width=300)


dataloader = DataLoader(dataframe)
view = View(dataloader)
view.add_tab(name='Inicio', func=home, icon_name='dashboard')
view.add_tab(name='Magia', func=magic, icon_name='money')
view.add_tab(name='Nosotros', func=ous, icon_name='face')
view.render()
