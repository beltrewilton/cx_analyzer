import pandas as pd
import numpy as np
import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_echarts import st_echarts
import plotly.figure_factory as ff
from components.dataloader import DataLoader
from components.sentiment_emotions import SentimentEmotion
from components.topic_discovery import TopicDiscovery
from view.render import View

st.set_page_config(layout="wide", page_icon="🦈")


# dataframe = pd.read_csv('./dataset/Cosecha_OTONO_Final.csv', names=['edad', 'genero', 'departamento', 'tema'])
# dataframe.dropna(axis=0, inplace=True)


def home(tab):
    colored_header(
        label="Landing page",
        description="",
        color_name="violet-70",
    )
    st.markdown("For companies that genuinelly cares about their customers, `luzia` customer experience and voice of customer "
                "translates your customers voice in a structured, easy to read and valuable informacion because "
                "is realiable fast and cost saving.")
    st.write('La opción seleccionada  {}'.format(tab))


def magic(tab):
    colored_header(
        label="Dashboard",
        description="Detalle sobre análisis de emociones y sentimientos.",
        color_name="violet-70",
    )
    # st.write('La opción seleccionada  {}'.format(tab))

    def render_bar_sentiment(key=1, pos=1, neg=1, neu=1):
        options = {
            "xAxis": {
                "type": "category",
                "data": ["Positivo", "Negativo", "Neutro"],
            },
            "yAxis": {"type": "value"},
            "series": [
                {
                    "data": [
                        {"value": pos, "itemStyle": {"color": "#14e07e"}},
                        {"value": neg, "itemStyle": {"color": "#e01499"}},
                        {"value": neu, "itemStyle": {"color": "#d6cc6d"}},
                    ],
                    "type": "bar",
                }
            ],
        }
        st_echarts(
            options=options,
            height="400px",
            key=key
        )

    def render_pie_sentiment(key=1, title='Title', pos=0.33, neu=0.33, neg=0.33):
        options = {
            "title": {"text": f"{title}",  "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [
                {
                    "type": "pie",
                    "radius": "60%",
                    "data": [
                            {"value": pos, "name": "Positivo", "itemStyle": {"color": "#2eb2ff"}},
                            {"value": neu, "name": "Neutro", "itemStyle": {"color": "#ebf5bc"}},
                            {"value": neg, "name": "Negativo", "itemStyle": {"color": "#e30b69"}},
                        ],
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                        }
                    },
                }
            ],
        }
        st_echarts(
            options=options, height="400px", key=key
        )

    def render_pie_emotion(key=1, title='Title', joy=1/7, sadness=1/7, anger=1/7, surprise=1/7, disgust=1/7, fear=1/7, others=1/7 ):
        options = {
            "title": {"text": f"{title}",  "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [
                {
                    "type": "pie",
                    "radius": "60%",
                    "data": [
                            {"value": joy, "name": "Alegre", "itemStyle": {"color": "#5e0be3"}},
                            {"value": sadness, "name": "Triste", "itemStyle": {"color": "#850be3"}},
                            {"value": anger, "name": "Furia", "itemStyle": {"color": "#970be3"}},
                            {"value": surprise, "name": "Sorpresa", "itemStyle": {"color": "#b10be3"}},
                            {"value": disgust, "name": "Disgusto", "itemStyle": {"color": "#cd0be3"}},
                            {"value": fear, "name": "Miedo", "itemStyle": {"color": "#e30bd1"}},
                            # {"value": others, "name": "others", "itemStyle": {"color": "#9ea8a8"}},
                        ],
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                        }
                    },
                }
            ],
        }
        st_echarts(
            options=options, height="400px", key=key
        )

    def render_pie_hate(key=1, title='Title', hateful=0.33, targeted=0.33, aggressive=0.33):
        options = {
            "title": {"text": f"{title}",  "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [
                {
                    "type": "pie",
                    "radius": "60%",
                    "data": [
                            {"value": hateful, "name": "Odio", "itemStyle": {"color": "#ed5f00"}},
                            {"value": targeted, "name": "Enfoque", "itemStyle": {"color": "#eda600"}},
                            {"value": aggressive, "name": "Agresividad", "itemStyle": {"color": "#e5ed00"}},
                        ],
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                        }
                    },
                }
            ],
        }
        st_echarts(
            options=options, height="400px", key=key
        )

    uploaded_file = st.file_uploader("Cargá archivo", type={"csv"})
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file, sep=";", encoding_errors="ignore", encoding='utf_8')
        dataframe.dropna(axis=0, inplace=True)

        options = tuple(dataframe.columns)
        text_field = st.selectbox('Seleccioná el feature-text:', options=options)
        process = st.button('Proceder')
        if process:
            se = SentimentEmotion()
            td = TopicDiscovery()
            ### Preprocessing step in action
            dataframe = td.preprocessing(data_tr=dataframe, text_field=text_field)

            with st.spinner(text="Procesando la información, tome un café ☕ que ya casi ..."):
                ### Topic Discovering
                df_feaured = td.discover(data=dataframe, text_field=text_field,
                                         stopw_path='./components/stopwords_spanish.txt')
                df_feaured = df_feaured.sample(600, random_state=42)

                ### Extracting Sentiment and Emotionals features
                df_feaured = se.extract(df=df_feaured, text_field=text_field, sample=500)
                df_sent_mean = pd.DataFrame(df_feaured[se.extra_fields].mean(axis=0)).T
                df_sent_sum = pd.DataFrame(df_feaured[se.extra_fields].sum(axis=0)).T

                df_feaured.fillna(0, inplace=True)

            kol1, kol2, kol3 = st.columns(3)
            with kol1:
                pos = df_sent_sum['sent_pos'].values[0]
                neu = df_sent_sum['sent_neutral'].values[0]
                neg = df_sent_sum['sent_neg'].values[0]
                render_bar_sentiment(key=1, pos=pos, neu=neu, neg=neg)
            with kol2:
                joy = df_sent_mean['joy'].values[0]
                sadness = df_sent_mean['sadness'].values[0]
                anger = df_sent_mean['anger'].values[0]
                surprise = df_sent_mean['surprise'].values[0]
                disgust = df_sent_mean['disgust'].values[0]
                fear = df_sent_mean['fear'].values[0]
                others = df_sent_mean['others'].values[0]
                render_pie_emotion(key=2, title='Emociones', joy=joy, sadness=sadness, anger=anger, surprise=surprise, disgust=disgust, fear=fear, others=others)
            with kol3:
                hateful = df_sent_mean['hateful'].values[0]
                targeted = df_sent_mean['targeted'].values[0]
                aggressive = df_sent_mean['aggressive'].values[0]
                render_pie_hate(key=3, title='Hate', hateful=hateful, targeted=targeted, aggressive=aggressive)

            st.dataframe(df_feaured, width=1200, height=500)


    # st.warning('El gráfico mostrado es con propósito de pruebas.', icon="⚠️")
    #
    # # Add histogram data
    # x1 = np.random.randn(200) - 2
    # x2 = np.random.randn(200)
    # x3 = np.random.randn(200) + 2
    #
    # # Group data together
    # hist_data = [x1, x2, x3]
    #
    # group_labels = ['Group 1', 'Group 2', 'Group 3']
    #
    # # Create distplot with custom bin_size
    # fig = ff.create_distplot(
    #     hist_data, group_labels, bin_size=[.1, .25, .5])
    #
    # # Plot!
    # st.plotly_chart(fig, use_container_width=True)


def ous(tab):
    colored_header(
        label="Conocé",
        description="",
        color_name="violet-70",
    )
    # st.write('La opción seleccionada  {}'.format(tab))
    st.image('https://proyectorfantasma.com.ar/wp-content/uploads/2018/02/Fantastic-Four-reboot.jpg', width=300)


dataloader = DataLoader(None)
view = View(dataloader)
view.add_tab(name='Inicio', func=home, icon_name='dashboard')
view.add_tab(name='Magia', func=magic, icon_name='money')
view.add_tab(name='Nosotros', func=ous, icon_name='face')
view.render()
