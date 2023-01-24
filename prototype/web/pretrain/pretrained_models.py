import streamlit as st
from pysentimiento import create_analyzer


@st.experimental_singleton
def models():
    analyzer = create_analyzer(task="sentiment", lang="es")
    emotion_analyzer = create_analyzer(task="emotion", lang="es")
    hate_speech_analyzer = create_analyzer(task="hate_speech", lang="es")
    ner_analyzer = create_analyzer("ner", lang="es")
    pos_tagger = create_analyzer("pos", "es")
    return analyzer, emotion_analyzer, hate_speech_analyzer, ner_analyzer, pos_tagger
