import pandas as pd
from web.pretrain.pretrained_models import models
pd.set_option('max_colwidth', None)

sentiment_analyzer, emotion_analyzer, hate_speech_analyzer, ner_analyzer, pos_tagger = models()

dataframe = pd.read_csv('./web/dataset/Cosecha_OTONO_Final.csv', names=['edad', 'genero', 'departamento', 'tema'])
dataframe.dropna(axis=0, inplace=True)

text_field = 'tema'

def extract_sentiment(text):
     probas = sentiment_analyzer.predict(text).probas
     return probas['NEG'], probas['NEU'], probas['POS']


def extract_emotion(text):
     probas = emotion_analyzer.predict(text).probas
     return probas['joy'], probas['sadness'], probas['anger'], probas['surprise'], probas['disgust'], probas['fear'], probas['others']


def extract_hate(text):
     probas = hate_speech_analyzer.predict(text).probas
     return probas['hateful'], probas['targeted'], probas['aggressive']


textual = dataframe.sample(10, random_state=33)
textual['sent_neg'], textual['sent_neutral'], textual['sent_pos'] = zip(*textual[text_field].apply(lambda x: extract_sentiment(x)))
textual['joy'], textual['sadness'], textual['anger'], textual['surprise'], textual['disgust'], textual['fear'], textual['others'] = zip(*textual[text_field].apply(lambda x: extract_emotion(x)))
textual['hateful'], textual['targeted'], textual['aggressive'] = zip(*textual[text_field].apply(lambda x: extract_hate(x)))

print()
