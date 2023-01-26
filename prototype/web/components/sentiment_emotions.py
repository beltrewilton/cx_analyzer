import pandas as pd
import sys; sys.path.append('..')
from pretrain.pretrained_models import models
pd.set_option('max_colwidth', None)


class SentimentEmotion(object):
     extra_fields = ['sent_neg', 'sent_neutral', 'sent_pos', 'joy', 'sadness',
                     'anger', 'surprise', 'disgust', 'fear', 'others', 'hateful', 'targeted', 'aggressive']
     def __init__(self):
          self.sentiment_analyzer, self.emotion_analyzer, self.hate_speech_analyzer, \
          self.ner_analyzer, self.pos_tagger = models()

     def __extract_sentiment(self, text):
          probas = self.sentiment_analyzer.predict(text).probas
          return probas['NEG'], probas['NEU'], probas['POS']

     def __extract_emotion(self, text):
          probas = self.emotion_analyzer.predict(text).probas
          return probas['joy'], probas['sadness'], probas['anger'], probas['surprise'], probas['disgust'], probas[
               'fear'], probas['others']

     def __extract_hate(self, text):
          probas = self.hate_speech_analyzer.predict(text).probas
          return probas['hateful'], probas['targeted'], probas['aggressive']

     def extract(self, df, text_field, sample=30):
          self.text_field = text_field  # 'tema'
          dataframe = df
          if sample is not None:
               textual = dataframe.sample(sample, random_state=33)
          else:
               textual = dataframe
          textual['sent_neg'], textual['sent_neutral'], textual['sent_pos'] = zip(
               *textual[self.text_field].apply(lambda x: self.__extract_sentiment(x)))

          textual['joy'], textual['sadness'], textual['anger'], textual['surprise'], textual['disgust'], textual[
               'fear'], textual['others'] = zip(*textual[self.text_field].apply(lambda x: self.__extract_emotion(x)))

          textual['hateful'], textual['targeted'], textual['aggressive'] = zip(
               *textual[self.text_field].apply(lambda x: self.__extract_hate(x)))

          return textual
