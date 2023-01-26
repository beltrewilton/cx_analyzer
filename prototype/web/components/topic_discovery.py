import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TopicDiscovery(object):
    """Topic Discovery class based on LatentDirichletAllocation"""
    def __init__(self):
        print('##### TopicDiscovery __init__ #####')

    def preprocessing(self, data_tr, text_field):
        """Dataset preprocessing"""
        data_tr = data_tr.dropna()
        data_tr[text_field] = data_tr[text_field].str.lower()
        data_tr[text_field] = data_tr[text_field].replace(',', "")
        data_tr[text_field] = data_tr[text_field].replace('.', "")
        data_tr[text_field] = data_tr[text_field].replace(';', "")
        return data_tr

    def __get_stopwords(self, stop_file_path):
        """load stop words """
        with open(stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = f.readlines()
            stop_set = set(m.strip() for m in stopwords)
            stop_spanish = list(frozenset(stop_set))
            stop_spanish.extend(['si', 'ahi', 'ahí', 'ah'])
            return stop_spanish

    def discover(self, data, text_field, stopw_path):
        """Discovery fun algorithm :) """
        stop_spanish = self.__get_stopwords(stopw_path)
        count = CountVectorizer(stop_words=stop_spanish, max_df=0.1, max_features=5000)

        X = count.fit_transform(data[text_field].values)

        n_comp = np.where(len(data.index) < 500, 8,
                          np.where(len(data.index) < 1000, 10,
                                   np.where(len(data.index) < 2000, 12,
                                            np.where(len(data.index) < 3000, 15, 20))))

        n_comp = n_comp[()]
        n_comp = 5
        lda = LatentDirichletAllocation(n_components=n_comp, random_state=123, learning_method='batch')

        X_topics = lda.fit_transform(X)

        n_top_words = 3
        topics_discovered = []
        feature_names = count.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_, 1):
            #     print(f'Topic {topic_idx}')
            discovered = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            #     print(' '.join(discovered))
            topics_discovered.append(discovered)

        # topics_discovered = [t[0] for t in topics_discovered]
        topics_discovered = [f'{t[0]} | {t[1]} | {t[2]}' for t in topics_discovered]

        topics = pd.DataFrame(lda.transform(X))
        topics_encode = pd.DataFrame(np.where(topics >= 0.3, 1, 0))
        other = pd.DataFrame(np.where(topics_encode.sum(axis=1) == 0, 1, 0))
        other.columns = ["otros"]

        topics_encode.columns = topics_discovered

        full_data = data.join(topics_encode)
        full_data = full_data.join(other)

        return full_data
