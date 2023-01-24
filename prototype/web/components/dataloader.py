import pandas as pd


class DataLoader(object):
    def __init__(self, dataframe):
        self.df = dataframe
        # TODO some dataframe transformation

    def get_dataframe(self):
        return self.df
