import pandas as pd
import numpy as np
import s3fs
from sklearn.model_selection import train_test_split




class Pipeline(object):

    def __init__(self, path):
        # using chunks while on local machine
        chunks = pd.read_csv(path,index_col=0, parse_dates=[0], skip_blank_lines=True, iterator=True)
        self.df = chunks.get_chunk(40000)
        # X and y values to be assigned when create_holdout() is run
        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None

        self.X_test = None
        self.y_test = None

        self.X_holdout = None
        self.y_holdout = None

        # to be assigned if a groupby is necessary
        self.grouped_avg = None

    @classmethod
    def from_df(cls, df):
        obj = cls.__new__(cls)
        super(Pipeline, obj).__init__()
        obj.df = df
        return obj

    def read(self,path):
        pass

    def reset_index(self):
        self.df.index = pd.to_datetime(self.df.index, utc=True)
        return

    def merge_dfs(self,new_df):
        return Pipeline.from_df(self.df.merge(new_df, right_index=True, left_index=True))

    def getXy(self,target):
        "Target (string): name of the column that contains the y values"
        self.y = self.df.pop(target)
        self.X = self.df
        return self.X, self.y

    def consolidate(self, group_on):
        gb = self.df.groupby(group_on)
        self.grouped_avg = gb.mean()
        return

    def featurize_col(self, origin_col, new_features):
        '''
        when a column contains values that share an index

        origin_col: str name of column that holds soon-to-be features
        new_features: list of strings that appear in a column, to be made their own columns.

        ex: 

        '''

        for feat in new_features:
            for col in self.df.columns:
                self.df[f"{feat}_{col}"] = self.df[col][self.df[origin_col] == feat]

    def featurize_cities(self, names):
        new_dfs = []
        for i in names:
            city = self.df[self.df['city_name'] == i]
            new_cols = [f"{i}_{j}" for j in city.columns]
            mapper = {k:v for (k, v) in zip(city.columns, new_cols)}
            new_df = city.rename(mapper, axis=1)
            new_dfs.append(new_df)
        
        return new_dfs


    def clean_categoricals(self, cols):
        'takes a list of columns to make dummies and drop from the df'
        for i in cols:
            dummies = pd.get_dummies(i)
            pd.concat([self.df, dummies], axis=1)
            self.df.drop(i)



    def create_holdout(self):
        X_cv, self.X_holdout, y_cv, self.y_holdout = train_test_split(self.X,self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X_cv, y_cv)

        return