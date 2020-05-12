import pandas as pd
import numpy as np
import s3fs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class Pipeline(object):

    def __init__(self, path):
        # using chunks while on local machine
        #chunks = pd.read_csv(path,index_col=0, parse_dates=[0], skip_blank_lines=True, iterator=True)
        #self.df = chunks.get_chunk(40000)

        # for aws
        self.df = pd.read_csv(path,index_col=0, parse_dates=[0], skip_blank_lines=True)

        # X and y values to be assigned when create_holdout() is run
        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None

        self.X_test = None
        self.y_test = None

        self.X_holdout = None
        self.y_holdout = None

        self.X_std = None
        self.y_std = None

        # to be assigned if a groupby is necessary
        self.grouped_avg = None

    @classmethod
    def from_df(cls, df):
        obj = cls.__new__(cls)
        super(Pipeline, obj).__init__()
        obj.df = df
        return obj

    def my_reset_index(self):
        self.df.index = pd.to_datetime(self.df.index, utc=True)
        return

    def merge_dfs(self,new_df):
        return Pipeline.from_df(self.df.merge(new_df, right_index=True, left_index=True))

    def consolidate(self, group_on):
        gb = self.df.groupby(group_on)
        self.grouped_avg = gb.mean()
        return

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

    def standardize(self, X, y):
        "do this AFTER your 1st train/test/split, no leakage here."
        Xscaler = StandardScaler()
        yscaler = StandardScaler()
        self.X_std = Xscaler.fit_transform(X)
        self.y_std = yscaler.fit_transform(y)
        return


    def getXy(self,target):
        "Target (string): name of the column that contains the y values"
        self.y = self.df.pop(target)
        self.X = self.df
        return self.X, self.y

    def create_holdout(self):
        X_cv, self.X_holdout, y_cv, self.y_holdout = train_test_split(self.X,self.y)
        self.standardize(X_cv, y_cv) #standardizes the cv set after holdout is created
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X_cv, y_cv)

        return
