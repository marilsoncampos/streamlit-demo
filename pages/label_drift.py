import numpy as np
import pandas as pd
import streamlit as st
import json

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from deepchecks import Dataset
from deepchecks.checks import TrainTestFeatureDrift
from pages.page_utils import hide_footer


class DataDriftTester:
    def __init__(self):
        self.model = None
        self.dataset_train = None
        self.dataset_test = None
        self.df_train = None
        self.df_test = None
        self.result = None

    def build_data(self):
        np.random.seed(42)
        train_data = np.concatenate([
            np.random.randn(1000, 2),
            np.random.choice(a=['apple', 'orange', 'banana'], p=[0.5, 0.3, 0.2], size=(1000, 2))], axis=1)
        test_data = np.concatenate([
            np.random.randn(1000, 2),
            np.random.choice(a=['apple', 'orange', 'banana'], p=[0.5, 0.3, 0.2], size=(1000, 2))], axis=1)

        df_train = pd.DataFrame(
            train_data,
            columns=['numeric_without_drift', 'numeric_with_drift',
                     'categorical_without_drift', 'categorical_with_drift'])
        df_test = pd.DataFrame(test_data, columns=df_train.columns)

        df_train = df_train.astype({'numeric_without_drift': 'float', 'numeric_with_drift': 'float'})
        df_test = df_test.astype({'numeric_without_drift': 'float', 'numeric_with_drift': 'float'})

        df_test['numeric_with_drift'] = (
                df_test['numeric_with_drift'].astype('float') +
                abs(np.random.randn(1000)) +
                np.arange(0, 1, 0.001) * 4)
        df_test['categorical_with_drift'] = np.random.choice(
            a=['apple', 'orange', 'banana', 'lemon'],
            p=[0.5, 0.25, 0.15, 0.1], size=(1000, 1))

        model = Pipeline([
            ('handle_cat', ColumnTransformer(
                transformers=[
                    ('num', 'passthrough',
                     ['numeric_with_drift', 'numeric_without_drift']),
                    ('cat',
                     Pipeline([
                         ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                     ]),
                     ['categorical_with_drift', 'categorical_without_drift'])
                ]
            )),
            ('model', DecisionTreeClassifier(random_state=0, max_depth=2))]
        )
        label = np.random.randint(0, 2, size=(df_train.shape[0],))
        cat_features = ['categorical_without_drift', 'categorical_with_drift']
        df_train['target'] = label
        train_dataset = Dataset(df_train, label='target', cat_features=cat_features)

        model.fit(train_dataset.data[train_dataset.features], label)
        label = np.random.randint(0, 2, size=(df_test.shape[0],))
        df_test['target'] = label
        test_dataset = Dataset(df_test, label='target', cat_features=cat_features)

        self.dataset_train = train_dataset
        self.dataset_test = test_dataset
        self.model = model
        self.df_train = df_train
        self.df_test = df_test

    def run_tests(self):
        check = TrainTestFeatureDrift()
        self.result = check.run(
            train_dataset=self.dataset_train,
            test_dataset=self.dataset_test,
            model=self.model)

    def format_results(self):
        return json.dumps(self.result.value, indent=4, sort_keys=False)


def app():
    st.markdown("__Label Drift Demo__")
    tester = DataDriftTester()
    level = st.slider("Level of Noise", min_value=0, max_value=100)
    recalc = st.button('Calculate')
    hide_footer(st)
    if recalc:
        tester.build_data()
        tester.run_tests()
        st.markdown("__Results__")
        tester.result.show(show_additional_outputs=False)
        st.json(tester.format_results())
