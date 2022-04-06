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
        self.adjusted_probs = None
        self.adjusted_ratio = None

    @classmethod
    def calculate_adjusted_probs(cls, level):
        # baseline = [0.5, 0.3, 0.2, 0]
        # original =[0.5, 0.25, 0.15, 0.1]
        step = 0.05 * ((level - 10) / 10.0)
        return [0.5, 0.25 + step, 0.15 - step, 0.1]

    @classmethod
    def calculate_adjusted_ratio(cls, level):
        return 1 + ((level - 10) / 12.0)

    def build_data(self, level):
        self.adjusted_probs = self.calculate_adjusted_probs(level)
        self.adjusted_ratio = self.calculate_adjusted_ratio(level)
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
                np.arange(0, 1, 0.001) * 4 * self.adjusted_ratio)
        df_test['categorical_with_drift'] = np.random.choice(
            a=['apple', 'orange', 'banana', 'lemon'],
            p=self.adjusted_probs, size=(1000, 1))  # [0.5, 0.25, 0.15, 0.1], [0.5, 0.3, 0.2, 0]

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
    hide_footer(st)
    side_col, dummy, work_area = st.columns([10, 1, 18])

    side_col.markdown("__Feature Drift__")
    level = side_col.slider("Inject drift", min_value=0, max_value=20, step=1, value=10)
    recalc = side_col.button('Calculate')

    tester = DataDriftTester()
    if recalc:
        tester.build_data(level)
        tester.run_tests()
        work_area.markdown("__Drift parameters__")
        work_area.text('Categorical Probabilities : {0}'.format([round(x, 3) for x in tester.adjusted_probs]))
        work_area.text('Numeric adjustment ratio  : {0}'.format(round(tester.adjusted_ratio, 3)))

        work_area.markdown("__Categorical Values Comparison__")
        cat_data = tester.df_test[['categorical_without_drift', 'categorical_with_drift']].copy()
        cat_data_counts1 = cat_data.groupby('categorical_without_drift').count()
        work_area.bar_chart(cat_data_counts1, height=160)

        cat_data_counts2 = cat_data.groupby('categorical_with_drift').count()
        work_area.bar_chart(cat_data_counts2, height=160)

        # chart_data = tester.df_test[['categorical_without_drift', 'categorical_with_drift']].copy()
        chart_data = tester.df_test[['numeric_without_drift', 'numeric_with_drift']].copy()
        chart_data.columns = ['clean', 'drift']
        work_area.markdown("__Numeric Values Comparison__")
        work_area.line_chart(chart_data, height=120)

        work_area.markdown("__Train Dataframe__")
        work_area.dataframe(tester.df_train.head())

        work_area.markdown("__Test Dataframe__")
        work_area.dataframe(tester.df_test.head())

        work_area.markdown("__Results__")
        work_area.json(tester.format_results())
