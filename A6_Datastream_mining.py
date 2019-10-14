from skmultiflow.trees import HoeffdingTree
from scipy.io import arff
import pandas as pd
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import DataStream
from sklearn.preprocessing import LabelEncoder
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.trees import RegressionHAT, HATT, LCHT, MultiTargetRegressionHoeffdingTree, RegressionHoeffdingTree

# Load data
data = arff.loadarff("elecNormNew.arff")
data_df = pd.DataFrame(data[0])

# Encode target class, 1 = UP 0 = DOWN
label_enc = LabelEncoder()
new_class_col = label_enc.fit_transform(data_df["class"])
data_df["class"] = new_class_col

# set target and feature columns
target_class = ["class"]
feature_cols = data_df.drop(["class"], axis=1)

# Use datastream to get in suitable form
stream = DataStream(feature_cols, data_df[target_class])
stream.prepare_for_use()


def question_1(data_stream):

    # Instantiate models
    h_tree = HoeffdingTree(nominal_attributes=[0, 1])
    n_bayes = NaiveBayes(nominal_attributes=[0, 1])

    # Perform PrequentialEvaluation to compare the two models NB and Htree
    """
    for max_samples in range(10000, 20000, 1000):
        evaluator = EvaluatePrequential(max_samples=max_samples, metrics=['accuracy'], n_wait=1500)
        evaluator.evaluate(stream=data_stream, model=[h_tree, n_bayes], model_names=["hTree", "NBayes"])
    """
    evaluator = EvaluatePrequential(max_samples=1800, metrics=['accuracy'], n_wait=1500)
    evaluator.evaluate(stream=data_stream, model=[h_tree, n_bayes], model_names=["hTree", "NBayes"])


def question_2(data_stream):

    h_tree = HoeffdingTree(nominal_attributes=[0, 1], leaf_prediction="mc")
    ha_tree = HAT(nominal_attributes=[0, 1])

    evaluator = EvaluatePrequential(max_samples=1800, metrics=['accuracy'], n_wait=1500)
    evaluator.evaluate(stream=data_stream, model=[h_tree, ha_tree], model_names=["HTree", "HAT"])


def question_3(data_df):
    label_enc = LabelEncoder()
    new_class_col = label_enc.fit_transform(data_df["class"])
    data_df["class"] = new_class_col

    # set target and feature columns
    target_class = ["class"]
    feature_cols = data_df.drop(["class"], axis=1)

    # Use datastream to get in suitable form
    data_stream = DataStream(feature_cols, data_df[target_class])
    data_stream.prepare_for_use()

    hatt = HATT(nominal_attributes=[0, 1])

    evaluator = EvaluatePrequential(max_samples=1800, metrics=['accuracy'], n_wait=1500)
    evaluator.evaluate(stream=data_stream, model=[hatt], model_names=["HATT"])


#question_1(stream)
#question_2(stream)
question_3(data_df)
