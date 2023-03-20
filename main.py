import pickle

import pandas as pd
import pydotplus
from category_encoders import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from yellowbrick import ROCAUC
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, PrecisionRecallCurve


def preprocessing_train_test():
    df = pd.read_csv("heart.csv", delimiter=",")

    le_sex = LabelEncoder()
    le_exercise_angina = LabelEncoder()
    chest_pain_mapping = [{'col': 'ChestPainType', 'mapping': {'ASY': 0, 'NAP': 1, 'ATA': 2, 'TA': 3}}]
    oe_chest_pain = OrdinalEncoder(mapping=chest_pain_mapping)

    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["ExerciseAngina"] = le_exercise_angina.fit_transform(df["ExerciseAngina"])
    df = oe_chest_pain.fit_transform(df)

    df = pd.get_dummies(df, columns=["RestingECG", "ST_Slope"])

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=["HeartDisease"]),
        df["HeartDisease"],
        test_size=0.2,
        random_state=0
    )

    # Save train and test sets to a file using pickle
    with open("heart_train_test.pkl", "wb") as f:
        pickle.dump((x_train.columns, x_train.values, x_test.values, y_train.values, y_test.values), f)


def decision_tree():
    # open train and test sets
    with open('heart_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        # Train a decision tree classifier on the training set
        dtc_model = DecisionTreeClassifier()
        dtc_model.fit(x_train, y_train)

        # Evaluate the classifier on the test set
        y_pred = dtc_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        class_report = ClassificationReport(dtc_model, classes=["NotDisease", "Disease"])
        class_report.fit(x_train, y_train)
        class_report.score(x_test, y_test)
        class_report.show(outpath="heart_decision_tree_class_report.png", clear_figure=True)

        prc = PrecisionRecallCurve(dtc_model, classes=["NotDisease", "Disease"])
        prc.fit(x_train, y_train)
        prc.score(x_test, y_test)
        prc.show(outpath="heart_decision_tree_precision_recall_curve.png", clear_figure=True)

        roc = ROCAUC(dtc_model, classes=["NotDisease", "Disease"])
        roc.fit(x_train, y_train)
        roc.score(x_test, y_test)
        roc.show(outpath="heart_decision_tree_roc.png", clear_figure=True)

        cm = ConfusionMatrix(dtc_model, classes=["NotDisease", "Disease"])
        cm.fit(x_train, y_train)
        cm.score(x_test, y_test)
        cm.show(outpath="heart_decision_tree_confusion_matrix.png")


        dot_data = export_graphviz(
            dtc_model,
            out_file=None,
            feature_names=feature_names,
            class_names=["NotDisease", "Disease"],
            filled=True,
            rounded=True
        )
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png("heart_decision_tree_graph.png")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    preprocessing_train_test()
    decision_tree()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
