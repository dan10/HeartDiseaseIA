import pickle

import pandas as pd
import pydotplus
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from yellowbrick import ROCAUC
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, PrecisionRecallCurve


def check(dataframe):
    # count the number of samples in each class in the original dataset
    original_counts = dataframe['HeartDisease'].value_counts()

    print(original_counts)

    # Calculate the imbalance ratio
    imbalance_ratio = original_counts.min() / original_counts.max()

    # Print the imbalance ratio
    print(f'Imbalance Ratio: {imbalance_ratio:.2f}')

    duplicates = dataframe[dataframe.duplicated()]

    if duplicates.shape[0] == 0:
        print("There are no duplicate instances in the dataset.")
    else:
        print("There are {} duplicate instances in the dataset.".format(duplicates.shape[0]))


def preprocessing_train_test(dataframe):
    le_sex = LabelEncoder()
    le_exercise_angina = LabelEncoder()
    chest_pain_mapping = [{'col': 'ChestPainType', 'mapping': {'ASY': 0, 'NAP': 1, 'ATA': 2, 'TA': 3}}]
    oe_chest_pain = OrdinalEncoder(mapping=chest_pain_mapping)

    dataframe["Sex"] = le_sex.fit_transform(dataframe["Sex"])
    dataframe["ExerciseAngina"] = le_exercise_angina.fit_transform(dataframe["ExerciseAngina"])
    dataframe = oe_chest_pain.fit_transform(dataframe)

    dataframe = pd.get_dummies(dataframe, columns=["RestingECG", "ST_Slope"])

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=["HeartDisease"]),
        dataframe["HeartDisease"],
        test_size=0.2,
        random_state=0
    )

    # Save train and test sets to a file using pickle
    with open("heart_train_test.pkl", "wb") as f:
        pickle.dump((x_train.columns, x_train.values, x_test.values, y_train.values, y_test.values), f)


def fit_and_evaluate(model, x_train, x_test, y_train, y_test, feature_names):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    class_report = ClassificationReport(model, classes=["NotDisease", "Disease"])
    class_report.fit(x_train, y_train)
    class_report.score(x_test, y_test)
    class_report.show(outpath=f"{model.__class__.__name__}_class_report.png", clear_figure=True)

    prc = PrecisionRecallCurve(model, classes=["NotDisease", "Disease"])
    prc.fit(x_train, y_train)
    prc.score(x_test, y_test)
    prc.show(outpath=f"{model.__class__.__name__}_precision_recall_curve.png", clear_figure=True)

    roc = ROCAUC(model, classes=["NotDisease", "Disease"])
    roc.fit(x_train, y_train)
    roc.score(x_test, y_test)
    roc.show(outpath=f"{model.__class__.__name__}_roc.png", clear_figure=True)

    cm = ConfusionMatrix(model, classes=["NotDisease", "Disease"])
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.show(outpath=f"{model.__class__.__name__}_confusion_matrix.png")

    plt.close()

    if isinstance(model, DecisionTreeClassifier):
        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=feature_names,
            class_names=["NotDisease", "Disease"],
            filled=True,
            rounded=True
        )
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png(f"{model.__class__.__name__}_graph.png")


def decision_tree_grid_search():
    # open train and test sets
    with open('heart_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        range_array = list(range(1, len(feature_names) + 1))
        range_array.append(None)

        param_grid = {
            'max_depth': range_array,
            'min_samples_split': range(2, len(feature_names) + 1),
            'min_samples_leaf': range(1, len(feature_names) + 1),
            'criterion': ['gini', 'entropy']
        }

        # Perform a grid search with cross-validation
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=0),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        # Fit the grid search object to the training data
        grid_search.fit(x_train, y_train)

        # Print the best hyperparameters and the corresponding score
        print("Best hyperparameters for DecisionTree:", grid_search.best_params_)
        print("Best score for DecisionTree:", grid_search.best_score_)

        # Train a decision tree classifier on the training set
        dtc_model = DecisionTreeClassifier(**grid_search.best_params_, random_state=0)
        fit_and_evaluate(dtc_model, x_train, x_test, y_train, y_test, feature_names)


def random_forest_grid_search():
    # open train and test sets
    with open('heart_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        # Define the hyperparameter grid
        param_grid = {
            'n_estimators': [10, 50, 100, 150, 200, 250, 300],
            'max_features': range(1, len(feature_names) + 1),
            'criterion': ['gini', 'entropy']
        }

        # Perform grid search with 5-fold cross-validation
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=0),
            param_grid=param_grid,
            cv=5,
            n_jobs=-1
        )
        grid_search.fit(x_train, y_train)

        # Print the best hyperparameters and corresponding accuracy score
        print(f"Best parameters for RandomFlorest: {grid_search.best_params_}")
        print(f"Best accuracy score for RandomFlorest: {grid_search.best_score_}")

        rfc_model = RandomForestClassifier(**grid_search.best_params_, random_state=0)
        fit_and_evaluate(rfc_model, x_train, x_test, y_train, y_test, feature_names)

        importances = rfc_model.feature_importances_

        # Sort the features by importance in descending order
        indices = importances.argsort()[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for i in range(x_train.shape[1]):
            print("%d. feature %s (%f)" % (i + 1, feature_names[indices[i]], importances[indices[i]]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv("heart.csv", delimiter=",")
    check(df)
    preprocessing_train_test(df)
    decision_tree_grid_search()
    random_forest_grid_search()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
