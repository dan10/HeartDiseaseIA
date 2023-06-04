import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import scipy.stats as st
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from yellowbrick import ROCAUC
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, PrecisionRecallCurve


def check(dataframe):
    # Remove instances where 'RestingBP' and 'Cholesterol' are zero
    dataframe = dataframe[dataframe.RestingBP != 0]
    dataframe = dataframe[dataframe.Cholesterol != 0]

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
    # Remove instances where 'RestingBP' and 'Cholesterol' are zero
    dataframe = dataframe[dataframe.RestingBP != 0]
    dataframe = dataframe[dataframe.Cholesterol != 0]

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

    # Normalize the data
    scaler = StandardScaler()
    x_train_n = scaler.fit_transform(x_train)
    x_test_n = scaler.transform(x_test)

    # Save train and test sets to a file using pickle
    with open("heart_train_test_normalized.pkl", "wb") as f:
        pickle.dump((x_train_n.columns, x_train_n.values, x_test_n.values, y_train.values, y_test.values), f)


# Function for calculating confidence interval from cross-validation
def interval_confidence(values):
    return st.t.interval(confidence=0.95, df=len(values) - 1, loc=np.mean(values), scale=st.sem(values))


def fit_and_evaluate(model, x_train, x_test, y_train, y_test, feature_names):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    model.fit(x_train, y_train)

    precision_scorer_not_disease = make_scorer(precision_score, pos_label=0, average='binary')
    precision_scorer_disease = make_scorer(precision_score, pos_label=1, average='binary')

    recall_scorer_not_disease = make_scorer(recall_score, pos_label=0, average='binary')
    recall_scorer_disease = make_scorer(recall_score, pos_label=1, average='binary')

    f1_scorer_not_disease = make_scorer(f1_score, pos_label=0, average='binary')
    f1_scorer_disease = make_scorer(f1_score, pos_label=1, average='binary')

    score_precision_model_not_disease = cross_val_score(
        model, x_train, y_train, cv=kf, scoring=precision_scorer_not_disease
    )
    precision_ic_not_disease = interval_confidence(score_precision_model_not_disease)

    score_precision_model_disease = cross_val_score(model, x_train, y_train, cv=kf, scoring=precision_scorer_disease)
    precision_ic_disease = interval_confidence(score_precision_model_disease)

    score_recall_model_not_disease = cross_val_score(
        model, x_train, y_train, cv=kf, scoring=recall_scorer_not_disease
    )
    recall_ic_not_disease = interval_confidence(score_recall_model_not_disease)

    score_recall_model_disease = cross_val_score(model, x_train, y_train, cv=kf, scoring=recall_scorer_disease)
    recall_ic_disease = interval_confidence(score_recall_model_disease)

    score_f1_model_not_disease = cross_val_score(model, x_train, y_train, cv=kf, scoring=f1_scorer_not_disease)
    f1_ic_not_disease = interval_confidence(score_f1_model_not_disease)

    score_f1_model_disease = cross_val_score(model, x_train, y_train, cv=kf, scoring=f1_scorer_disease)
    f1_ic_disease = interval_confidence(score_f1_model_disease)

    test_score = model.score(x_test, y_test)
    print(f"Test score {model.__class__.__name__}", test_score)
    # y_pred = model.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)

    print(
        f"Precision medias de classe 0: {model.__class__.__name__}", score_precision_model_not_disease,
        "Media:", score_precision_model_not_disease.mean(), "=-", score_precision_model_not_disease.std()
    )
    print(f"Intervalo de confiança Precision do modelo {model.__class__.__name__} da classe NotDisease:",
          precision_ic_not_disease
          )

    print(f"Precision medias de classe 1: {model.__class__.__name__}", score_precision_model_disease,
          "Media:", score_precision_model_disease.mean(), "=-", score_precision_model_disease.std()
          )
    print(f"Intervalo de confiança Precision do modelo {model.__class__.__name__} da classe Disease:",
          precision_ic_disease
          )

    print("\n")

    print(
        f"Recall medias de classe 0: {model.__class__.__name__}", score_recall_model_not_disease,
        "Media:", score_recall_model_not_disease.mean(), "=-", score_recall_model_not_disease.std()
    )
    print(f"Intervalo de confiança Recall do modelo {model.__class__.__name__} da classe NotDisease:",
          recall_ic_not_disease
          )

    print(f"Recall medias de classe 1: {model.__class__.__name__}", score_recall_model_disease,
          "Media:", score_recall_model_disease.mean(), "=-", score_recall_model_disease.std()
          )
    print(f"Intervalo de confiança Recall do modelo {model.__class__.__name__} da classe Disease:",
          recall_ic_disease
          )

    print("\n")

    print(
        f"F-measure medias de classe 0: {model.__class__.__name__}", score_f1_model_not_disease,
        "Media:", score_f1_model_not_disease.mean(), "=-", score_f1_model_not_disease.std()
    )
    print(f"Intervalo de confiança F1 do modelo {model.__class__.__name__} da classe NotDisease: {f1_ic_not_disease}")

    print(f"F-measure medias de classe 1: {model.__class__.__name__}", score_f1_model_disease,
          "Media:", score_f1_model_disease.mean(), "=-", score_f1_model_disease.std()
          )
    print(f"Intervalo de confiança do modelo {model.__class__.__name__} da classe Disease: {f1_ic_disease}")

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
            'criterion': ['gini', 'entropy'],
            'random_state': [0]
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
        dtc_model = DecisionTreeClassifier(**grid_search.best_params_)
        fit_and_evaluate(dtc_model, x_train, x_test, y_train, y_test, feature_names)


def neural_network_grid_search():
    with open('heart_train_test_normalized.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        # Define the hyperparameter grid
        param_grid = {
            'solver': ['lbfgs', 'adam'],
            'hidden_layer_sizes': [(5, 2), (10, 5), (20, 10), (30, 15)],
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
            'max_iter': [1000, 5000, 10000, 200000, 5000000],
            'activation': ['tanh', 'relu'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'random_state': [1]
        }

        # Perform grid search with 5-fold cross-validation
        grid_search = GridSearchCV(
            MLPClassifier(),
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )

        # Fit the grid search object to the training data
        grid_search.fit(x_train, y_train)

        # Print the best hyperparameters and corresponding accuracy score
        print("Best hyperparameters for MLPClassifier:", grid_search.best_params_)
        print("Best accuracy score for MLPClassifier:", grid_search.best_score_)

        mlp_model = MLPClassifier(**grid_search.best_params_)
        fit_and_evaluate(mlp_model, x_train, x_test, y_train, y_test, feature_names)


def neural_network():
    with open('heart_train_test_normalized.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

    mlp_model = MLPClassifier(
        activation='relu',
        alpha=0.0001,
        hidden_layer_sizes=(10, 5),
        learning_rate='constant',
        max_iter=1000,
        random_state=1,
        solver='lbfgs'
     )

    fit_and_evaluate(mlp_model, x_train, x_test, y_train, y_test, feature_names)



def random_forest_grid_search():
    # open train and test sets
    with open('heart_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        # Define the hyperparameter grid
        param_grid = {
            'n_estimators': [10, 50, 100, 150, 200, 250, 300],
            'max_features': range(1, len(feature_names) + 1),
            'criterion': ['gini', 'entropy'],
            'random_state': [1]
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

def create_graphic():
    # Substitua esses valores pelas suas métricas
    NN = [0.9, 0.8, 0.85]  # NeuralNetwork
    RF = [0.8, 0.7, 0.75]  # Random Forest
    DT = [0.7, 0.6, 0.65]  # Decision Tree

    # Configurando a posição das barras no eixo X
    barWidth = 0.25
    r1 = np.arange(len(NN))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Criando as barras
    plt.bar(r1, NN, color='b', width=barWidth, edgecolor='grey', label='NeuralNetwork')
    plt.bar(r2, RF, color='g', width=barWidth, edgecolor='grey', label='RandomForest')
    plt.bar(r3, DT, color='r', width=barWidth, edgecolor='grey', label='DecisionTree')

    # Adicionando os nomes para o eixo X
    plt.xlabel('Métricas', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(NN))], ['Recall', 'Precision', 'F1-Score'])

    # Criando a legenda do gráfico
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)

    # Exibindo o gráfico
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv("heart.csv", delimiter=",")
    check(df)
    # preprocessing_train_test(df)
    decision_tree_grid_search()
    # random_forest_grid_search()
    #neural_network_grid_search()
    #neural_network()
    create_graphic()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
