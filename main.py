import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import scipy.stats as st
from category_encoders import OrdinalEncoder
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train_n = scaler.fit_transform(x_train)
    x_test_n = scaler.transform(x_test)

    # Save train and test sets to a file using pickle
    with open("heart_train_test_normalized.pkl", "wb") as f:
        pickle.dump((x_train.columns, x_train_n, x_test_n, y_train.values, y_test.values), f)


# Function for calculating confidence interval from cross-validation
def interval_confidence(values):
    return st.t.interval(confidence=0.95, df=len(values) - 1, loc=np.mean(values), scale=st.sem(values))


def cross_val(model, x_train, y_train, scorer):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(model, x_train, y_train, cv=kf, scoring=scorer)
    std = scores.std()
    mean_score = scores.mean()
    ic = interval_confidence(scores)
    return std, mean_score, ic, scores


def compute_metrics(model, x_train, y_train):
    precision_scorer_not_disease = make_scorer(precision_score, pos_label=0, average='binary')
    precision_scorer_disease = make_scorer(precision_score, pos_label=1, average='binary')

    recall_scorer_not_disease = make_scorer(recall_score, pos_label=0, average='binary')
    recall_scorer_disease = make_scorer(recall_score, pos_label=1, average='binary')

    f1_scorer_not_disease = make_scorer(f1_score, pos_label=0, average='binary')
    f1_scorer_disease = make_scorer(f1_score, pos_label=1, average='binary')

    precision_not_disease_std, precision_not_disease_mean, precision_ic_not_disease, precision_not_disease_values = \
        cross_val(model, x_train, y_train, precision_scorer_not_disease)
    precision_disease_std, precision_disease_mean, precision_ic_disease, precision_disease_values = \
        cross_val(model, x_train, y_train, precision_scorer_disease)

    recall_not_disease_std, recall_not_disease_mean, recall_ic_not_disease, recall_not_disease_values = \
        cross_val(model, x_train, y_train, recall_scorer_not_disease)
    recall_disease_std, recall_disease_mean, recall_ic_disease, recall_disease_values = \
        cross_val(model, x_train, y_train, recall_scorer_disease)

    f1_not_disease_std, f1_not_disease_mean, f1_ic_not_disease, f1_not_disease_values = \
        cross_val(model, x_train, y_train, f1_scorer_not_disease)
    f1_disease_std, f1_disease_mean, f1_ic_disease, f1_disease_values = \
        cross_val(model, x_train, y_train, f1_scorer_disease)

    metrics = {
        'precision_not_disease_values': precision_not_disease_values.tolist(),
        'precision_not_disease_std': precision_not_disease_std,
        'precision_not_disease_mean': precision_not_disease_mean,
        'precision_not_disease_ic': precision_ic_not_disease,
        'precision_disease_values': precision_disease_values.tolist(),
        'precision_disease_std': precision_disease_std,
        'precision_disease_mean': precision_disease_mean,
        'precision_disease_ic': precision_ic_disease,
        'recall_not_disease_values': recall_not_disease_values.tolist(),
        'recall_not_disease_std': recall_not_disease_std,
        'recall_not_disease_mean': recall_not_disease_mean,
        'recall_not_disease_ic': recall_ic_not_disease,
        'recall_disease_values': recall_disease_values.tolist(),
        'recall_disease_std': recall_disease_std,
        'recall_disease_mean': recall_disease_mean,
        'recall_disease_ic': recall_ic_disease,
        'f1_not_disease_values': f1_not_disease_values.tolist(),
        'f1_not_disease_std': f1_not_disease_std,
        'f1_not_disease_mean': f1_not_disease_mean,
        'f1_not_disease_ic': f1_ic_not_disease,
        'f1_disease_values': f1_disease_values.tolist(),
        'f1_disease_std': f1_disease_std,
        'f1_disease_mean': f1_disease_mean,
        'f1_disease_ic': f1_ic_disease
    }

    return metrics


def fit_and_evaluate(model, x_train, x_test, y_train, y_test, feature_names):
    model.fit(x_train, y_train)

    test_score = model.score(x_test, y_test)
    print(f"Test score {model.__class__.__name__}", test_score)
    metrics = compute_metrics(model, x_train, y_train)
    y_pred = model.predict(x_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # accuracy = accuracy_score(y_test, y_pred)

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

    results = {
        'model_name': model.__class__.__name__,
        'test_score': test_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    results.update(metrics)

    results_json = json.dumps(results)
    return results_json


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
        return fit_and_evaluate(dtc_model, x_train, x_test, y_train, y_test, feature_names)


def decision_tree():
    # open train and test sets
    with open('heart_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)
        dtc_model = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=6,
            min_samples_leaf=13,
            min_samples_split=2,
            random_state=0
        )
        return fit_and_evaluate(dtc_model, x_train, x_test, y_train, y_test, feature_names)


def random_forest_grid_search():
    # open train and test sets
    with open('heart_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        # Define the hyperparameter grid
        param_dist = {
            'n_estimators': [10, 50, 100, 150, 200, 250, 300],
            'max_features': ['log2', 'sqrt'],
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': range(2, 10),
            'min_samples_split': range(2, 10),
        }

        # Perform grid search with 5-fold cross-validation
        grid_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=0),
            param_distributions=param_dist,
            cv=5,
            n_iter=100,
            n_jobs=-1
        )
        grid_search.fit(x_train, y_train)

        # Print the best hyperparameters and corresponding accuracy score
        print(f"Best parameters for RandomFlorest: {grid_search.best_params_}")
        print(f"Best accuracy score for RandomFlorest: {grid_search.best_score_}")

        rfc_model = RandomForestClassifier(**grid_search.best_params_, random_state=0)
        results = fit_and_evaluate(rfc_model, x_train, x_test, y_train, y_test, feature_names)

        importances = rfc_model.feature_importances_

        # Sort the features by importance in descending order
        indices = importances.argsort()[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for i in range(x_train.shape[1]):
            print("%d. feature %s (%f)" % (i + 1, feature_names[indices[i]], importances[indices[i]]))

        return results


def random_forest():
    # open train and test sets
    with open('heart_train_test.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)
        rfc_model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=2,
            min_samples_leaf=4,
            max_features='log2',
            criterion='entropy',
            random_state=0
        )

        results = fit_and_evaluate(rfc_model, x_train, x_test, y_train, y_test, feature_names)

        importances = rfc_model.feature_importances_

        # Sort the features by importance in descending order
        indices = importances.argsort()[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for i in range(x_train.shape[1]):
            print("%d. feature %s (%f)" % (i + 1, feature_names[indices[i]], importances[indices[i]]))

        return results


def neural_network_grid_search():
    with open('heart_train_test_normalized.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        # Define the hyperparameter grid
        param_dist = {
            'solver': ['lbfgs', 'adam', 'sgd'],
            'hidden_layer_sizes': [(5, 2), (10, 5), (20, 10), (30, 15)],
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
            'max_iter': range(2000, 20000),
            'activation': ['tanh', 'relu', 'logistic'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
        }

        # Perform grid search with 5-fold cross-validation
        grid_search = RandomizedSearchCV(
            MLPClassifier(random_state=0),
            param_distributions=param_dist,
            cv=5,
            n_jobs=-1,
            n_iter=150,
            scoring='accuracy'
        )

        # Fit the grid search object to the training data
        grid_search.fit(x_train, y_train)

        # Print the best hyperparameters and corresponding accuracy score
        print("Best hyperparameters for MLPClassifier:", grid_search.best_params_)
        print("Best accuracy score for MLPClassifier:", grid_search.best_score_)

        mlp_model = MLPClassifier(**grid_search.best_params_, random_state=0)
        return fit_and_evaluate(mlp_model, x_train, x_test, y_train, y_test, feature_names)


def neural_network():
    with open('heart_train_test_normalized.pkl', 'rb') as f:
        feature_names, x_train, x_test, y_train, y_test = pickle.load(f)

        mlp_model = MLPClassifier(
            solver='adam',
            max_iter=10995,
            learning_rate='adaptive',
            hidden_layer_sizes=(10, 5),
            alpha=0.0001,
            activation='logistic',
            random_state=0
        )
        return fit_and_evaluate(mlp_model, x_train, x_test, y_train, y_test, feature_names)


def test_t_metrics(tree_results, forest_results, neural_results):
    # Substitua esses valores pelas suas métricas
    tree_results = json.loads(tree_results)
    forest_results = json.loads(forest_results)
    neural_results = json.loads(neural_results)

    precision_disease_nn = neural_results['precision_disease_values']
    precision_disease_rf = forest_results['precision_disease_values']

    # Realizando o teste t
    t_statistic, p_value = stats.ttest_ind(precision_disease_nn, precision_disease_rf)

    # Imprimindo os resultados
    print("Teste t para a precisão na classe 'disease' NN e RF:")
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)

    precision_not_disease_nn = neural_results['precision_not_disease_values']
    precision_not_disease_rf = forest_results['precision_not_disease_values']

    # Realizando o teste t
    t_statistic, p_value = stats.ttest_ind(precision_not_disease_nn, precision_not_disease_rf)

    # Imprimindo os resultados
    print("Teste t para a precisão na classe 'not_disease' NN e RF:")
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)


def create_graphic(tree_results, forest_results, neural_results):
    # Substitua esses valores pelas suas métricas
    tree_results = json.loads(tree_results)
    forest_results = json.loads(forest_results)
    neural_results = json.loads(neural_results)

    NN = [neural_results['precision'], neural_results['recall'], neural_results['f1_score']]  # NeuralNetwork
    RF = [forest_results['precision'], forest_results['recall'], forest_results['f1_score']]  # Random Forest
    DT = [tree_results['precision'], tree_results['recall'], tree_results['f1_score']]  # DecisionTree

    # Crie uma lista com os nomes das métricas
    labels = ['Precision', 'Recall', 'F1 Score']

    # Configurando a posição das barras no eixo X
    barWidth = 0.25
    r1 = np.arange(len(NN))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Criando as barras
    bar1 = plt.bar(r1, NN, color='b', width=barWidth, edgecolor='grey', label='NeuralNetwork')
    bar2 = plt.bar(r2, RF, color='g', width=barWidth, edgecolor='grey', label='RandomForest')
    bar3 = plt.bar(r3, DT, color='r', width=barWidth, edgecolor='grey', label='DecisionTree')

    # Função para adicionar valor em cima da barra
    def add_values_on_bars(bars):
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 3), ha='center', va='bottom')

    # Adicionar valores nas barras
    add_values_on_bars(bar1)
    add_values_on_bars(bar2)
    add_values_on_bars(bar3)

    # Adicionando os nomes para o eixo X
    plt.xlabel('Métricas', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(NN))], labels)

    # Criando a legenda do gráfico
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)

    plt.savefig('graphic_results.png')
    # Exibindo o gráfico
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv("heart.csv", delimiter=",")
    check(df)
    # preprocessing_train_test(df)
    # tree_results = decision_tree_grid_search()
    tree_results = decision_tree()
    # forest_results = random_forest_grid_search()
    forest_results = random_forest()
    # neural_results = neural_network_grid_search()
    neural_results = neural_network()
    create_graphic(tree_results, forest_results, neural_results)
    test_t_metrics(tree_results, forest_results, neural_results)

    results = [tree_results, forest_results, neural_results]

    # Open a file for writing
    with open('results.json', 'w') as f:
        # Use json.dump to write the results list to a file
        json.dump(results, f, indent=4)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
