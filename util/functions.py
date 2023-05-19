def train(model):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report,  ConfusionMatrixDisplay
    from matplotlib.pylab import rcParams

    dataframes = {
        'mean': pd.read_csv('imputed/mean.csv'),
        'median': pd.read_csv('imputed/median.csv'),
        'mode': pd.read_csv('imputed/mode.csv'),
        'KNN': pd.read_csv('imputed/KNN.csv'),
        'MICE': pd.read_csv('imputed/MICE.csv'),
        'iterative': pd.read_csv('imputed/iterative.csv'),
    }

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    from sklearn.model_selection import learning_curve

    models = {}
    metrics = {
        'accuracies': {},
        'precisions': {},
        'recalls': {},
        'f1_scores': {},
    }
    confusion_matrices = {}
    learning_curves = {}
    # feature_importances = {}

    for (name, dataframe) in dataframes.items():
        print(f'learning with {name} imputed data')
        y = np.asarray(dataframe['class'])
        X = np.asarray(dataframe.drop(columns=['class']))

        # split the dataset to train and test sets. set the test set size to 20%.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, )

        # Train Decision Tree Classifer
        model = model.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = model.predict(X_test)

        models[name] = model
        metrics['accuracies'][name] = model.score(X_test, y_test)
        metrics['precisions'][name] = precision_score(
            y_test, y_pred, average='weighted', )
        metrics['recalls'][name] = recall_score(
            y_test, y_pred, average='weighted')
        metrics['f1_scores'][name] = f1_score(
            y_test, y_pred, average='weighted')
        confusion_matrices[name] = confusion_matrix(
            y_test, y_pred, labels=model.classes_)
        learning_curves[name] = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
        # feature_importances[name] = pd.DataFrame(model.feature_importances_, index=dataframe.drop(columns=['class']).columns, columns=["Importance"])

    return {
        'models': models,
        'metrics': metrics,
        'confusion_matrices': confusion_matrices,
        'learning_curves': learning_curves,
    }


def plot_metrics(metrics):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    for i, (name, metric) in enumerate(metrics.items()):
        axs.flat[i].bar(metric.keys(), metric.values())
        axs.flat[i].set_ylim(0.7, 1)
        axs.flat[i].set_xlabel('Imputation Method')
        axs.flat[i].set_ylabel(name)
        axs.flat[i].set_title(f'{name}')

    fig.tight_layout()
    plt.show()


def plot_confusion_matrices(confusion_matrices):
    import matplotlib.pyplot as plt
    import numpy as np
    # plot all confusion matrices as subplots
    num_matrices = len(confusion_matrices.items())
    matrix_size = 2

    # Set up the figure and subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # Iterate over each confusion matrix and plot it as a heatmap in a subplot
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        im = axs.flat[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axs.flat[i].set_title(f'Confusion Matrix {name}')
        axs.flat[i].set_xticks(np.arange(matrix_size))
        axs.flat[i].set_yticks(np.arange(matrix_size))
        axs.flat[i].set_xlabel('Predicted Label')
        axs.flat[i].set_ylabel('True Label')

        # Add text annotations to each cell
        for j in range(matrix_size):
            for k in range(matrix_size):
                axs.flat[i].text(k, j, str(cm[j, k]), ha='center', va='center',
                                 color='white' if cm[j, k] > cm.max() / 2 else 'black')

    # Add a colorbar and adjust the layout
    fig.tight_layout()

    # Show the plot
    plt.show()


def plot_learning_curves(learning_curves):
    import matplotlib.pyplot as plt
    # plot all learning curves as subplots
    num_curves = len(learning_curves.items())

    # Set up the figure and subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # Iterate over each confusion matrix and plot it as a heatmap in a subplot
    for i, (name, curve) in enumerate(learning_curves.items()):
        # curve is this: learning_curve(model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
        train_sizes = curve[0]
        train_scores = curve[1]
        test_scores = curve[2]

        # plot the train and test scores
        axs.flat[i].grid()
        axs.flat[i].plot(train_sizes, train_scores.mean(
            axis=1), 'o-', color="r", label="Training score")
        axs.flat[i].plot(train_sizes, test_scores.mean(
            axis=1), 'o-', color="g", label="Cross-validation score")
        axs.flat[i].legend(loc="best")
        axs.flat[i].set_title(f'Learning Curve {name}')
        axs.flat[i].set_xlabel('Training examples')
        axs.flat[i].set_ylabel('Score')

    # Add a colorbar and adjust the layout
    fig.tight_layout()

    # Show the plot
    plt.show()

# feature importances is a dict of dataframes


def plot_feature_importances(feature_importances: dict):
    import matplotlib.pyplot as plt

    color_palette = plt.cm.Set3

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    for i, (name, importance) in enumerate(feature_importances.items()):
        importance = importance.sort_values(by='Importance', ascending=False)
        importance = importance.head(10)

        importance.plot.bar(ax=axs.flat[i])
        axs.flat[i].set_title(f'Feature Importance {name}')
        axs.flat[i].set_xlabel('Feature')
        axs.flat[i].set_ylabel('Importance')

    fig.tight_layout()
    plt.show()
