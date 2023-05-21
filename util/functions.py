import pandas as pd


def snap_dataframe(dataframe: pd.DataFrame):
    binary_columns = []
    if 'diabetes_mellitus' in dataframe.columns:
        binary_columns = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'hypertension',
                          'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia', ]
    else:
        binary_columns = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'hypertension',
                          'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia', ]
    zero_to_five = ['albumin', 'sugar']
    sg = ['specific_gravity']
    snapped_bin = dataframe[binary_columns].applymap(
        lambda x: 1 if x > 0.5 else 0)
    dataframe[binary_columns] = snapped_bin
    snapped_zero_to_five = dataframe[zero_to_five].applymap(
        lambda x: 0 if x < 0.5 else 1 if x < 1.5 else 2 if x < 2.5 else 3 if x < 3.5 else 4 if x < 4.5 else 5)
    dataframe[zero_to_five] = snapped_zero_to_five
    snapped_sg = dataframe[sg].applymap(lambda x: 1.005 if x < 1.0075 else 1.010 if x <
                                        1.0125 else 1.015 if x < 1.0175 else 1.020 if x < 1.0225 else 1.025)
    dataframe[sg] = snapped_sg
    return dataframe


def impute_and_train(dataframe, model_constructor, params={}):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from fancyimpute import IterativeImputer as MICE
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
    from sklearn.model_selection import learning_curve

    imputation_methods = {
        'mean': SimpleImputer(strategy='mean'),
        'median': SimpleImputer(strategy='median'),
        'mode': SimpleImputer(strategy='most_frequent'),
        'KNN': KNNImputer(n_neighbors=2),
        'MICE': MICE(),
        'iterative': IterativeImputer(random_state=0),
    }
    models = {method: [] for method in imputation_methods}
    metrics = {
        'accuracies': {method: [] for method in imputation_methods},
        'precisions': {method: [] for method in imputation_methods},
        'recalls': {method: [] for method in imputation_methods},
        'f1_scores': {method: [] for method in imputation_methods},
    }
    confusion_matrices = {method: [] for method in imputation_methods}

    learning_curves = {method: [] for method in imputation_methods}
    dataframes = {method: [] for method in imputation_methods}
    # feature_importances = {}

    for (name, imputer) in imputation_methods.items():
        print(f'learning with {name} imputed data')

        # Define the number of desired cross-validation iterations
        num_cv_iterations = 5

        X = np.asarray(dataframe.drop(columns=['class']))
        y = np.asarray(dataframe['class'])

        # Perform cross-validation iterations
        for i in range(num_cv_iterations):
            print('Cross-validation iteration {}/{}'.format(i + 1, num_cv_iterations))

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=i)

            # Perform imputation on the training set
            X_train_imputed = pd.DataFrame(X_train, columns=dataframe.drop(
                columns=['class']).columns
            )
            X_train_imputed[X_train_imputed.columns] = imputer.fit_transform(
                X_train)
            X_train_imputed = snap_dataframe(pd.DataFrame(X_train_imputed))

            # Drop NaN values from the test set
            X_test_dropped = X_test[~np.isnan(X_test).any(axis=1)]
            y_test_dropped = y_test[~np.isnan(X_test).any(axis=1)]

            # Train the model on the imputed training set
            model = model_constructor(**params)
            model.fit(X_train_imputed.values, y_train)

            # Make predictions on the dropped test set
            y_pred = model.predict(X_test_dropped)

            # Calculate the accuracy of the model

            dataframes[name].append(
                (X_train_imputed, y_train, X_test_dropped, y_test_dropped))
            models[name].append(model)
            metrics['accuracies'][name].append(
                accuracy_score(y_test_dropped, y_pred))
            metrics['precisions'][name].append(precision_score(
                y_test_dropped, y_pred, average='weighted', ))
            metrics['recalls'][name].append(recall_score(
                y_test_dropped, y_pred, average='weighted'))
            metrics['f1_scores'][name].append(f1_score(
                y_test_dropped, y_pred, average='weighted'))
            confusion_matrices[name].append(confusion_matrix(
                y_test_dropped, y_pred, labels=model.classes_))
            try:
                # learning curve
                learning_curves[name].append(learning_curve(
                    model, X_train_imputed, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50)))
            except:
                print('learning curve could not be created')

    return {
        'models': models,
        'metrics': metrics,
        'confusion_matrices': confusion_matrices,
        'learning_curves': learning_curves,
        'dataframes': dataframes,
    }


def plot_metrics(metrics):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    for i, (name, metric) in enumerate(metrics.items()):
        mean_arrays = {key: np.mean(arr) for key, arr in metric.items()}
        axs.flat[i].bar(mean_arrays.keys(), mean_arrays.values())
        axs.flat[i].set_ylim(0.7, 1)
        axs.flat[i].set_xlabel('Imputation Method')
        axs.flat[i].set_ylabel(name)
        axs.flat[i].set_title(f'{name}')

    fig.tight_layout()
    plt.show()


def plot_mean_confusion_matrices(confusion_matrices):
    import matplotlib.pyplot as plt
    import numpy as np
    # plot all confusion matrices as subplots
    num_matrices = len(confusion_matrices.items())
    matrix_size = 2

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
    for i, (name, matrices) in enumerate(confusion_matrices.items()):
        cm = np.mean(matrices, axis=0)

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
    import numpy as np
    # plot all learning curves as subplots

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    for i, (name, curves) in enumerate(learning_curves.items()):
        train_sizes = []
        train_scores = []
        test_scores = []
        for c in curves:
            train_sizes.append(c[0])
            train_scores.append(c[1])
            test_scores.append(c[2])
        train_sizes = np.mean(train_sizes, axis=0)
        train_scores = np.mean(train_scores, axis=0)
        test_scores = np.mean(test_scores, axis=0)

        # plot the train and test scores
        axs.flat[i].grid()
        axs.flat[i].plot(train_sizes, train_scores.mean(
            axis=1), '-', color="r", label="Training score")
        axs.flat[i].plot(train_sizes, test_scores.mean(
            axis=1), '-', color="g", label="Cross-validation score")
        axs.flat[i].legend(loc="best")
        axs.flat[i].set_title(f'Learning Curve {name}')
        axs.flat[i].set_xlabel('Training examples')
        axs.flat[i].set_ylabel('Score')

    # Add a colorbar and adjust the layout
    fig.tight_layout()

    # Show the plot
    plt.show()

# feature importances is a dict of dataframes


def plot_feature_importances(feature_importances: dict[str, pd.DataFrame]):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    bar_width = 0.1
    x = np.arange(len(feature_importances['mean']))

    for i, (name, importance) in enumerate(feature_importances.items()):
        plt.bar(x + i * bar_width,
                importance['Importance'], width=bar_width, label=name)

    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(x + bar_width * (len(feature_importances) - 1) /
               2, feature_importances['mean'].index,  rotation=90)
    plt.legend()
    plt.show()
