import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def dt(file, target_column=None, numerical_columns=None, test_size=0.2, random_state=42):
    """
    Train a Decision Tree Classifier on the given dataset.

    Parameters:
        file (str): Path to the CSV file containing the dataset.
        target_column (str, optional): Name of the target column. If None, it will attempt to auto-detect.
        numerical_columns (list, optional): List of numerical columns to scale. Defaults to None.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        clf (DecisionTreeClassifier): Trained Decision Tree Classifier.
    """
    # Read the CSV file
    read = pd.read_csv(file)

    # Auto-detect the target column if not provided
    if target_column is None:
        # Assume the target column is the only non-numeric column
        non_numeric_columns = read.select_dtypes(exclude=['number']).columns
        if len(non_numeric_columns) == 1:
            target_column = non_numeric_columns[0]
        else:
            raise ValueError("Unable to auto-detect the target column. Please specify it explicitly.")

    # Separate features and target
    y = read[target_column]
    X = read.drop(target_column, axis=1)

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Handle missing values
    # Fill missing values in numerical columns with the mean
    numerical_cols = X.select_dtypes(include=['number']).columns
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())

    # Fill missing values in categorical columns with the mode
    categorical_cols = X.select_dtypes(exclude=['number']).columns
    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

    # Scale numerical columns if provided
    if numerical_columns:
        scaler = StandardScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    return clf