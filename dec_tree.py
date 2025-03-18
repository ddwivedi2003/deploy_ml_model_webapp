import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def dt(file):
# Read the CSV file
    read = pd.read_csv(file)

    # Print column names to identify numerical columns
    print(read.columns)

    # Separate the target column
    target_column = 'Drug'
    y = read[target_column]

    # Drop the target column from the features
    X = read.drop(target_column, axis=1)

    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Fill missing values with the mean for numerical columns
    X.fillna(X.mean(), inplace=True)

    # Fill missing values with the mode for categorical columns
    X.fillna(X.mode().iloc[0], inplace=True)

    # Replace 'num_col1' and 'num_col2' with actual numerical column names
    numerical_columns = ['Age', 'Na_to_K']  # Example numerical columns

    # Standardize numerical columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Import necessary libraries for decision tree and visualization
    import matplotlib.pyplot as plt

    # Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Plot the decision tree
    plt.figure(figsize=(20,10))
    tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_, rounded=True)
    plt.show()

    y_pred = clf.predict(X_test)

    # Print accuracy
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("Confusion Matrix:")
    # Print confusion matrix
    print(confusion_matrix(y_test, y_pred))


    plt.figure(figsize=(20,10))
    tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_, rounded=True)
    plt.savefig('decision_tree.pdf', format='pdf')
    plt.close()