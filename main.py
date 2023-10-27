import argparse
import logging
from data_preparation import load_data, split_data
from classifier import train_classifier
from evaluation import calculate_roc_auc
from plotting import plot_roc_curve

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Hand Gestures Detection with ROC Analysis")
    parser.add_argument("--dataset", type=str, default="breast_cancer.csv", help="Path to the dataset file")
    args = parser.parse_args()

    # Load and preprocess data
    logging.info("Loading and preprocessing the dataset...")
    X, y = load_data(args.dataset)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train a classifier
    logging.info("Training the classifier...")
    clf = train_classifier(X_train, y_train)

    # Calculate ROC curve and AUC
    logging.info("Calculating ROC curve and AUC...")
    fpr, tpr, roc_auc = calculate_roc_auc(clf, X_test, y_test)

    # Plot the ROC curve
    logging.info("Plotting the ROC curve...")
    plot_roc_curve(fpr, tpr, roc_auc)

if __name__ == "__main__":
    main()

