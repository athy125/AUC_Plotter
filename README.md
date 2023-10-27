# AUC Plotter

 This project focuses on visualizing the performance of machine learning classifiers by plotting ROC (Receiver Operating Characteristic) curves and calculating the AUC (Area Under the Curve) scores.

## Getting Started

### Prerequisites

Make sure you have the required Python packages installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

### Usage

To run this project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/athy125/AUC_Plotter.git
cd auc-plotting-project
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Execute the project:

```bash
python main.py --dataset path_to_your_dataset.csv
```

Replace `path_to_your_dataset.csv` with the path to your dataset file. You can also specify additional parameters like `n_estimators`, `max_depth`, and `min_samples_split` to fine-tune the Random Forest classifier.

## Project Structure

The project is organized as follows:

- `main.py`: The main entry point of the project that orchestrates the execution.
- `data_preparation.py`: Contains functions for loading and preprocessing the dataset.
- `classifier.py`: Includes the `train_classifier` function for training a Random Forest classifier.
- `evaluation.py`: Defines the `calculate_roc_auc` function for calculating ROC curves and AUC.
- `plotting.py`: Contains the `plot_roc_curve` function to visualize ROC curves and AUC.
- `requirements.txt`: Lists the required Python packages and their versions.

## Features

This project supports features like:

- Handling binary and multiclass classification scenarios.
- Customizing Random Forest hyperparameters.
- Visualizing multiple ROC curves with AUC values.

## Contributing

We welcome contributions to this project! Feel free to open issues, suggest improvements, or submit pull requests.

## Acknowledgments

- Thanks to the open-source community for valuable tools and libraries used in this project.
