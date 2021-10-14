import os
import warnings
import sys
import click
import argparse

import mlflow
import mlflow.sklearn

import numpy as np


from sklearn import metrics
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

def parse_args():
    parser = argparse.ArgumentParser(description="Boston Housing Price example")
    
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="number of n_estimators (default: 100)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="tree depth (default: 4)",
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="min sample split (default: 2)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="learning rate to update step size at each boosting step (default: 0.01)",
    )
    return parser.parse_args()

def train_model():	

    # parsing command line arguments
	args = parse_args()

	mlflow.set_tracking_uri("http://localhost:5000")	

	# ===============================
	# Load data
	# ===============================
	boston = datasets.load_boston()
	X, y = shuffle(boston.data, boston.target, random_state=13)
	X = X.astype(np.float32)
	offset = int(X.shape[0] * 0.9)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

	with mlflow.start_run() as run:
		mlflow.log_param("MLflow version", mlflow.version.VERSION)

		params = {
			'n_estimators': args.n_estimators, 
			'max_depth': args.max_depth, 
			'min_samples_split': args.min_samples_split,
			'learning_rate': args.learning_rate, 
			'loss': 'ls'
		}
		mlflow.log_param('loss', params['loss'])

		gbr = ensemble.GradientBoostingRegressor(**params)

		gbr.fit(X_train, y_train)

		mlflow.sklearn.log_model(gbr, "GradientBoostingRegressor")

		y_pred = gbr.predict(X_test)

		# calculate error metrics
		mae = metrics.mean_absolute_error(y_test, y_pred)
		mse = metrics.mean_squared_error(y_test, y_pred)
		rsme = np.sqrt(mse)
		r2 = metrics.r2_score(y_test, y_pred)

		# Log metrics
		mlflow.log_metric("mae", mae)
		mlflow.log_metric("mse", mse)
		mlflow.log_metric("rsme", rsme)
		mlflow.log_metric("r2", r2)

		print("Done training model")
		print("Run_id: {}".format(run.info.run_id))

		experiment = mlflow.get_experiment(run.info.experiment_id)
		print("Experiment name: {}".format(experiment.name))
		print("Experiment id: {}".format(run.info.experiment_id))

		return (run.info.experiment_id, run.info.run_id)

if __name__ == "__main__":

	print("mlflow version: " , mlflow.version.VERSION)

	train_model()