import os
import warnings
import sys
import click

import mlflow
import mlflow.sklearn

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    # setting up MLflow related information
	mlflow.set_tracking_uri("http://localhost:5000")
	experiment_name = "auto-logging-boston"
	experiment = mlflow.get_experiment_by_name(experiment_name)
	experiment_id = experiment.experiment_id if experiment else None

	if experiment_id is None:
		print("INFO: '{}' does not exist. Creating a new experiment".format(experiment_name))
		experiment_id = mlflow.create_experiment(experiment_name)

	print("starting a run with experiment_id {}".format(experiment_id))

	# ===============================
	# Load data
	# ===============================
	boston = datasets.load_boston()
	X, y = shuffle(boston.data, boston.target, random_state=13)
	X = X.astype(np.float32)
	offset = int(X.shape[0] * 0.9)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

    # enable logging
	mlflow.autolog()
	with mlflow.start_run(experiment_id=experiment_id) as run:
		mlflow.log_param("MLflow version", mlflow.version.VERSION)
		
		params = {'n_estimators': 10, 'max_depth': 4, 'min_samples_split': 2,
				  'learning_rate': 0.01, 'loss': 'ls'}		

		gbr = ensemble.GradientBoostingRegressor(**params)

		gbr.fit(X_train, y_train)

		y_pred = gbr.predict(X_test)

		# calculate error metrics
		mae = metrics.mean_absolute_error(y_test, y_pred)
		mse = metrics.mean_squared_error(y_test, y_pred)
		rsme = np.sqrt(mse)
		r2 = metrics.r2_score(y_test, y_pred)

		experiment = mlflow.get_experiment(experiment_id)
		print("Done training model")
		print("experiment_id: {}".format(experiment.experiment_id))
		print("run_id: {}".format(run.info.run_id))

