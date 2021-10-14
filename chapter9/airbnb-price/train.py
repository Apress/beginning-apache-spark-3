import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import mlflow
import mlflow.sklearn
from  mlflow.tracking import MlflowClient


def parse_args():
    parser = argparse.ArgumentParser(description="AirBnb Price example")
    
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
    return parser.parse_args()

def train_model(data_file="./airbnb-sf-cleaned.csv"):
	print("Predicting AirBnb price with data file: {}".format(data_file))
	    # parsing command line arguments
	args = parse_args()

	mlflow.set_tracking_uri("http://localhost:5000")	
	
	raw_airbnb_df = pd.read_csv(data_file)

	training_df = raw_airbnb_df[["bedrooms", "beds", "bathrooms"]]
	label_df = raw_airbnb_df[["price"]]
	X_train, X_test, y_train, y_test = train_test_split(training_df, label_df.values.ravel(), random_state=31)

	params = {
		'n_estimators': args.n_estimators, 
		'max_depth': args.max_depth,
		'random_state' : 31
	}

    # enable logging
	mlflow.autolog()

	with mlflow.start_run() as run:
		mlflow.log_param("MLflow version", mlflow.version.VERSION)

		# model training
		print("training with param: {}".format(params))
		lr = RandomForestRegressor(**params)
		lr.fit(X_train, y_train)

		#mlflow.sklearn.log_model(lr, "AirBnb-RandomForest")

		predictions = lr.predict(X_test)

		# evaluation
		mse = metrics.mean_squared_error(y_test, predictions)
		rmse = np.sqrt(mse)
		mae = metrics.mean_absolute_error(y_test, predictions)
		r2 = metrics.r2_score(y_test, predictions)

		print("-" * 100)
		

		print("Done training model")
		print("Run_id: {}".format(run.info.run_id))

		#experiment = mlflow.get_experiment(run.info.experiment_id)
		#print("Experiment name: {}".format(experiment.name))
		#print("Experiment id: {}".format(run.info.experiment_id))

		print("  mse: {}".format(mse))
		print(" rmse: {}".format(rmse))
		print("  mae: {}".format(mae))
		print("  R2 : {}".format(r2))

		return (run.info.experiment_id, run.info.run_id)


if __name__ == "__main__":

	print("mlflow version: " , mlflow.version.VERSION)


	train_model()