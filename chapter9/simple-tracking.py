import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from random import random, randint
from mlflow import log_metric, log_param, log_params, log_artifacts

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "simple-tracking-experiment"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else None
    if experiment_id is None:
        print("INFO: '{}' does not exist. Creating a new experiment".format(experiment_name))
        experiment_id = mlflow.create_experiment(experiment_name)

    print("starting a run with experiment_id {}".format(experiment_id))
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log a parameter (key-value pair)
        log_param("mlflow", "is cool")
        log_param("mlflow-version", mlflow.version.VERSION)

        params = {"learning_rate": 0.01, "n_estimators": 10}
        log_params(params)

        # Log a metric; metrics can be updated throughout the run
        log_metric("metric-1", random())
        for x in range(1,11):        
            log_metric("metric-1", random() + x)
        

        # Log an artifact (output file)
        if os.path.exists("images"):
            log_artifacts("images")
            print("done logging artifact")
        else:
            print("images directory does not exists")

        image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        mlflow.log_image(image, "random-image.png")

        fig, ax = plt.subplots()
        ax.plot([0, 2], [2, 5])
        mlflow.log_figure(fig, "figure.png")
    
    
    experiment = mlflow.get_experiment(experiment_id)
    print("Done tracking on run")
    print("experiment_id: {}".format(experiment.experiment_id))
    print("run_id: {}".format(run.info.run_id))