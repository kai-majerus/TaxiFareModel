import mlflow
import joblib

from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from memoized_property import memoized_property

from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data
from TaxiFareModel.utils import compute_rmse

class Trainer():
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[GB] [London] [kai-majerus] TaxiFareModel 1.0"

    def __init__(self, X, y, model):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.model = model

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('model', self.model)
        ])
        return self

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test, cross_val=False):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        
        rmse = compute_rmse(y_pred, y_test)
        return rmse
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')


if __name__ == "__main__":
    # store the data in a DataFrame
    df = get_data()

    # clean the data
    df = clean_data(df)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # train and fit and evaluate
    for model in [LinearRegression(), RandomForestRegressor()]:
        trainer = Trainer(X_train, y_train, model)
        rmse = trainer.set_pipeline().run().evaluate(X_val, y_val)
        # create experiment and log model name and metric
        trainer.mlflow_log_metric("rmse", rmse)
        trainer.mlflow_log_param("model", model)

    # retrieve experiment ID
    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

    # save model best model
    # best_model = 
    # trainer.save_model()
