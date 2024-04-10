from dataset_loader import DATASET_LOADER
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score,PredictionErrorDisplay
from sklearn.model_selection import learning_curve,validation_curve,LearningCurveDisplay,ValidationCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time

class PREPROCESSING_HOUSE_PRICING:
    """Object to preprocess the dataset before training and algorithm
    """

    def __init__(self):
        """Instanciate the preprocessing of the dataset
        """
        self.dataset = DATASET_LOADER(fileName="Boston.csv",plotDataFlag=False).dataset

        # if missing value in the dataframe, send a message.
        if all([i == 0 for i in self.check_missing_values()]):
            print("There is no missing value in the dataset \n")
        else: 
            print("There is a missing value in the dataset \n")
            print(self.check_missing_values())

        # if duplicated sample in the dataframe, send a message and supress the row 
        if all([boolean == False for boolean in self.check_duplicated_sample()]):
            print("There is no duplicated lines in the dataset.")
        else:
            print("There are duplicated samples in the dataset. They will be supress.")
            self.dataset.drop_duplicates(inplace=True)

    def get_nbr_samples(self):
        """Get the number of samples from the dataset

        Returns:
            int: number of samples in the dataset
        """
        return int(self.dataset.shape[0])
    
    def get_nbr_features(self):
        """Get the number of feature from the dataset. The features does not include the target one.

        Returns:
            int : number of feature in the dataset - does not include the target value.
        """
        return int(self.dataset.shape[1]-1)
    
    def check_missing_values(self):
        """Check for missing values in the dataset

        Returns:
            pandas.Series: Series containig the number of Nan values per rows of the dataset
        """
        return self.dataset.isnull().sum()
    
    def check_duplicated_sample(self):
        """Check if there are duplicated samples in the dataset.

        Returns:
            pandas.Series [bool]: Series of bool False if the rows is not duplicated and True otherwise
        """
        return self.dataset.duplicated()
    
    def generate_training_test_data(self,test_size=0.1):
        """Generate training and test subset based on sklearn library

        Args:
            test_size (float, optional): Percent of the sample . Defaults to 0.1.

        Returns:
            list : Splitting of the dataset into training data and test data
            X_train: DataFrame containing a subset of the samples for all the features to train a model 
            X_test : DataFrame containing a subset of the samples for all features to test a model
            y_train : DataFrame containing a subset of the samples for the output data to train the model
            y_test : DataFrame containing a subset of the samples for the output data to test the model
        """
        X_train,X_test,y_train,y_test = train_test_split(self.dataset.iloc[:,:-1],self.dataset.iloc[:,-1],test_size=test_size)
        return X_train,X_test,y_train,y_test

    
class MODEL_HOUSE_PRICING:
    """Object to model the Boston House pricing
    """
    # mapping of the type of regressor use with a name to identify the plots saved.
    MAP_REGRESSOR_PLOTNAME = {LinearRegression:"LinearRegression",
                              GradientBoostingRegressor:"GradientBoostingRegressor",
                              RandomForestRegressor:"RandomForestRegressor"}
    
    
    def __init__(self,X_train,X_test,y_train,y_test):
        """Instanciate the MODEL_HOUSE_PRICING class.

        Args:
            nbr_test_data (int, optional): number of samples use as test data. The other samples will be use as training data. Defaults to 5.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.nbr_features = X_train.shape[1]
    
    def linear_model(self):
        """Create a multi linear modelisation and train the model on the training dataset.

        Returns:
            sklearn.linear_model._base.LinearRegression: Linear regression model fitted on training dataset.
        """
        reg = LinearRegression()
        reg.fit(self.X_train,self.y_train)
        return reg
    
    def gradient_boost_model(self):
        """Create a Gradient Boosting regression - Method based on multiples decisions trees.

        Returns:
            sklearn.ensemble._gb.GradientBoostingRegressor: Gradient Boosting Regressor fitted on the training dataset.
        """
        reg = GradientBoostingRegressor(loss="squared_error",learning_rate=0.1,n_estimators=100,random_state=0)
        reg.fit(self.X_train,self.y_train)
        return reg
    
    def random_forest_model(self):
        """Create a Random Forest regression - Method based on multiples random decisions trees.

        Returns:
            sklearn.ensemble.RandomForestRegressor: Random Forest Regressor fitted on the training dataset.
        """
        reg = RandomForestRegressor(random_state=0)
        reg.fit(self.X_train,self.y_train)
        return reg 
    
    def predict_house_pricing(self,model):
        """Predict the Boston House Pricing based on a sklear model.

        Args:
            model (sklearn.linear_model._base.LinearRegression): sklearn model fitted on training dataset.

        Returns:
            numpy.ndarray : array contained the prediction of the target values for the test dataset.
        """
        medv_prediction = model.predict(self.X_test)
        return medv_prediction
    
    def plot_house_pricing(self,reg,medv_prediction):
        """Plot the target values and the predicted one for each features for the test dataset.

        Args:
            medv_prediction (numpy.ndarray): Prediction of the target value

        Returns:
            None: 
        """
        # create multiple subplots for all the features on 2 rows.
        fig,axs = plt.subplots(2,int(self.nbr_features/2),sharey=True)
        count = 0
        for i in range(2):
            for j in range(int(self.nbr_features/2)):
                axs[i,j].scatter(self.X_test.iloc[:,count],self.y_test[:],color='blue',label='real_value')
                axs[i,j].scatter(self.X_test.iloc[:,count],medv_prediction[:],color='red',label='prediction')
                axs[i,j].set_xlabel(list(self.X_test.columns)[count])
                count+=1
        # increase the size of the fig so it s more readable
        fig.set_figwidth(10)
        fig.set_figheight(10)
        # y label common for all plots
        fig.supylabel("MEDV")
        fig.suptitle("Prediction of Median value of owner-occupied homes in $1000s vs features.")
        # add a common legend for the subplots
        plt.figlegend(["real_values","prediction"])
        plt.savefig(f"plots_prediction_boston_pricing_dataset_{self.MAP_REGRESSOR_PLOTNAME[type(reg)]}.png",dpi=300.)
        plt.show()
        return None
          
    def plot_house_pricing_seaborn(self,reg,medv_prediction):
        """Plot the target values and the predicted one for each features for the test dataset.

        Args:
            medv_prediction (numpy.ndarray): Prediction of the target value

        Returns:
            None
        """
        # build a dataset to plot it with seaborn
        data_to_plot = pd.concat([self.X_test]*2)
        # add a columns legend for the multiple scatter plots
        data_to_plot["legend"] = ["real_values" for i in range(self.X_test.shape[0])] + ["prediction" for i in range(self.X_test.shape[0])]
        # concatenate the y values with the test data and the prediction values
        y_to_plot = self.y_test.tolist() + medv_prediction.tolist()
        # adding the y values to the dataframe
        data_to_plot["MEDV"] = y_to_plot
        features_to_scatter = [featureName for featureName in self.X_train.columns if featureName not in ["MEDV"]]
        sns.set_style(style="ticks")
        g = sns.pairplot(data=data_to_plot,x_vars=features_to_scatter,y_vars=["MEDV"],kind='scatter',height=2,hue="legend")
        plt.savefig(f"plot_test_data_vs_prediction_{self.MAP_REGRESSOR_PLOTNAME[type(reg)]}.png",dpi=300.)
        plt.show()
        return None

    def print_metrics_model(self,medv_prediction):
        """Print important metrics for a given prediction to evaluate the quality of the model.

        Args:
            medv_prediction (numpy.ndarray): Prediction of the target value
        Returns:
            None: 
        """
        mean_sq_error = mean_squared_error(self.y_test[:],medv_prediction)
        print(f"The mean squared error is {mean_sq_error:.2f} ")
        mean_abs_error = mean_absolute_error(self.y_test[:],medv_prediction)
        print(f"The abs mean squared error is : {mean_abs_error:.2f}")
        score = r2_score(self.y_test[:],medv_prediction)   
        print(f"Coefficient of determination r2 (1 mean perfect prediction) : {score:.2f}") 
        mean_percent_error = mean_absolute_percentage_error(self.y_test[:],medv_prediction)
        print(f"Mean percentage error : {mean_percent_error:.2f}")
        print(f"Accuracy of the model : {(1-mean_percent_error)*100:.2f}%")
        print('\n')
        return None

    def display_prediction(self,reg,medv_prediction):
        """Plot the prediction of the target versus the real values for the test dataset.

        Args:
            medv_prediction (numpy.ndarray): Prediction of the target value

        Returns:
            None
        """
        PredictionErrorDisplay.from_predictions(y_true=self.y_test,y_pred=medv_prediction)
        plt.savefig(f"prediction_vs_reals_values_{self.MAP_REGRESSOR_PLOTNAME[type(reg)]}.png",dpi=300.)
        plt.show()
        return None
    
    def get_learning_curve(self,reg):
        """Create a learning curve for a given regressor (score vs number of samples.)
        Plot informations about the learning rate in the console

        Args:
            reg (sklearn): Regressor fit on trained data such as sklearn.linear_model._base.LinearRegression,sklearn.ensemble._gb.GradientBoostingRegressor

        Returns:
            None
        """
        train_size_abs, train_scores, test_scores = learning_curve(reg, self.X_train, self.y_train) 
        for (train_size_abs,train_score,test_score) in zip(train_size_abs, train_scores, test_scores):
            print(f"Number of samples to train the model :{train_size_abs}")
            print(f"The train score obtained is {train_score}")
            print(f"The test score obtained is {test_score}")
        return train_size_abs, train_scores, test_scores
    
    def get_validation_curve(self,reg):
        """Create a validation curve for a given regressor and for a given hyperparameter.
        Plot informations about the validation curve in the console.

        Args:
            reg (sklearn): Regressor fit on trained data such as sklearn.linear_model._base.LinearRegression, sklearn.ensemble._gb.GradientBoostingRegressor

        Raises:
            NotImplemented: None
        """
        if type(reg) == LinearRegression:
            raise NotImplemented("To be implemented.")
        if type(reg) == GradientBoostingRegressor:
            paramName = "learning_rate"
            train_scores,valid_scores = validation_curve(reg,self.X_train,self.y_train,param_name=paramName,param_range=np.logspace(-6,-1,10))
            print(f"Validation of the parameter : {paramName}")
            print(f"Train scores : {train_scores}")
            print(f"Valid scores : {valid_scores}")
        return train_scores,valid_scores

    def plot_learning_curve(self,reg):
        """Plot the learning curve for a given regressor. 

        Args:
            reg (sklearn): Regressor fit on trained data such as sklearn.linear_model._base.LinearRegression or sklearn.ensemble._gb.GradientBoostingRegressor

        Returns:
            None
        """
        LearningCurveDisplay.from_estimator(reg,self.X_train,self.y_train)
        plt.savefig(f"learning_curve_{self.MAP_REGRESSOR_PLOTNAME[type(reg)]}.png",dpi=300.)
        plt.plot()
        return None
    
    def plot_validation_curve(self,reg):
        """Plot the validation curve for a given regressor for specific hyperparameters.

        Returns:
            None
        """
        if reg == LinearRegression:
            raise NotImplemented("To be implemented.")
        if type(reg) ==GradientBoostingRegressor:
            paramName = "learning_rate"
            ValidationCurveDisplay.from_estimator(reg,self.X_train,self.y_train,param_name=paramName,param_range=np.logspace(-6,-1,10))
            plt.savefig(f"Validation_curve_{paramName}_{self.MAP_REGRESSOR_PLOTNAME[type(reg)]}.png",dpi=300.)
            plt.show()
        return None
    
if __name__ == "__main__":
    """VALIDATION OF THE MODULE 
    """
    preprocess_data = PREPROCESSING_HOUSE_PRICING()
    X_train,X_test,y_train,y_test = preprocess_data.generate_training_test_data(test_size=10)
    model_house = MODEL_HOUSE_PRICING(X_train,X_test,y_train,y_test)
    
    # Map regressor model with a bool to apply a regressor or not
    flag_map_regressor = {"linear":True,"random_forest":True,"gradBoost":True}

    # Plot the learning curve for the linear regression model 
    if flag_map_regressor["linear"]:
        print("INFORMATION FOR THE LINEAR REGRESSION MODEL\n")
        linear_reg = model_house.linear_model()
        model_house.plot_learning_curve(linear_reg)
        prediction = model_house.predict_house_pricing(linear_reg)
        model_house.plot_house_pricing_seaborn(linear_reg,prediction)
        model_house.display_prediction(linear_reg,prediction)
        model_house.print_metrics_model(prediction)
        
    # Testing the Random Forest Regressor to model the Boston House Pricing
    # This algorithm should work well because we have structured data categorical or not
    if flag_map_regressor["random_forest"]:
        print("INFORMATION FOR THE RANDOM FOREST REGRESSOR MODEL\n")
        start = time.time()
        random_forest_reg = model_house.random_forest_model()
        end = time.time()
        print(f'Time to learn : {end-start} s')
        model_house.plot_learning_curve(random_forest_reg)
        prediction = model_house.predict_house_pricing(random_forest_reg)
        model_house.plot_house_pricing_seaborn(random_forest_reg,prediction)
        model_house.display_prediction(random_forest_reg,prediction)
        model_house.print_metrics_model(prediction)

    # Testing the Gradient Boosted Regressor to model the Boston House Pricing
    # This regressor should give similar results than Random Forest Regressor.
    # However the speed for the learning phase should be better with the Boosting.
    if flag_map_regressor["gradBoost"]:
        print("INFORMATION FOR THE GRADIENT BOOSTED REGRESSOR MODEL\n")
        start = time.time()
        gradBoost_reg = model_house.gradient_boost_model()
        end = time.time()
        print(f'Time to learn : {end-start} s')
        model_house.plot_validation_curve(gradBoost_reg)
        model_house.plot_learning_curve(gradBoost_reg)
        prediction = model_house.predict_house_pricing(gradBoost_reg)
        model_house.plot_house_pricing_seaborn(gradBoost_reg,prediction)
        model_house.display_prediction(gradBoost_reg,prediction)
        model_house.print_metrics_model(prediction)

   
    