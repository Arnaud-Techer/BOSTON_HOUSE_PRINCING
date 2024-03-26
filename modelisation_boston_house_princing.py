from dataset_loader import DATASET_LOADER
from sklearn.linear_model  import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt

class MODEL_HOUSE_PRICING:

    def __init__(self,nbr_test_data=5):
        """Instanciate the MODEL_HOUSE_PRICING class.

        Args:
            nbr_test_data (int, optional): number of samples use as test data. The other samples will be use as training data. Defaults to 5.
        """
        self.dataset = DATASET_LOADER(fileName="Boston.csv",plotDataFlag=False).dataset
        self.nbr_samples = self.get_nbr_samples()
        self.nbr_learning_samples = self.nbr_samples-nbr_test_data
        self.nbr_features = self.get_nbr_features()
        self.training_data_subset = self.get_training_data_subset()
        self.training_target_subset = self.get_training_target_subset()
        self.test_data_subset = self.get_test_data_subset()
        self.test_target_subset = self.get_test_target_subset()

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
    
    def get_training_data_subset(self):
        """Get a subset of the dataset used to trained the model.

        Returns:
            DataFrame : subset of the dataset used for the training.
        """
        return self.dataset.iloc[:self.nbr_learning_samples,:self.nbr_features]

    def get_training_target_subset(self):
        """Get a subset of the target value dataset used to train the model

        Returns:
            DataFrame : subset of the target value dataset.
        """
        return self.dataset.iloc[:self.nbr_learning_samples,-1]

    def get_test_data_subset(self):
        """Get a subset of the dataset used to test the model

        Returns:
            DataFrame: subset of the data used to test the model
        """
        return self.dataset.iloc[self.nbr_learning_samples:,:self.nbr_features]
    
    def get_test_target_subset(self):
        """Get a subset of the target value dataset used to test the model

        Returns:
            DataFrame : subset of the target value which correspond to the test data
        """
        return self.dataset.iloc[self.nbr_learning_samples:,-1]
    
    def linear_model(self):
        """Create a multi linear modelisation and train the model on the training dataset.

        Returns:
            sklearn.linear_model._base.LinearRegression: Linear regression model fitted on training dataset.
        """
        reg = LinearRegression()
        reg.fit(self.training_data_subset,self.training_target_subset)
        return reg
    
    def predict_house_pricing(self,model):
        """Predict the Boston House Pricing based on a sklear model.

        Args:
            model (sklearn.linear_model._base.LinearRegression): sklearn model fitted on training dataset.

        Returns:
            numpy.ndarray : array contained the prediction of the target values for the test dataset.
        """
        medv_prediction = model.predict(self.test_data_subset)
        return medv_prediction
    
    def plot_house_pricing(self,medv_prediction):
        """Plot the target values and the predicted one for each features for the test dataset.

        Args:
            medv_prediction (numpy.ndarray): Prediction of the target value

        Returns:
            None: 
        """
        fig,axs = plt.subplots(2,int(self.nbr_features/2),sharey=True)
        count = 0
        for i in range(2):
            for j in range(int(self.nbr_features/2)):
                axs[i,j].scatter(self.test_data_subset.iloc[:,count],self.test_target_subset[:],color='blue',label='real_value')
                axs[i,j].scatter(self.test_data_subset.iloc[:,count],medv_prediction[:],color='red',label='prediction')
                axs[i,j].set_xlabel(list(self.test_data_subset.columns)[count])
                count+=1
        # increase the size of the fig so it s more readable
        fig.set_figwidth(10)
        fig.set_figheight(10)
        # y label common for all plots
        fig.supylabel("MEDV")
        fig.suptitle("Prediction of Median value of owner-occupied homes in $1000s vs features.")
        plt.savefig("plots_prediction_boston_princing_dataset.png",dpi=300.)
        plt.show()
        return None
    
    def print_metrics_model(self,medv_prediction):
        """Print important metrics for a given prediction to evaluate the quality of the model.

        Args:
            medv_prediction (numpy.ndarray): Prediction of the target value
        Returns:
            None: 
        """
        mean_sq_error = mean_squared_error(self.test_target_subset[:],medv_prediction)
        print(f"The mean squared error is {mean_sq_error} ")
        mean_abs_error = mean_absolute_error(self.test_target_subset[:],medv_prediction)
        print(f"The abs mean squared error is : {mean_abs_error}")
        score = r2_score(self.test_target_subset[:],medv_prediction)   
        print(f"Coefficient of determination r2 (1 mean perfect prediction) : {score}") 
        return None


if __name__ == "__main__":
    """VALIDATION OF THE MODULE 
    """
    model_house = MODEL_HOUSE_PRICING()
    reg = model_house.linear_model()
    prediction = model_house.predict_house_pricing(reg)
    model_house.plot_house_pricing(prediction)
    model_house.print_metrics_model(prediction)
    