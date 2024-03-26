'''
Author : Arnaud TECHER 
Date : 25/03/2024
Goal : Practice of machine learning on a dataset from Kaggle - The Boston Housing Dataset
'''

import pandas as pd
import matplotlib.pyplot as plt

class DATASET_LOADER:
    """
    Object that contain the dataset in a DataFrame format
    """
        
    def __init__(self,fileName,plotDataFlag=True):
        """Instance the class to load the dataset in a dataframe

        Args:
            fileName (str): Name of the file that contains the data
        """
        self.dataset = self.read_dataset(fileName)
        if plotDataFlag:
            self.plot_data_flag()

    def __str__(self) -> str:
        """Representation of the loader with a str

        Returns:
            str: Description of the dataset and the size associated
        """
        return f"Boston Housing Dataset with {self.dataset.shape[0]} samples and {self.dataset.shape[1]} features"
    
    def read_dataset(self,fileName):
        """Read the fileName and arrange the dataset in a DataFrame

        Args:
            fileName (str): Name of the file that contains the data

        Returns:
            DataFrame: pandas DataFrame that contains all the data
        """
        dataset = pd.read_csv(fileName)
        return dataset
    
    def plot_data_flag(self):
        """Plots of the target features in function of the different features in the dataset.
            Save the plot in the same folder that the script
        Returns:
            None:
        """
        # get the number of features in the dataset 
        nbr_features = int(self.dataset.shape[1])
        # create 2 lines of subplots
        fig,axs = plt.subplots(2,int(nbr_features/2),sharey=True)
        count = 0
        for i in range(2):
            for j in range(int(nbr_features/2)):
                axs[i,j].scatter(self.dataset.iloc[:,count],self.dataset.loc[:,"MEDV"])
                axs[i,j].set_xlabel(list(self.dataset.columns)[count])
                count+=1
        # increase the size of the fig so it s more readable
        fig.set_figwidth(10)
        fig.set_figheight(10)
        # y label common for all plots
        fig.supylabel("MEDV")
        fig.suptitle("Median value of owner-occupied homes in $1000s vs features.")
        plt.savefig("plots_boston_princing_dataset.png",dpi=300.)
        plt.show()
        return None
            
if __name__ == '__main__':
        """VALIDATION OF THE MODULE
        """
    fileName = "Boston.csv"
    data_loader = DATASET_LOADER(fileName)
    print(data_loader)
    dataset = data_loader.dataset
    print(dataset.describe())
