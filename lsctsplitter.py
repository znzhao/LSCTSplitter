import os
import pandas as pd
import numpy as np
import tqdm
import datetime
from scipy.optimize import minimize
from matplotlib import pyplot
from helper.utils import Timer

def loadYC():
    '''
    Load the yield curve data from the saved drive
    '''
    path = 'https://raw.githubusercontent.com/znzhao/LSCTSplitter/main/sample_data/ycg.csv'
    data = pd.read_csv(path, index_col = 'date')
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data

class LSCTSplitter(object):
    def __init__(self, lambdas = 1, maturities = [1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]):
        """
        Initializes the LSCTSplitter object.

        Args:
        - lambdas (float): Lambda parameter for the LSCT model.
        - maturities (list): List of maturities for the model.
        """
        self.lambdas = lambdas
        self.maturities = maturities
        loadL = np.ones(len(maturities)).tolist()
        loadS = [(1-np.exp(-self.lambdas*t)) / (self.lambdas*t) for t in self.maturities]
        loadC = [(1-np.exp(-self.lambdas*t)) / (self.lambdas*t) - np.exp(-self.lambdas*t) for t in self.maturities]
        loadT = [2*(1-np.exp(-self.lambdas*t)) / (self.lambdas*t) - np.exp(-self.lambdas*t)*(self.lambdas*t + 2) for t in self.maturities]
        
        self.loadings = np.array([loadL, loadS, loadC, loadT])

    def __error(self):
        """
        Calculates the error between data and the model's predictions.
        """
        err = self.data - pd.DataFrame(self.factors.values.dot(self.loadings), columns = self.data.columns, index = self.factors.index)
        self.errmean = np.nanmean(err, axis=0)
        self.errstd = np.nanstd(err, axis=0)
        return None

    def fit(self, data, display = True):
        """
        Fits the LSCT model to the provided data.

        Args:
        - data (pandas DataFrame): Input data.
        - display (bool): If True, display optimization progress.

        Returns:
        - self: Returns the LSCTSplitter object.
        """
        self.data = data
        if display:
            print('Run LSCT splitter optimization.')
        with Timer('LSCT Optimization', display = display):
            values = data.values
            self.factors = []
            for row in tqdm.tqdm(values, disable = not display):
                res = minimize(lambda x: self.__objfunc(x, row), x0 = np.zeros(4))
                self.factors.append(pd.DataFrame(res.x).T)
            self.factors = pd.concat(self.factors, axis = 0)
            self.factors.columns = ['level', 'slope', 'curvature', 'twist']
            self.factors.index = data.index
            self.__error()
        return self
    
    def __objfunc(self, factors, trueys):
        """
        Objective function used in optimization.

        Args:
        - factors (array): Model factors.
        - trueys (array): True values.

        Returns:
        - float: Sum of squared differences between true and predicted values.
        """
        factors = np.expand_dims(factors, axis=0)
        return np.nansum((trueys - self.__pred(factors))**2)

    def __pred(self, factors):
        """
        Predicts values using model factors.

        Args:
        - factors (array): Model factors.

        Returns:
        - array: Predicted values.
        """
        return factors.dot(self.loadings)

    def inverse(self, factors = None):
        """
        Performs inverse transformation using model factors.

        Args:
        - factors (pandas DataFrame): Model factors.

        Returns:
        - pandas DataFrame: Inverse transformed data.
        """
        if factors is None:
            factors = self.factors
        preds = pd.DataFrame(factors.values.dot(self.loadings), columns = self.data.columns, index = factors.index)
        for i in range(len(self.data.columns)):
            preds.iloc[:,i] = preds.iloc[:,i] + self.errmean[i]
        self.preds = preds
        return preds
    
    def invCov(self, cov):
        """
        Calculates the inverse covariance.

        Args:
        - cov (array): Covariance matrix.

        Returns:
        - pandas DataFrame: Inverse covariance matrix.
        """
        sig2 = self.loadings.T.dot(cov.dot(self.loadings)).diagonal()
        sig2 = np.expand_dims(sig2, axis = 0)
        return pd.DataFrame(sig2, columns = self.data.columns)
        
    def plotfactors(self, *args, **kwargs):
        """
        Plots the factors.

        Args:
        - *args, **kwargs: Additional arguments for plotting.
        """
        if 'end' not in kwargs.keys():
            end = datetime.datetime.now()
        else:
            end = kwargs['end']

        if 'start' not in kwargs.keys():
            start = end - datetime.timedelta(weeks=52*10)
        else:
            start = kwargs['start']
        
        data_to_plot = self.factors[(self.factors.index >= start) & (self.factors.index <= end)]

        fig = pyplot.figure(figsize=(16,9))
        axfactors  = []
        axloadings = []
        titles = ['level', 'slope', 'curvature', 'twist']
        for i in range(4):
            axfactors.append(pyplot.subplot2grid((4,3), (i,0), colspan = 2))
            axloadings.append(pyplot.subplot2grid((4,3), (i,2)))
            axfactors[i].plot(data_to_plot.index, data_to_plot.iloc[:,i].values)
            axfactors[i].set_title(titles[i])
            [axfactors[i].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
            axloadings[i].plot(self.maturities, self.loadings[i], marker = 'o')
            axloadings[i].set_title(titles[i] +' loading')
            [axloadings[i].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        fig.tight_layout()
        pyplot.show()

    def plot(self, columns = [[0, 2, 4], [5, 8, 10]], resid = False, *args, **kwargs):
        """
        Plots the data.

        Args:
        - columns (list of lists): List of columns to plot.
        - resid (bool): If True, plot residuals.

        Returns:
        - None
        """
        if 'end' not in kwargs.keys():
            end = datetime.datetime.now()
        else:
            end = kwargs['end']

        if 'start' not in kwargs.keys():
            start = end - datetime.timedelta(weeks=52*10)
        else:
            start = kwargs['start']
        
        data_to_plot = self.data[(self.data.index >= start) & (self.data.index <= end)]
        pred_to_plot = self.preds[(self.preds.index >= start) & (self.preds.index <= end)]
        columns = np.array(columns)
        fig, ax = pyplot.subplots(columns.shape[0], columns.shape[1], figsize = (16,9))
        for i in range(columns.shape[0]):
            for j in range(columns.shape[1]):
                if not resid:
                    ax[i,j].plot(data_to_plot.index, data_to_plot.iloc[:,columns[i][j]], label = 'True Data')
                    ax[i,j].plot(pred_to_plot.index, pred_to_plot.iloc[:,columns[i][j]], label = 'Pred Data')
                else:
                    ax[i,j].plot(data_to_plot.index, data_to_plot.iloc[:,columns[i][j]] - pred_to_plot.iloc[:,columns[i][j]], label = 'Residual')                
                [ax[i,j].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
                ax[i,j].set_title(data_to_plot.columns[columns[i][j]])
        pyplot.legend(loc = 'best', frameon = False)
        fig.tight_layout()
        pyplot.show()

if __name__ == "__main__":
    data = loadYC()
    data = data.tail(252)
    lscsplitter = LSCTSplitter()
    lscsplitter.fit(data=data)
    lscsplitter.inverse()
    lscsplitter.plotfactors()
    lscsplitter.plot()
    lscsplitter.plot(resid = True)