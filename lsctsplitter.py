from matplotlib import pyplot
from data import yc
from api.sklearn_lsct import SklearnLSCTSplitter

class LSCTSplitter(SklearnLSCTSplitter):
    def plot_factors(self, X, t = None, fig = None, figsize = (16,9), show = True):
        """
        Plots the factors.

        Args:
        - *args, **kwargs: Additional arguments for plotting.
        """
        factors = self.fit_transform(X)
        t = range(factors.shape[0]) if t is None else t
        fig = pyplot.figure(figsize=figsize)
        axfactors  = []
        axloadings = []
        titles = ['level', 'slope', 'curvature', 'twist']
        for i in range(self.n_factors):
            axfactors.append(pyplot.subplot2grid((self.n_factors,3), (i,0), colspan = 2))
            axloadings.append(pyplot.subplot2grid((self.n_factors,3), (i,2)))
            axfactors[i].plot(t, factors[:,i])
            axfactors[i].set_title(titles[i])
            [axfactors[i].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
            axloadings[i].plot(self.maturities, self.loadings_[i], marker = 'o')
            axloadings[i].set_title(titles[i] +' loading')
            [axloadings[i].spines[loc_axis].set_visible(False) for loc_axis in ['top','right', 'bottom']]
        fig.tight_layout()
        if show:
            pyplot.show()
        else:
            return fig

if __name__ == "__main__":
    data = yc.load('./data/yc_data.csv')
    lscsplitter = LSCTSplitter([1.0/12, 2.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0], n_factors=4, lambdas=0.49, verbose=0)
    factors = lscsplitter.fit_transform(data.values)
    Xs = lscsplitter.inverse_transform(factors)
    lscsplitter.plot_factors(data.values, t = data.index)
    print(Xs.shape)
