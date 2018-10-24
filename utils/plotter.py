## =============================================================================== ##
## 																				   ##
##	This file contains a number of plotting functions for use within the labs/	   ##
##	assignments.																   ##
## 																				   ##
## =============================================================================== ##
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_grid(x, shape=None, **heatmap_params):
    """Function for reshaping and plotting vector data.
    If shape not given, assumed square.
    """
    if shape is None:
        width = int(np.sqrt(len(x)))
        if width == np.sqrt(len(x)):
            shape = (width, width)
        else:
            print('Data not square, supply shape argument')
    sns.heatmap(x.reshape(shape), annot=True, **heatmap_params)


def plot_hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix.
    Source: https://matplotlib.org/examples/specialty_plots/hinton_demo.html
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
	

def plot_confusion_matrix(cm, classes=None, norm=True, title='Confusion matrix', ax=None, **kwargs):
    """Plots a confusion matrix."""
    heatmap_kwargs = dict(annot=True, fmt='d')
    if norm:
        cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        heatmap_kwargs['data'] = cm_norm
        heatmap_kwargs['vmin']=0.
        heatmap_kwargs['vmax']=1.
        heatmap_kwargs['fmt']='.3f'
    else:
        heatmap_kwargs['data'] = cm
    if classes is not None:
        heatmap_kwargs['xticklabels']=classes
        heatmap_kwargs['yticklabels']=classes
    if ax is None:
        ax = plt.gca()
    heatmap_kwargs['ax'] = ax
    heatmap_kwargs.update(kwargs)
    sns.heatmap(**heatmap_kwargs)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
	
	
def scatter_jitter(arr1, arr2, jitter=0.2):
    """
     Plots a joint scatter plot of two arrays by adding small noise to each example. 
     The noise is proportional to variance in each dimension.
     
     :param arr1:   1D numpy array containing the first data-variable (will be plotted along x-axis)
     :param arr2:   1D numpy array containing the second data-variable (will be plotted along y-axis)
     :param jitter: Amount of noise to add: this is a proportion (0 to 1) of the variance in each dimension
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    arr1 = arr1 + jitter*arr1.std(axis=0)*np.random.standard_normal(arr1.shape)
    arr2 = arr2 + jitter*arr2.std(axis=0)*np.random.standard_normal(arr2.shape)
    plt.scatter(arr1, arr2, marker=4)

    
def plot_SVM_DecisionBoundary(clfs, X, y, title=None, labels=None):
    """
    Plots decision boundaries for classifiers with 2D inputs.
    
    Acknowldgement: Based on Example in http://scikit-learn.org/%SCIKIT_VERSION%/auto_examples/svm/plot_iris.html
    
    Parameters
    ----------
    clf : list
        Classifiers for which decision boundaries will be displayed. These should have been already trained (fit)
        with the necessary data.
    X : array
        Input features used to train the classifiers.
    y : array
        Class Labels corresponding to each row of X
    title : list, optional
        Titles for classifiers.
    labels : list, optional
        Feature names (in order as they appear in X)
    
    """
    
    assert X.shape[1] == 2 # Input should be 2D
    if title is not None:
        assert len(clfs) == len(title)
    
    h = .04 # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.figure(figsize=(15,5))
    for i, clf in enumerate(clfs):
        plt.subplot(1, len(clfs), i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        # Training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        if labels is not None:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        else:
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        if title is not None:
            plt.title(title[i])