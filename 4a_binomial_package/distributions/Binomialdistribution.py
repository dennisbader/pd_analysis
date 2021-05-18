import math
import numpy as np

from matplotlib import pyplot as plt
from .Generaldistribution import Distribution


class Binomial(Distribution):
    """ Binomial distribution class for calculating and 
    visualizing a Binomial distribution.
    
    Attributes:
        mean (float) representing the mean value of the distribution
        stdev (float) representing the standard deviation of the distribution
        data_list (list of floats) a list of floats to be extracted from the data file
        p (float) representing the probability of an event occurring
                
    """

    def __init__(self, p=1, n=1):
        self.p = p
        self.n = n
        Distribution.__init__(self, self.calculate_mean(), self.calculate_stdev())

    def calculate_mean(self):
        """Function to calculate the mean from p and n
        
        Args: 
            None
        
        Returns: 
            float: mean of the data set
    
        """
        return self.p * self.n

    def calculate_stdev(self):
        """Function to calculate the standard deviation from p and n.
        
        Args: 
            None
        
        Returns: 
            float: standard deviation of the data set
        """
        return math.sqrt(self.n * self.p * (1 - self.p))

    def replace_stats_with_data(self):
        """Function to calculate p and n from the data set. The function updates the p and n variables of the object.
        
        Args: 
            None
        
        Returns: 
            float: the p value
            float: the n value
    
        """
        self.n = self.data.shape[0]
        self.p = (self.data == 1).sum() / self.n
        self.mean = self.calculate_mean()
        self.stdev = self.calculate_stdev()
        return self.p, self.n
    
    # TODO: write a method plot_bar() that outputs a bar chart of the data set according to the following specifications.
    def plot_bar(self):
        """Function to output a histogram of the instance variable data using 
        matplotlib pyplot library.
        
        Args:
            None
            
        Returns:
            None
        """
        xs, ys = np.unique(self.data, return_counts=True)
        plt.bar(x=xs, height=ys)
        plt.xticks(ticks=xs, labels=[str(x) for x in xs])
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of data')
        plt.show()
        return

    def pdf(self, k):
        """Probability density function calculator for the binomial distribution.
        
        Args:
            k (float): point for calculating the probability density function
            
        
        Returns:
            float: probability density function output
        """
        n, p = self.n, self.p
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k)) * p**k * (1 - p)**(n - k)

    def plot_pdf(self):
        """Function to plot the pdf of the binomial distribution
        
        Args:
            None
        
        Returns:
            list: x values for the pdf plot
            list: y values for the pdf plot
            
        """
        ks = [k for k in range(self.n + 1)]
        ys = [self.pdf(k) for k in ks]
        plt.bar(x=ks, height=ys, width=1)
        plt.xlabel('k')
        plt.ylabel('PDF(k)')
        plt.title('Binomial PDF for different k values')
        plt.show()
        return

    def __add__(self, other):
        """Function to add together two Binomial distributions with equal p
        
        Args:
            other (Binomial): Binomial instance
            
        Returns:
            Binomial: Binomial distribution
            
        """
        
        try:
            assert self.p == other.p, 'p values are not equal'
        except AssertionError as error:
            raise
        p_new = self.p
        n_new = self.n + other.n
        return Binomial(p=p_new, n=n_new)

    def __repr__(self):
        """Function to output the characteristics of the Binomial instance
        
        Args:
            None
        
        Returns:
            string: characteristics of the Binomial object
        
        """
        return 'mean {}, standard deviation {}, p {}, n {}'.format(self.mean, self.stdev, self.p, self.n)
