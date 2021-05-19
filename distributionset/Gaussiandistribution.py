import math
import numpy as np

from matplotlib import pyplot as plt
from .Generaldistribution import Distribution


class Gaussian(Distribution):
	""" Gaussian distribution class for calculating and
	visualizing a Gaussian distribution.
	
	Attributes:
		mean (float) representing the mean value of the distribution
		stdev (float) representing the standard deviation of the distribution
		data_list (list of floats) a list of floats extracted from the data file
			
	"""
	def __init__(self, mu=0, sigma=1):
		Distribution.__init__(self, mu, sigma)

	def calculate_mean(self):
		"""Function to calculate the mean of the data set.
		
		Args: 
			None
		
		Returns: 
			float: mean of the data set
	
		"""
		return np.mean(self.data)

	def calculate_stdev(self, sample=True):
		"""Function to calculate the standard deviation of the data set.

		Args: 
			sample (bool): whether the data represents a sample or population
		
		Returns: 
			float: standard deviation of the data set
	
		"""
		if sample:
			n = int(self.data.shape[0] - 1)
		else:
			n = int(self.data.shape[0])

		mean = self.calculate_mean()
		return math.sqrt(np.sum((self.data - mean)**2) / n)

	def replace_stats_with_data(self, sample=True):
		"""Function to calculate mean and standard devation from the data set.
		The function updates the mean and stdev variables of the object.

		Args:
			None

		Returns:
			float: the mean value
			float: the stdev value

		"""
		self.mean = self.calculate_mean()
		self.stdev = self.calculate_stdev(sample=sample)
		return self.mean, self.stdev

	def plot_histogram(self):
		"""Function to output a histogram of the instance variable data using
		matplotlib pyplot library.
		
		Args:
			None
			
		Returns:
			None
		"""
		plt.hist(self.data)
		plt.title('Histogram of Data')
		plt.xlabel('data')
		plt.ylabel('count')
		plt.show()
		return

	def pdf(self, x):
		"""Probability density function calculator for the gaussian distribution.
		
		Args:
			x (float): point for calculating the probability density function
			
		
		Returns:
			float: probability density function output
		"""
		return (1.0 / (self.stdev * math.sqrt(2*math.pi))) * math.exp(-0.5*((x - self.mean) / self.stdev) ** 2)

	def plot_histogram_pdf(self, n_spaces=50):
		"""Function to plot the normalized histogram of the data and a plot of the
		probability density function along the same range
		
		Args:
			n_spaces (int): number of data points 
		
		Returns:
			list: x values for the pdf plot
			list: y values for the pdf plot
			
		"""
		mu = self.mean
		sigma = self.stdev

		min_range = self.data.min()
		max_range = self.data.max()

		# calculates the interval between x values
		interval = (max_range - min_range) / n_spaces

		# calculate the x values to visualize
		interval_counter = np.arange(n_spaces)
		x = min_range + interval * interval_counter
		y = self.pdf(x)

		# make the plots
		fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
		fig.subplots_adjust(hspace=.5)
		ax1.hist(self.data, density=True)
		ax1.set_title('Normed Histogram of Data')
		ax1.set_ylabel('Density')

		ax2.plot(x, y)
		ax2.set_title('Normal Distribution for \n Sample Mean and Sample Standard Deviation')
		ax1.set_ylabel('Density')
		plt.show()
		return x, y

	def __add__(self, other):
		"""Function to add together two Gaussian distributions
		
		Args:
			other (Gaussian): Gaussian instance
			
		Returns:
			Gaussian: Gaussian distribution
			
		"""
		result = Gaussian()
		result.mean = self.mean + other.mean
		result.stdev = math.sqrt(self.stdev ** 2 + other.stdev ** 2)
		return result

	def __repr__(self):
		"""Function to output the characteristics of the Gaussian instance
		
		Args:
			None
		
		Returns:
			string: characteristics of the Gaussian
		
		"""
		return "mean {}, standard deviation {}".format(self.mean, self.stdev)


# x = Gaussian()
# x.read_data_file('/Users/Dennis/projects/udacity/data_science_course/pd_analysis/4a_binomial_package/numbers.txt')
# x.replace_stats_with_data()