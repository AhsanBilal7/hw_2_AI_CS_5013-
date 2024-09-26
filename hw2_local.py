# Importing required libaries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./CreditCard.csv')

df['CarOwner'] = df['CarOwner'].map({'Y': 1, 'N': 0})
df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N': 0})
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
df.head()



df.duplicated().sum()
df.columns


df.info()


cols = ['Gender', 'CarOwner', 'PropertyOwner']

# Set up the figure and axes

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot univariate distributions for each column
for i, col in enumerate(cols):
    sns.countplot(data=df , x=col, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}' , fontsize=18 )
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x')
    axes[i].grid(True)

plt.tight_layout()
# plt.show()



features=['CreditApprove',	'Gender',	'CarOwner',	'PropertyOwner',	'#Children',	'WorkPhone',	'Email_ID']

plt.figure(figsize = (10,6))
sns.heatmap(df[features].corr(), annot=True , cmap='coolwarm')
plt.title("Correlation Heatmap")
# plt.show()



# example of hill climbing the test set for a classification task
from random import randint
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

import numpy as np
import pandas as pd

# Visualization libaraires
import matplotlib.pyplot as plt
import seaborn as sns

# ignore warning
import warnings
warnings.filterwarnings("ignore")


# load or prepare the classification dataset
def load_dataset():
	df = pd.read_csv('./CreditCard.csv')

	df['CarOwner'] = df['CarOwner'].map({'Y': 1, 'N': 0})
	df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N': 0})
	df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
	X = df[['Gender',	'CarOwner',	'PropertyOwner',	'#Children',	'WorkPhone',	'Email_ID']].to_numpy()
	Y = df[['CreditApprove',]].to_numpy()
	return X, Y


# Create a list with random entries of 1 or -1
# list_length = 8
# random_list = [random.choice([1, -1]) for _ in range(list_length)]

# print(random_list)



# example of hill climbing the test set for a classification task
from random import randint
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import random
# load or prepare the classification dataset
# def load_dataset():
# 	return make_classification(n_samples=5000, n_features=20, n_informative=15, n_redundant=5, random_state=1)


# load or prepare the classification dataset
def load_dataset():
	df = pd.read_csv('./CreditCard.csv')
	df = df.dropna()
	df['CarOwner'] = df['CarOwner'].map({'Y': 1, 'N': 0})
	df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N': 0})
	df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
	X = df[['Gender',	'CarOwner',	'PropertyOwner',	'#Children',	'WorkPhone',	'Email_ID']].to_numpy()
	Y = df[['CreditApprove',]].to_numpy()
	return X, Y

# evaluate a set of predictions
def evaluate_predictions(y_test, yhat):
	return accuracy_score(y_test, yhat)

# create a random set of predictions
def random_predictions(n_examples):
	return [randint(0, 1) for _ in range(n_examples)]


def random_start_point(number_of_weight):
	random_list = [random.choice([1, -1]) for _ in range(number_of_weight)]
	return random_list

def adjcent_weight(incoming_list):
	adjcent_list = [-x for x in incoming_list]
	return adjcent_list

# Define the function f(x) with weights
def f(x, weights):
    return np.dot(x, weights)


# Define the loss function r(w)
def loss_function(X, y, weights):
    # X is the matrix of input features
    # y is the vector of ground truth labels
    # weights is the vector of weights
    predictions = f(X, weights)
    errors = predictions - y
    squared_errors = np.square(errors)
    return np.mean(squared_errors)

# modify the current set of predictions
def modify_predictions(current, n_changes=1):
	# copy current solution
	updated = current.copy()
	for i in range(n_changes):
		# select a point to change
		ix = randint(0, len(updated)-1)
		# flip the class label
		updated[ix] = 1 - updated[ix]
	return updated

# run a hill climb for a set of predictions
def hill_climb_testset(sample_x, sample_y, max_iterations):
    total_number_weight = sample_x.shape[1]
    errors = list()
    values = list()
    # generate the initial solution
    weights = random_start_point(number_of_weight = total_number_weight)
    # evaluate the initial solution
    starting_weights = [-1]*6
    solution_weights = adjcent_weight(starting_weights)
    
    error = loss_function(sample_x, sample_y,starting_weights)
    # value = loss_function(sample_x, sample_y,adjcent_weight([-1]*6))
    print(error)
    print(sample_x.shape)
    print(sample_y.shape)
    # hill climb to a solution
    # record error
    
    
    for i in range(max_iterations):
        value = loss_function(sample_x, sample_y, solution_weights)
        if value < error:
            # weights = adjcent_weight(weights)
            best_weights, best_error = solution_weights, error	
            solution_weights = adjcent_weight(weights)
            # solution_weights, error = weights, value
            error = value
            print('>%d, error=%.3f' % (i, error))
        else:
            solution_weights = random_start_point(number_of_weight = total_number_weight)
			# weights = random_start_point(number_of_weight = total_number_weight)
		
		
		
		
        errors.append(error)
        values.append(value)

        # stop once we achieve the best error
        if error == 0.0:
            break
        # evaluate candidate
        # check if it is as good or better
        # print(f"Value:	{value} --------  error: {error}")

    return best_weights, errors, values

# load the dataset
X, y = load_dataset()
print(X.shape, y.shape)
# split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# run hill climb
# yhat, scores = hill_climb_testset(X_test, y_test, 20000)
solution_weights, errors, values = hill_climb_testset(X_train, y_train, 100)
# plot the scores vs iterations
# pyplot.plot(errors)
plt.figure(figsize = (10,6))

plt.plot(errors, label='Errors 1', color='blue', marker='o')  # First dataset
plt.plot(values, label='Errors 2', color='red', marker='x')   # Second dataset


print(f"The train error for weights: {solution_weights} is {loss_function(X_train, y_train, solution_weights)}")
print(f"The test error for weights: {solution_weights} is {loss_function(X_test, y_test, solution_weights)}")
pyplot.show()