# Importing required libaries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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
# example of hill climbing the test set for a classification task
from random import randint
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import random
# ignore warning
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('./CreditCard.csv')
df['CarOwner'] = df['CarOwner'].map({'Y': 1, 'N': 0})
df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N': 0})
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
df.duplicated().sum()
df.columns
df.info()
cols = ['Gender', 'CarOwner', 'PropertyOwner']
# Set up the figure and axes


def column_distribution():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    # Plot univariate distributions for each column
    for i, col in enumerate(cols):
        sns.countplot(data=df , x=col, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}' , fontsize=18 )
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x')
        axes[i].grid(True)
    plt.tight_layout()


def feature_corrlelation():
    features=['CreditApprove',	'Gender',	'CarOwner',	'PropertyOwner',	'#Children',	'WorkPhone',	'Email_ID']
    plt.figure(figsize = (10,6))
    sns.heatmap(df[features].corr(), annot=True , cmap='coolwarm')
    plt.title("Correlation Heatmap")


# load or prepare the classification dataset
def load_dataset():
	df = pd.read_csv('./CreditCard.csv')

	df['CarOwner'] = df['CarOwner'].map({'Y': 1, 'N': 0})
	df['PropertyOwner'] = df['PropertyOwner'].map({'Y': 1, 'N': 0})
	df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
	X = df[['Gender',	'CarOwner',	'PropertyOwner',	'#Children',	'WorkPhone',	'Email_ID']].to_numpy()
	Y = df[['CreditApprove',]].to_numpy()
	return X, Y



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



if __name__ == "__main__":
    # load the dataset
    X, Y = load_dataset()
    print(X.shape, Y.shape)
    # split dataset into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # run hill climb
    optimal_weight, errors, values = hill_climb_testset(X, Y, 100)
    best_error_optimal_weight = round(loss_function(X, Y, optimal_weight),3)
    # feature_corrlelation()
    # column_distribution()

    plt.figure(figsize = (10,6))
    plt.text(x=1.4, y=0.8, s=f"Optimal weights: {optimal_weight}\nError er(w): {round(loss_function(X, Y, optimal_weight),3)}", fontsize=12, color='green')
    plt.plot(values, label='Current Solution', color='red', marker='x')   # Second dataset
    plt.plot(errors, label='Best Solutions', color='blue', marker='o')  # First dataset
    # Labeling the axes
    plt.xlabel('Round of Search')  # Replace with your desired x-axis label
    plt.ylabel('Error of weights Er(w)')  # Replace with your desired y-axis label

    # Adding a title and legend
    plt.title('Figure 1 for Local Search')  # Optional: add a title
    plt.legend()
    print(f"For whole dataset, the error for weights: {optimal_weight} is {best_error_optimal_weight}")
    # print(f"The error for weights: {optimal_weight} is {loss_function(X_test, y_test, optimal_weight)}")
    pyplot.show()