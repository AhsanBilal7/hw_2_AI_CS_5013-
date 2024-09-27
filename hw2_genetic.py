# Importing required libaries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import product

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




GENES = list(product([-1, 1], repeat=6))
POP_SIZE = 20
MUT_RATE = 0.1
TOTAL_CHROMOSOMES = 2
CROSS_OVER_POINT = 3
TARGET = 'rayan ali' #total number of weights
# GENES = ' abcdefghijklmnopqrstuvwxyz'


def f(X, weights):
    """
    Function to compute predictions based on input features X and weights.
    """
    # weights = np.array(weights) 
    print(weights)
    return np.dot(X, weights)

# Define the loss function r(w)
def loss_function(X, y, weights):
    """
    Compute the loss function based on predictions and ground truth labels.
    """
    predictions = f(X, weights)
    errors = predictions - y
    squared_errors = np.square(errors)
    return np.mean(squared_errors)

# Define the fitness function as e^(-e_r(w))
def fitness_function(X, y, weights):
    """
    Compute the fitness function as e^(-e_r(w)).
    """
    loss = loss_function(X, y, weights)
    fitness = np.exp(-loss)
    return fitness, loss

def initialize_pop():
    population = list()

    for i in range(POP_SIZE):
        population.append(list(random.choice(GENES)))

    return population



def crossover(selected_chromo):
    offspring_cross = []
#   for i in range(int(POP_SIZE)):
#     parent1 = random.choice(selected_chromo)
#     parent2 = random.choice(population[:int(POP_SIZE*50)])

#     p1 = parent1[0]
#     p2 = parent2[0]

#     child =  p1[:CROSS_OVER_POINT] + p2[CROSS_OVER_POINT:]
#     offspring_cross.extend([child])

    # for i in range(int(POP_SIZE)):
    #     candidate_chromosomes = random.sample(selected_chromo, k=3)
    #     candidate_chromosomes_sorted = sorted(candidate_chromosomes, key= lambda x:x[1], reverse=True)
  

    #     p1 = candidate_chromosomes_sorted[0][0]
    #     p2 = candidate_chromosomes_sorted[1][0]

    #     # crossover_point = random.randint(1, CHROMO_LEN-1)
    #     child =  p1[:CROSS_OVER_POINT] + p2[CROSS_OVER_POINT:]
    #     offspring_cross.extend([child])
    # for i in range(int(POP_SIZE)):
    candidate_chromosomes = random.sample(selected_chromo, k=3)
    candidate_chromosomes_sorted = sorted(candidate_chromosomes, key= lambda x:x[1], reverse=True)


    p1 = candidate_chromosomes_sorted[0][0]        # [[1,1,1,1,1,1]]
    p2 = candidate_chromosomes_sorted[1][0]
    print("Parents >>>>: ",p1,p2)
    # crossover_point = random.randint(1, CHROMO_LEN-1)
    first_half = list(p1[:CROSS_OVER_POINT])
    second_half = list(p2[CROSS_OVER_POINT:])
    child = first_half + second_half
    # second_half.extend(first_half)
    # child = second_half
    # child = p1[:CROSS_OVER_POINT] + p2[CROSS_OVER_POINT:]
    offspring_cross.append(child)
    print("First half: ", p1[:CROSS_OVER_POINT])
    print("second half: ", p2[CROSS_OVER_POINT:])
    print("Offspring: ",offspring_cross)
    return offspring_cross
#   return offspring_cross
def mutate(offspring, MUT_RATE):
    mutated_offspring = []

    # for arr in offspring:
    #     print("Offspring: ",arr)
    #     for i in range(len(arr)):
    #         if random.random() < MUT_RATE:
    #             arr[i] = random.choice([1,-1])
    #     mutated_offspring.append(arr)
    for arr in offspring:
        noise = np.random.normal(0, MUT_RATE, (len(arr),))  # Gaussian noise (mean=0, std=MUT_RATE)
        # print("Random noise: ", noise, len(arr))
        mutated_arr = arr + noise  # Adding noise to the array
        mutated_offspring.append(mutated_arr)
    # print("mutate offspring: ", mutated_offspring)
    return mutated_offspring

def selection(population):
    sorted_chromo_pop = sorted(population, key= lambda x: x[2])
    return sorted_chromo_pop[:int(0.5*POP_SIZE)]

def fitness_cal(x, y, chromo_from_pop):
    fitness_probability, loss = fitness_function(x, y, chromo_from_pop)
    
    return [chromo_from_pop, fitness_probability, loss]

def replace(new_gen, population):
    new_gen = sorted(new_gen, key= lambda x: x[2])

    for _ in range(len(population)):
        if population[_][2] > new_gen[_][2]:
          population[_][0] = new_gen[_][0]
          population[_][1] = new_gen[_][1]
          population[_][2] = new_gen[_][2]
    return population


losses = []

def main(POP_SIZE, MUT_RATE, TARGET, GENES):
    # 1) initialize population
    initial_population = initialize_pop()
    found = False
    population = []
    generation = 1
    X, Y = load_dataset()
    # 2) Calculating the fitness for the current population
    for _ in range(len(initial_population)):
        population.append(fitness_cal(X, Y, initial_population[_]))

    # now population has 2 things, [chromosome, fitness]
    # 3) now we loop until TARGET is found
    # while not found:
    population = sorted(population, key= lambda x:x[2], reverse=True)
    while generation < 100:
        # print("populations: ", population)

        # 3.1) select best people from current population
        selected_chromosomes = selection(population)

        # 3.2) mate parents to make new generation
        population = sorted(population, key= lambda x:x[2], reverse=True)
        crossovered_child =  crossover(selected_chromosomes)
                
        # 3.3) mutating the childeren to diversfy the new generation
        mutated = mutate(crossovered_child, MUT_RATE)

        new_gen = []
        for _ in mutated:
            # print(crossovered_child, _)
            new_gen.append(fitness_cal(X,Y, _))
        
        # 3.4) replacement of bad population with new generation
        # we sort here first to compare the least fit population with the most fit new_gen

        new_gen = sorted(new_gen, key= lambda x:x[2])
        # population = replace(new_gen, population)
        population.extend(new_gen)
        # print(f"Generation {generation} -> New-Gen Length {len(new_gen)} -> Pop Length -> {len(population)}")
        # print(new_gen.shape)
        # print(population.shape)
        population = sorted(population, key= lambda x:x[2], reverse=True)
        # population = sorted(population, key= lambda x:x[2], reverse=True)
        if (population[-1][1] == 0):
            print('Target found')
            print('String: ' + str(population[0][0]) + ' Generation: ' + str(generation) + ' Fitness: ' + str(population[0][1]), ' Loss: ' + str(population[0][2]))
            break
        # print('String: ' + str(population[0][0]) + ' Generation: ' + str(generation) + ' Fitness: ' + str(population[0][1]), ' Loss: ' + str(population[0][2]))
        print("---------------------------------------")
        generation+=1
        losses.append(population[-1][2])
    return population


population = main(POP_SIZE, MUT_RATE, TARGET, GENES)


population = sorted(population, key= lambda x:x[2])
optimized_weights, best_loss = population[0][0], population[0][2]
print("Optimized: ", optimized_weights)
print("Best Loss: ", best_loss)
plt.plot(losses, label='Errors 1', color='blue', marker='o')  # First dataset

plt.show()