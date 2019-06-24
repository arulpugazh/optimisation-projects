from deap import base, creator, tools
import random
import math
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy

def generate_ind(icls):
    b1 = random.uniform(2.6, 3.6)`
    b2 = random.uniform(0.7, 0.8)
    b3 = random.uniform(17, 28)
    b4 = random.uniform(7.3, 8.3)
    b5 = random.uniform(7.3, 8.3)
    b6 = random.uniform(2.9, 3.9)
    b7 = random.uniform(5, 5.5)
    return icls([b1,b2,b3,b4,b5,b6,b7])

def golinksi(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 0.7854 * x1 * (x2 ** 2) * (3.3333 * (x3 ** 2) + 14.9334 * x3 - 43.0934) \
           - 1.5079 * x1 * ((x6 ** 2) + (x7 ** 2)) + 7.477 * ((x6 ** 3) + (x7 ** 3)) \
           + 0.7854 * (x4 * (x6 ** 2) + (x5 * (x7 ** 2)))

def gol_cons1(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    res = 1 - (27 * (x1 ** -1) * (x2 ** -2) * (x3 ** -1))
    if res==0:
        return True
    else:
        return False


def gol_cons2(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    res = 1 - (397.5 * (x1 ** -1) * (x2 ** -2) * (x3 ** -2))
    if res==0:
        return True
    else:
        return False


def gol_cons3(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return - (1.93 * (x2 ** -1) * (x3 ** -1) * (x4 ** 3) * (x6 ** -4) - 1)

def gol_cons4(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return - (1.93 * (x2 ** -1) * (x3 ** -1) * (x5 ** 3) * (x7 ** -4) - 1)

def gol_cons5(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    numerator = ((745 * x4 * (x2**-1) *(x3**-1))**2 + 16.9* (10**6)) ** 0.5
    denominator = (110.0 * (x6**3))
    return 1 - (numerator/denominator)

def gol_cons6(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    numerator = ((745 * x5 * (x2**-1) *(x3**-1))**2 + 157.5* (10**6)) ** 0.5
    denominator = (85.0 * (x7**3))
    return 1 - (numerator/denominator)

def gol_cons7(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 1 - (x2*x3/40)

def gol_cons8(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 1 - (5*x2/x1)

def gol_cons9(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 1 - (x1/(12*x2))

def gol_cons24(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 1 - ((1.5 * x6 + 1.9) * (x4**-1))

def gol_cons25(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 1 - ((1.1 * x7 + 1.9) * (x5**-1))

#Create Types
#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", list, fitness=creator.FitnessMin)

#Step 1: Setup for Initialize Population
IND_SIZE = 100

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)

toolbox.register("individual", generate_ind, icls=creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

def evaluate(individual):
    return golinksi(individual),

toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons1,10))
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons2,10))
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons3,10))
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons4,10))
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons5,10))
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons6,10))
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons7,10))
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons8,10))
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons9,10))
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons24,10))
toolbox.decorate("evaluate", tools.DeltaPenalty(gol_cons25,10))

def algo():
    pop = toolbox.population(n=500)
    print(f"Initial Population:{pop}")
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = [toolbox.evaluate(p) for p in pop]
    print(f"Fitnesses: {fitnesses}")
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = [toolbox.clone(off) for off in offspring]

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        fits=[]
        for off in offspring:
            fits.append(off.fitness.values[0])
    return pop,fits

sol,f=algo()
print(type(f))
l=[]
for s in sol:
    l.append(list(s))
df =DataFrame.from_records(l)
df.insert(loc=0,column="Fitness", value=f)
print(df)
# fig, ax = plt.subplots()
#
# for key, grp in df.iteritems():
#    ax = grp.plot(ax=ax, kind='line', x='x', y='y', c=numpy.random.rand(3,), label=key)

plt.plot(f)
plt.legend(loc='best')
plt.show()