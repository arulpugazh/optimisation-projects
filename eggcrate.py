from deap import base, creator, tools
import random
import math

def eggcrate(x):
    x1, x2 = list(x)
    return x1 ** 2 + x2 ** 2 + 25 * (math.sin(x1) ** 2 + math.sin(x2) ** 2)

#Create Types
#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", list, fitness=creator.FitnessMin)

#Step 1: Setup for Initialize Population
IND_SIZE = 100

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
def generate_ind(icls):
    return icls(random.sample(range(0 - (2 * (22//7)), int(2 * (22//7))),2))


toolbox.register("individual", generate_ind, icls=creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

def evaluate(individual):
    return eggcrate(individual),
toolbox.register("evaluate", evaluate)

def algo():
    pop = toolbox.population(n=50)
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

    return pop

sol = algo()
print(sol)