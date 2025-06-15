from random import uniform, choices
from .Individual import Individual


def fuss(population: list[Individual]) -> Individual:
    """
    Fitness Uniform Selection Scheme
    
    This selection scheme goes as follows:
    1. Choose a uniform random number between `f_min` and `f_max`,
        where `f_min` and `f_max` are the lowest and highest fitness
        value in the population.
    2. Selects the individual with the fitness score closest to the random
        number chosen beforehand.

    :param list[Individual] population: A list of Individuals to choose from.
    :return: The chosen Individual
    :rtype: Individual
    """
    f_rand = uniform(
        min(population, key=lambda ind: ind.fitness_score).fitness_score,
        max(population, key=lambda ind: ind.fitness_score).fitness_score
    )
    return min(population, key=lambda ind: abs(ind.fitness_score - f_rand))

def fitness_proportionate(population: list[Individual]) -> Individual:
    r"""
    Select a random Individual from a pool of Individuals based on 
    their fitness. For the *i*-th Individual, the probability :math:`p_i`
    of it being chosen is:

    .. math::
        p_i = \frac{\text{fit}_i}{\sum_{j=0}^{n-1}\text{fit}_j}

    where :math:`fit_i` is the fitness score of the *i*-th Individual.

    :param list[Individual] population: A list of Individuals to choose from.
    :return: The chosen Individual
    :rtype: Individual
    """
    return choices(
        population,
        [max(ind.fitness_score, 0) for ind in population],
        k=1
    )[0]