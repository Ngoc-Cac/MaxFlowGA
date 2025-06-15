from .Individual import Individual

from typing import (
    Callable,
    Literal
)


MY_SELECTION_METHODS = {
    'fps': fitness_proportionate,
    'fuss': fuss,
}
def max_flow_GA(capacity_matrix: list[list[int]],
                crossover_func: Literal[1, 2, 3] = 1, *,
                pop_size: int = 500, mutation_rate: float = 0.05,
                max_iter: int = 200, best_max_iter: int | None = None,
                selection_method: Literal['fps', 'fuss'] = 'fps',
                update_best_procedure: Callable[[int, Individual], Any] | None = None,
                update_pop_procedure: Callable[[int, list[Individual]], Any] | None = None)\
                -> dict[Literal["total_gen",
                                "gen_of_best_ind",
                                "best_of_all_gen",
                                "best_in_last_gen"], Individual | int]:
    if best_max_iter == None:
        best_max_iter = max_iter
    select_random = MY_SELECTION_METHODS[selection_method]
    maximal_capacity = min(sum(capacity_matrix[0]), col_sum(capacity_matrix, -1))

    population: list[Individual]
    new_pop: list[Individual] = [Individual(capacity_matrix) for _ in range(pop_size)]
    best_ind: Individual = new_pop[0]
    best_at_gen: int
    iter: int = 0
    best_over_gen: int = 0

    while (iter := iter + 1) < max_iter and (best_over_gen := best_over_gen + 1) < best_max_iter:
        population = new_pop
        new_pop = []
        for ind in population:
            ind.fitness(maximal_capacity)
            if ind.fitness_score > best_ind.fitness_score:
                best_over_gen = 0
                best_ind = ind
                best_at_gen = iter
                if update_best_procedure:
                    update_best_procedure(iter, best_ind)
        while len(new_pop) < pop_size:
            partA = select_random(population)
            partB = select_random(population)
            child = partA.crossover(partB, crossover_func)
            child.mutate(capacity_matrix, mutation_rate)
            new_pop.append(child)
        if update_pop_procedure:
            update_pop_procedure(iter, population)

    solution: dict[str, Individual | int] = {"total_gen": iter,
                                             "gen_of_best_ind": best_at_gen,
                                             "best_of_all_gen": best_ind,
                                             "best_in_last_gen": max(population, key=lambda x: x.fitness_score)}
    return solution