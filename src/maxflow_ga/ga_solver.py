import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

from .Individual import Individual
from .selection import fitness_proportionate, fuss


from numpy.typing import NDArray
from typing import (
    Any,
    Callable,
    Literal
)


class GA_Solver:
    __slots__ = (
        '_cap_mat',
        '_crossover_type',
        '_maximal_cap',
        '_mut_rate',
        '_pop_size',
        '_selection_scheme',
        '_select_random',
        '_update_procedures'
    )
    _CROSSOVER_TYPES = {'flow_conservation', '1-point', '2-point'}
    _SELECT_SCHEMES = {
        'fps': fitness_proportionate,
        'fuss': fuss,
    }

    def __init__(self,
        capacity_matrix: NDArray,
        pop_size: int = 500,
        mutation_rate: float = 0.05,
        *,
        crossover_type: Literal['flow_conservation', '1-point', '2-point'] = 'flow_conservation',
        selection_scheme: Literal['fps', 'fuss'] = 'fps',
        update_best_procedure: Callable[[int, Individual], Any] | None = None,
        update_pop_procedure: Callable[[int, list[Individual]], Any] | None = None
    ):
        self.capacity_matrix = capacity_matrix
        self.crossover_type = crossover_type
        self.mutation_rate = mutation_rate
        self.population_size = pop_size
        self.selection_scheme = selection_scheme

        self._update_procedures = {'best': None, 'pop': None}
        if update_best_procedure or update_pop_procedure:
            self.set_procedures(update_best_procedure, update_pop_procedure)

    @property
    def capacity_matrix(self) -> NDArray:
        return np.array(self._cap_mat)
    @capacity_matrix.setter
    def capacity_matrix(self, new_mat: NDArray):
        if not np.can_cast(new_mat, np.int64, 'same_kind'):
            raise TypeError(f'Expected numpy.NDArray of type integers, got {new_mat.dtype}')
        elif np.any(new_mat < 0):
            raise ValueError('Negative values found in capacity matrix')
        self._cap_mat = np.array(new_mat, dtype=np.int64)
        self._maximal_cap = min(self._cap_mat[0].sum(), self._cap_mat[:, -1].sum())
    
    @property
    def crossover_type(self
    ) -> Literal['flow_conservation', '1-point', '2-point']:
        return self._crossover_type
    @crossover_type.setter
    def crossover_type(self,
        new_type: Literal['flow_conservation', '1-point', '2-point']
    ):
        if not new_type in GA_Solver._CROSSOVER_TYPES:
            raise ValueError(f'No crossover type {new_type} found...')
        self._crossover_type = new_type

    @property
    def mutation_rate(self) -> float:
        return self._mut_rate
    @mutation_rate.setter
    def mutation_rate(self, new_value: int | float):
        if not isinstance(new_value, int | float):
            raise TypeError('Mutation rate must be a number')
        elif new_value < 0 or new_value > 1:
            raise ValueError('Mutation rate must be in range [0, 1]')
        self._mut_rate = new_value
    
    @property
    def population_size(self) -> int:
        return self._pop_size
    @population_size.setter
    def population_size(self, new_value: int):
        if not isinstance(new_value, int):
            raise TypeError('Population size msut be a positive integer')
        elif new_value <= 0:
            raise ValueError('Population size must be positive')
        self._pop_size = new_value

    @property
    def selection_scheme(self) -> Literal['fps', 'fuss']:
        return self._selection_scheme
    @selection_scheme.setter
    def selection_scheme(self, new_scheme: Literal['fps', 'fuss']):
        if not new_scheme in GA_Solver._SELECT_SCHEMES:
            raise ValueError(f'Selection scheme {new_scheme} not found')
        self._selection_scheme = new_scheme
        self._select_random = GA_Solver._SELECT_SCHEMES[self._selection_scheme]


    def set_procedures(self,
        update_best_procedure: Callable[[int, Individual], Any] | None = None,
        update_pop_procedure: Callable[[int, list[Individual]], Any] | None = None
    ):
        if update_best_procedure is None and update_pop_procedure is None:
            raise ValueError('At least one procedure must be set!')
        if not update_best_procedure is None:
            self._update_procedures['best'] = update_best_procedure
        if not update_pop_procedure is None:
            self._update_procedures['pop'] = update_pop_procedure

    def unset_procedure(self,
        procedure_type: Literal['best', 'pop', 'both']
    ):
        if procedure_type == 'best':
            self._update_procedures['best'] = None
        elif procedure_type == 'pop':
            self._update_procedures['pop'] = None
        elif procedure_type == 'both':
            self._update_procedures['best'] = None
            self._update_procedures['pop'] = None
        else:
            raise ValueError(f'Unrecognised procedure type {procedure_type}')

    def perform_crossover(self,
        partner_A: Individual,
        partner_B: Individual,
    ) -> Individual:
        if self._crossover_type == 'flow_conservation':
            new_dna = partner_A.flow_conservation_crossover(partner_B)
        elif self._crossover_type == '1-point':
            new_dna = partner_A.one_point_crossover(partner_B)
        elif self._crossover_type == '2-point':
            new_dna = partner_A.two_point_crossover(partner_B)

        child = Individual()
        child.dna = new_dna
        child.mutate(self._cap_mat, self._mut_rate)
        child.fitness(self._maximal_cap)

        return child
    
    def optimize(self,
        max_iter: int,
        best_max_iter: int = 0,
        verbose: bool = False
    ):
        if best_max_iter <= 0:
            best_max_iter = max_iter

        new_pop = [Individual(self._cap_mat) for _ in range(self._pop_size)]
        for ind in new_pop: ind.fitness(self._maximal_cap)
        best_ind = new_pop[0]
        best_iter = 0

        iter_bar = range(1, max_iter + 1)
        if verbose:
            iter_bar = tqdm(iter_bar, desc='Generation')
        for iter in iter_bar:
            if (best_iter := best_iter + 1) > best_max_iter:
                break

            population = new_pop
            candidate = max(population, key=lambda ind: ind.fitness_score)
            if candidate.fitness_score > best_ind.fitness_score:
                best_ind = candidate
                best_iter = 0
                best_at_gen = iter
                if verbose:
                    iter_bar.set_postfix_str(
                        f"Generation of best Individual: {best_at_gen}"
                    )
                if self._update_procedures['best']:
                    self._update_procedures['best'](iter, best_ind)
            
            new_pop = [
                self.perform_crossover(
                    self._select_random(population),
                    self._select_random(population)
                ) for _ in range(self._pop_size)
            ]
            if self._update_procedures['pop']:
                self._update_procedures['pop'](iter, population)

        return {
            "total_gen": iter,
            "gen_of_best_ind": best_at_gen,
            "best_of_all_gen": best_ind,
        }