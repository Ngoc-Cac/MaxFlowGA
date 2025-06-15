import random as rand
import numpy as np

from maxflow_ga import _np_rng


from numpy.typing import NDArray
from typing import Literal


def random_flow(capacity_matrix: NDArray) -> NDArray:
    """Generate a random solution, given the capacity matrix.

    :param NDArray capacity_matrix: The capacity matrix of a network.
    :return: The flow matrix.
    :rtype: NDArray
    """
    return _np_rng.integers(
        np.zeros(capacity_matrix.shape),
        capacity_matrix,
        capacity_matrix.shape,
        endpoint=True
    )

class Individual():
    """
    ## Network Flow - Max Flow Optimisation
    An individual is a feasible solution of the problem. It's DNA is defined
    to be a network that has been filled in.

    Internally, the DNA is 2D-square matrix, similar to the capacity matrix.
    Each element shows the flow of a solution. Each element must be between
    0 and its flow capacity.
    """
    __slots__ = "_dna", "inflow", "outflow", "fitness_score"
    def __init__(self, capacity_matrix: NDArray | None = None):
        r"""
        When creating an individual, a capacity matrix is needed to
        generate a random DNA. If no capacity matrix is given then an
        Individual is created without declaring any of its attribute.

        A capacity matrix is a 2D-square matrix. The matrix shows the flow
        capacity from one vertex to another. For any element :math:`e_{ij}`, it
        represents the flow from the vertex indexed at i to the vertex indexed at j.

        For example, if the element at index `(0, 1)` has a value of `10`,
        that means the flow from the 0th vertex to the 1st vertex is `10`.

        :param NDArray | None capacity_matrix: The capacity matrix of the network.
            This is `None` by default. In such case, The DNA is not initialized.
        """
        if not capacity_matrix is None:
            self._dna = random_flow(capacity_matrix)
            self._update_flowrate()
    
    # getter and setter for dna attribute
    # note, every time a new dna is assigned, we must update the flowrate
    @property
    def dna(self) -> NDArray:
        if not hasattr(self, "_dna"):
            raise AttributeError("'dna' attribute has not been initialised")
        return np.array(self._dna)
    @dna.setter
    def dna(self, new_dna: NDArray):
        self._dna = np.array(new_dna)
        self._update_flowrate()

    
    def _update_flowrate(self):
        """
        Update the Inflows and Outflows of the solution.
        
        In simple words, inflow implies the incoming flow to a vertex.
        Similarly, outflow implies the outcoming flow at a vertex.

        For example: Consider the given solution to a graph containing 4 nodes
        ```
           [[0, 1, 0, 2],
            [0, 0, 2, 0],
            [0, 3, 0, 4],
            [0, 0, 0, 0]]
        ```
        """
        self.inflow = self._dna.sum(axis=0)
        self.outflow = self._dna.sum(axis=1)

    def _overflow_mutation(self,
        capacity_matrix: NDArray,
        mutable: NDArray
    ):
        overflow = (self.inflow > self.outflow)
        mutable_vertices = np.argwhere(overflow & mutable)[:, 0]


        # go over every rows (every outcoming edges)
        # and get the edges where we can increment
        mutable_idx = np.argwhere(
            (capacity_matrix[mutable_vertices] != 0) &
            (self._dna[mutable_vertices] < capacity_matrix[mutable_vertices])
        )
        unique, idx, counts = np.unique(
            mutable_idx[:, 0],
            return_counts=True,
            return_index=True,
        )
        if unique.shape[0]: # just condition for efficiency
            rand_idx = _np_rng.integers(
                np.zeros(counts.shape), counts,
                counts.shape
            ) + idx

            rows, cols = mutable_idx[rand_idx, 0], mutable_idx[rand_idx, 1]
            rows = mutable_vertices[rows]
            self._dna[rows, cols] += 1
        
        mutable_vertices = np.delete(mutable_vertices, unique)
        if mutable_vertices.shape[0]:
            # if there are still vertices that havent been mutated (outflow at capacity)
            # then go through every columns (every incoming edges)
            # and decrement the inflow
            mutable_idx = np.argwhere(
                (capacity_matrix[:, mutable_vertices] != 0) &
                (self._dna[:, mutable_vertices] > 0)
            )
            unique, idx, counts = np.unique(
                mutable_idx[:, 1],
                return_counts=True,
                return_index=True,
            )
            rand_idx = _np_rng.integers(
                np.zeros(counts.shape), counts,
                counts.shape
            ) + idx

            rows, cols = mutable_idx[rand_idx, 0], mutable_idx[rand_idx, 1]
            cols = mutable_vertices[cols]
            self._dna[rows, cols] -= 1

    def _underflow_mutation(self,
        capacity_matrix: NDArray,
        mutable: NDArray
    ):
        underflow = (self.inflow < self.outflow)
        mutable_vertices = np.argwhere(underflow & mutable)[:, 0]

        # slight opposite to the method above
        # we first increment the inflow
        mutable_idx = np.argwhere(
            (capacity_matrix[:, mutable_vertices] != 0) &
            (self._dna[:, mutable_vertices] < capacity_matrix[:, mutable_vertices])
        )
        unique, idx, counts = np.unique(
            mutable_idx[:, 1],
            return_counts=True,
            return_index=True,
        )
        if unique.shape[0]:
            rand_idx = _np_rng.integers(
                np.zeros(counts.shape), counts,
                counts.shape
            ) + idx

            rows, cols = mutable_idx[rand_idx, 0], mutable_idx[rand_idx, 1]
            cols = mutable_vertices[cols]
            self._dna[rows, cols] += 1

        mutable_vertices = np.delete(mutable_vertices, unique)
        if mutable_vertices.shape[0]:
            # if inflow can't be incremented
            # decrement outflow instead
            mutable_idx = np.argwhere(
                (capacity_matrix[mutable_vertices] != 0) &
                (self._dna[mutable_vertices] > 0)
            )
            unique, idx, counts = np.unique(
                mutable_idx[:, 0],
                return_counts=True,
                return_index=True,
            )
            rand_idx = _np_rng.integers(
                np.zeros(counts.shape), counts,
                counts.shape
            ) + idx

            rows, cols = mutable_idx[rand_idx, 0], mutable_idx[rand_idx, 1]
            rows = mutable_vertices[rows]
            self._dna[rows, cols] -= 1

    def _balanced_mutation(self,
        capacity_matrix: NDArray,
        mutable: NDArray
    ):
        balanced = (self.inflow == self.outflow) & mutable
        mutable_vertices = np.argwhere(balanced & mutable)[:, 0]

        # go over every row and col (outflow and inflow)
        # and check which edge has not reached capacity yet
        mutable_out_idx = np.argwhere(
            self._dna[mutable_vertices] < capacity_matrix[mutable_vertices]
        )
        mutable_in_idx = np.argwhere(
            self._dna[:, mutable_vertices] < capacity_matrix[:, mutable_vertices]
        )

        if not mutable_out_idx.shape[0] or not mutable_in_idx.shape[0]:
            # if all has reached capacity then ignore
            return
        
        # begin choosing a random outgoing edge and incoming edge
        # to increment flow
        _, out_idx, counts_1 = np.unique(
            mutable_out_idx[:, 0],
            return_counts=True,
            return_index=True,
        )
        _, in_idx, counts_2 = np.unique(
            mutable_in_idx[:, 1],
            return_counts=True,
            return_index=True,
        )

        rand_out_idx = _np_rng.integers(
            np.zeros(counts_1.shape), counts_1,
            counts_1.shape
        ) + out_idx
        rand_in_idx = _np_rng.integers(
            np.zeros(counts_2.shape), counts_2,
            counts_2.shape
        ) + in_idx

        rows_1, cols_1 = mutable_out_idx[rand_out_idx, 0], mutable_out_idx[rand_out_idx, 1]
        rows_1 = mutable_vertices[rows_1]

        rows_2, cols_2 = mutable_in_idx[rand_in_idx, 0], mutable_in_idx[rand_in_idx, 1]
        cols_2 = mutable_vertices[cols_2]
        self._dna[rows_1, cols_1] += 1
        self._dna[rows_2, cols_2] += 1


    def check_balanced(self) -> NDArray:
        """
        Check the balance of graph's vertices

        A vertex is said to be balance if its inflow is equal to its outflow.
        Inflow implies the incoming flow to a vertex. Outflow implies the outcoming
        flow at a vertex.

        :return: A numpy array of bool where the element at index
            `i` is `True` iff the `i`th vertex is balance.
        :rtype: NDArray
        """
        return self.inflow == self.outflow
    
    def fitness(self, maximal_capacity: int):
        """
        Calculate the fitness of an individual
        
        :param int maximal_capacity: A value shows the maximal flow
            of given graph assuming that all intermediate edges'
            capacity allows for such flow.
        """
        balance_matrix = self.check_balanced()
        excess_flow = abs(self.inflow[1:-1] - self.outflow[1:-1]).sum()
        total_flow = self.outflow.sum()
        if total_flow == 0:
            self.fitness_score = 0
        else:
            self.fitness_score = (
                balance_matrix.sum() / balance_matrix.shape[0]
                - excess_flow / total_flow
                + min(self.inflow[-1], self.outflow[0]) / maximal_capacity
            )

    def crossover(self,
        partner: 'Individual',
        function_index: Literal[1, 2, 3] = 1
    ) -> 'Individual':
        """
        Apply crossover between this object and another Individual.
        See each respective crossover function for more details.

        :param Individual partner: An Individual to perform crossover with.
        :param Literal[1, 2, 3] function_index: The crossover function to use,
            should be a number between 1 and 3. By default, use crossover function 1.
        """
        if function_index == 1:
            return self.c1(partner)
        elif function_index == 2:
            return self.c2(partner)
        elif function_index == 3:
            return self.c3(partner)
        else:
            raise ValueError(f"There are only 3 crossover function! {function_index=} was passed")
        
    def mutate(self,
        capacity_matrix: NDArray,
        mutation_rate: float
    ) -> None:
        """
        Apply mutation over an individual. Mutation tries to adjust
        the solution for more optimisation. Mutation is done as follows:
        1. Start at the first vertex.
        2. Determine whether or not to mutate this vertex,
            if no mutation, go to step 4.
        3. If the vertex is not balance, try to change the flow at
            the vertex by increment/decrement of 1. For better optimisation,
            we always prioritise incrementing the flow first. If no outflowing edge
            can be incremented due to reaching the flow capacity. We will try to
            decrement an inflowing edge.
        4. Repeat step 2 with the next vertex.

        :param NDArray capacity_matrix: The capacity matrix of the network.
        :param float mutation_rate: A float representing the rate of mutation. Must be between 0 and 1
        """
        proba = _np_rng.random(self._dna.shape[0])
        needs_mutation = proba <= mutation_rate
        needs_mutation[[0, -1]] = False

        self._overflow_mutation(capacity_matrix, needs_mutation)
        self._underflow_mutation(capacity_matrix, needs_mutation)
        self._balanced_mutation(capacity_matrix, needs_mutation)
        
        self._update_flowrate()
    
    def c1(self, partner: 'Individual') -> 'Individual':
        """Perform crossover on two individuals
        When doing crossover, we go through each vertices and decide whether we would take the partner's vertex
        instead. Decision is based on the balance of that vertex as well as its flow.
        When choosing between two vertices, we prioritise taking the more balanced vertex, that is the vertex
        with less excess flow (excess flow is difference between inflow and outflow).
        In the case where both vertices have the same balance, we prioritise taking the vertex with higher flow.
        This comparision can be done on either the inflow or the outflow. This implementation chooses to use the
        inflow for comparision.

        When a vertex is chosen to be encoded to the child, every edges that connects the vertex is written into
        the child's DNA. In the case where the edge has already been written to beforehand. We take the average
        between the old and new flow.
        partner: an Individual to perform crossover with
        """
        new_dna = np.zeros(self._dna.shape)
        row_assigned = np.zeros(self._dna.shape, dtype=bool)
        col_assigned = np.zeros(self._dna.shape, dtype=bool)
        
        for i in range(1, len(self._dna) - 1):
            # note: order in which these conditions are checked MATTERS!!!
            # DO NOT TRY TO REFACTOR THIS CONDITION CHECKING!!!
            if (exA := abs(self.inflow[i] - self.outflow[i])) < (exB := abs(partner.inflow[i] - partner.outflow[i])):
                chosen = self._dna
            elif exA > exB:
                chosen = partner._dna
            elif self.inflow[i] > partner.inflow[i]:
                chosen = self._dna
            elif self.inflow[i] < partner.inflow[i]:
                chosen = partner._dna
            else:
                chosen = self._dna if rand.random() < 0.5 else partner._dna
            
            #row assignment (outflow)
            row_assigned[i] = True
            new_dna[i, ~col_assigned] = chosen[i, ~col_assigned]
            new_dna[i, col_assigned] = round((new_dna[i, col_assigned] + chosen[i, col_assigned]) // 2)
            # for j in range(1, len(self._dna)):
            #     if col_assigned[j]:
            #         new_dna[i][j] = (new_dna[i][j] + chosen[i][j]) // 2
            #     else:
            #         new_dna[i][j] = chosen[i][j]

            # col assigment (inflow)
            col_assigned[i] = True
            new_dna[:, ~row_assigned] = chosen[:, ~row_assigned]
            new_dna[:, row_assigned] = round((new_dna[:, row_assigned] + chosen[:, row_assigned]) // 2)
            # for j in range(len(self._dna) - 1):
            #     if j == 0 or not row_assigned[j]:
            #         new_dna[j][i] = chosen[j][i]
            #     else:
            #         new_dna[j][i] = (new_dna[j][i] + chosen[j][i]) // 2
        child = Individual()
        child.dna = new_dna
        return child
    def c2(self, partner: 'Individual') -> 'Individual':
        """
        Perform one-point crossover.
        Start by choosing a random vertex i (excluding the sink vertex). Then, child inherits every vertices
        and outflowing edges starting from the 0th to the ith vertex from this instance. For every other
        outflowing edges and vertices from (i+1)th to the sink vertex, child inherits from the partner Individual.

        partner: an Individual to perform crossover with
        """
        new_dna = np.array(self._dna)
        vertices_from_self = rand.randint(0, len(self.dna) - 2)

        new_dna[vertices_from_self + 1:] = partner.dna[vertices_from_self + 1:]

        child = Individual()
        child.dna = new_dna
        return child
    def c3(self, partner: 'Individual') -> 'Individual':
        """
        Perform two-point crossover.
        Start by choosing two random vertex i and j. Then, child inherits every vertices and outflowing edges
        starting from the ith to the jth vertex from this instance. For every other outflowing edges and vertices
        from 0th to (i-1)th and the (j+1)th to the sink vertex, child inherits from the partner Individual.

        partner: an Individual to perform crossover with
        """
        new_dna = np.zeros(self._dna.shape, dtype=self._dna.dtype)
        endpoints = rand.sample(range(len(self.dna) - 1), k=2)
        if endpoints[0] > endpoints[1]: endpoints[::-1]

        new_dna[endpoints[0]:endpoints[1] + 1] += self._dna[endpoints[0]:endpoints[1] + 1]
        new_dna[:endpoints[0]] += partner._dna[:endpoints[0]]
        new_dna[endpoints[1] + 1:] += partner._dna[endpoints[1] + 1:]

        child = Individual()
        child.dna = new_dna
        return child
    
    
    def __str__(self) -> str:
        return str(self._dna)