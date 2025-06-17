import random as rand
import numpy as np

from maxflow_ga import _np_rng


from numpy.typing import NDArray


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
    An individual is a feasible solution of the max flow problem.
    Its DNA is defined to be a network that has been filled in.

    :param dna: The dna of the Individual. Internally, the DNA is
        a 2D-square matrix, similar to the capacity matrix.
        Each element shows the flow of a solution, an integer between
        0 and the corresponding flow capacity.
    :type dna: numpy.NDArray

    :param fitness_score: The fitness score of the Individual.
    :type fitness_score: float
    """
    __slots__ = "_dna", "_inflow", "_outflow", "fitness_score"
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

        :param capacity_matrix: The capacity matrix of the network.
            This is `None` by default. In such case, The DNA is not initialized.
        :type capacity_matrix: NDArray or None
        """
        if not capacity_matrix is None:
            self.dna = random_flow(capacity_matrix)
    
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

        ## Examples and code
        Consider the given solution to a graph containing 4 nodes
        ```
           [[0, 1, 0, 2],
            [0, 0, 2, 0],
            [0, 3, 0, 4],
            [0, 0, 0, 0]]
        ```
        The inflow is the sum of values along each column. For example, the inflow
        of the 1st vertext (column at index 1) is 4.

        Similarly, the outflow is the sum of values along each row. For example,
        the outflow of the first vertex (row at index 1) is 2.
        ```
        >>> from maxflow_ga.Individual import Individual
        >>> ind = Individual()
        >>> ind._dna = np.array([[0, 1, 0, 2], [0, 0, 2, 0], [0, 3, 0, 4], [0, 0, 0, 0]])
        >>> ind._update_flowrate()
        >>> ind._inflow
        array([0, 4, 2, 6])
        >>> ind._outflow
        array([3, 2, 7, 0])
        ```
        """
        self._inflow = self._dna.sum(axis=0)
        self._outflow = self._dna.sum(axis=1)

    def _overflow_mutation(self,
        capacity_matrix: NDArray,
        mutable: NDArray
    ):
        overflow = (self._inflow > self._outflow)
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
        underflow = (self._inflow < self._outflow)
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
        balanced = (self._inflow == self._outflow) & mutable
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
        Check the balance of solution's vertices.

        A vertex is said to be balance if its inflow is equal to its outflow.

        ## Examples
        Consider the given solution to a graph containing 4 nodes
        ```
           [[0, 1, 0, 2],
            [0, 0, 2, 0],
            [0, 1, 0, 4],
            [0, 0, 0, 0]]
        ```
        The 1st vertex has outflow (row index 1) of 2 and inflow (column index 1)
        of 2. Thus, this is a balance vertex.

        Whereas, the 2nd vertex has outflow of 5 and inflow of 2. This is not a
        balanced vertex.

        :return: A numpy array of bool where the element at index
            `i` is `True` iff the `i`th vertex is balance.
        :rtype: NDArray
        """
        bal_mat = self._inflow == self._outflow
        bal_mat[[0, -1]] = True
        return bal_mat
    
    def fitness(self, maximal_capacity: int):
        """
        Calculate the fitness of an individual.
        
        :param int maximal_capacity: A value shows the maximal flow
            of given graph assuming that all intermediate edges'
            capacity allows for such flow.
        """
        balance_matrix = self.check_balanced()
        excess_flow = abs(self._inflow[1:-1] - self._outflow[1:-1]).sum()
        total_flow = self._outflow.sum()
        if total_flow == 0:
            self.fitness_score = 0
        else:
            self.fitness_score = (
                balance_matrix.sum() / balance_matrix.shape[0]
                - excess_flow / total_flow
                + min(self._inflow[-1], self._outflow[0]) / maximal_capacity
            )
        
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
    
    def flow_conservation_crossover(self,
        partner: 'Individual'
    ) -> NDArray:
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
        new_dna = np.zeros(self._dna.shape, dtype=self._dna.dtype)
        row_assigned = np.zeros(self._dna.shape[0], dtype=bool)
        col_assigned = np.zeros(self._dna.shape[0], dtype=bool)
        
        for i in range(1, len(self._dna) - 1):
            # note: order in which these conditions are checked MATTERS!!!
            # DO NOT TRY TO REFACTOR THIS CONDITION CHECKING!!!
            if (exA := abs(self._inflow[i] - self._outflow[i])) < (exB := abs(partner._inflow[i] - partner._outflow[i])):
                chosen = self._dna
            elif exA > exB:
                chosen = partner._dna
            elif self._inflow[i] > partner._inflow[i]:
                chosen = self._dna
            elif self._inflow[i] < partner._inflow[i]:
                chosen = partner._dna
            else:
                chosen = self._dna if rand.random() < 0.5 else partner._dna
            
            #row assignment (outflow)
            row_assigned[i] = True
            new_dna[i, ~col_assigned] = chosen[i, ~col_assigned]
            new_dna[i, col_assigned] = (new_dna[i, col_assigned] + chosen[i, col_assigned]) // 2

            # col assigment (inflow)
            col_assigned[i] = True
            new_dna[~row_assigned, i] = chosen[~row_assigned, i]
            new_dna[row_assigned, i] = (new_dna[row_assigned, i] + chosen[row_assigned, i]) // 2
        return new_dna
    
    def one_point_crossover(self,
        partner: 'Individual'
    ) -> NDArray:
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

        return new_dna
    
    def two_point_crossover(self,
        partner: 'Individual'
    ) -> NDArray:
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

        return new_dna
    
    def __repr__(self) -> str:
        return f"""  Fitness: {self.fitness_score}
  Balanced vertices: {self.check_balanced().sum()}/{self._dna.shape[0]}
  Maximum flow: {self._dna[:, -1].sum()}
  Flow matrix:
{self._dna}"""