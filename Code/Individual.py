from copy import deepcopy
import random as rand

class Individual():
    """Network Flow - Max Flow Optimisation\n
    An individual is a feasible solution of the problem. It's DNA is defined to be a graph that has\
    been filled in.
    """
    __slots__ = "_dna", "inflow", "outflow", "fitness_score"
    def __init__(self, capacity_matrix: list[list[int]] | None = None) -> None:
        """
        When creating an individual, a capacity matrix is needed to actually generate a random DNA.\
        If no capacity matrix is given then an Individual is created without declaring any of its\
        attribute.\n\n

        Internally, the DNA is also 2D-square matrix, similar to the capacity matrix. Each element shows\
        the flow of a solution. Each element must be between 0 and its flow capacity.\n

        capacity_matrix: (Optional) A capacity matrix is a 2D-square matrix. The matrix shows the flow\
            capacity from one vertex to another. For any element e_{ij}, it represents the flow from the\
            vertex indexed at i to the vertex indexed at j.\
            For example, at the element at index (0, 1) has a value of 10. That means the flow from the 0th\
            vertex to the 1st vertex is 10.
        """
        if capacity_matrix:
            self._dna: list[list[int]] = Individual.random_flow(capacity_matrix)
            self.UpdateFlowrate()
    
    #getter and setter for dna attribute, note, every time a new dna is assigned, we must update the flowrate
    @property
    def dna(self) -> list[list[int]]:
        if not hasattr(self, "_dna"):
            raise AttributeError("'dna' attribute has not been initialised")
        return deepcopy(self._dna)
    @dna.setter
    def dna(self, new_dna: list[list[int]]) -> None:
        self._dna = new_dna
        self.UpdateFlowrate()

    def random_flow(capacity_matrix: list[list[int]]) -> list[list[int]]:
        """A class method used to generate a random solution, given the capacity matrix\n

        capacity_matrix: A capacity matrix is a 2D-square matrix. The matrix shows the flow\
            capacity from one vertex to another. For any element e_{ij}, it represents the flow from the\
            vertex indexed at i to the vertex indexed at j.\
            For example, at the element at index (0, 1) has a value of 10. That means the flow from the 0th\
            vertex to the 1st vertex is 10.
        """
        result: list[list[int]] = [[0] * len(capacity_matrix) for _ in range(len(capacity_matrix))]
        for i, row in enumerate(capacity_matrix):
            for j, val in enumerate(row):
                result[i][j] = rand.randint(0, val)
        return result
    def UpdateFlowrate(self) -> None:
        """Update the Inflows and Outflows of the solution\n
        In simple words, inflow implies the incoming flow to a vertex. Similarly, outflow implies the\
        outcoming flow at a vertex.\n

        For example: Consider the given solution to a graph containing 4 nodes\n
        0 1 0 2\n
        0 0 2 0\n
        0 3 0 4\n
        0 0 0 0\n
        At the 1st vertex, its inflow is 4 (1 + 3), represented by the sum of elements in the second\
        column. Its outflow is 2, represented by the sum of elements in the second row.\n
        At the 0th vertex, its inflow is 0, and it's outflow is 3 (1 + 2).\n
        At the 2nd vertex, its inflow is 2 and its outflow is 4.\n
        At the 3rd vertex, its inflow is 6 and its outflow is 0.\n
        Note that the source vertex (0th vertex) has no inflow, and the sink vertex (3rd vertex) has\
        no outflow.
        """
        self.inflow = [0] * len(self.dna)
        self.outflow = [0] * len(self.dna)
        for i, row in enumerate(self.dna):
            for j, val in enumerate(row):
                self.outflow[i] += val
                self.inflow[j] += val
    def CheckBalanced(self) -> list[bool]:
        """Check the balance of graph's vertices\n
        A vertex is said to be balance if its inflow is equal to its outflow.\n
        Inflow implies the incoming flow to a vertex. Outflow implies the\
        outcoming flow at a vertex.\n
        Function returns a list of booleans, where at each index i, the value is True if the ith vertex\
        is balance, False otherwise.
        """
        isBalanced: list[bool] = [False] * len(self.dna)
        for i, val in enumerate(self.inflow):
            if self.outflow[i] == val:
                isBalanced[i] = True
        isBalanced[0], isBalanced[len(self.dna) - 1] = True, True
        return isBalanced
    def fitness(self, maximal_capacity: int) -> None:
        """Calculate the fitness of an individual\n
        maximal_capacity: a value shows the maximal flow of given graph assuming that all\
            intermediate edges' capacity allows for such flow
        """
        balance_matrix: list[bool] = self.CheckBalanced()
        excess_flow: int = sum([abs(self.inflow[i] - self.outflow[i]) for i in range(1, len(self.inflow) - 1)])
        total_flow: int = sum(self.outflow)
        if total_flow == 0:
            self.fitness_score = 0
        else:
            self.fitness_score = sum(balance_matrix) / len(balance_matrix)\
                                - excess_flow / total_flow\
                                + min(self.inflow[-1], self.outflow[0]) / maximal_capacity
        if self.fitness_score < 0:
            self.fitness_score = 0
    def crossover(self, partner: 'Individual', function_index: int = 1) -> 'Individual':
        """
        Apply crossover between this object and another Individual. See each respective crossover
        function for more details.

        partner: an Individual to perform crossover with
        function_index: the crossover function to use, should be a number between 1 and 3. By default,
        use crossover function 1.
        """
        if function_index == 1:
            return self.c1(partner)
        elif function_index == 2:
            return self.c2(partner)
        elif function_index == 3:
            return self.c3(partner)
        else:
            raise ValueError(f"There are only 3 crossover function! {function_index=} was passed")
    def mutate(self, capacity_matrix: list[list[int]], mutation_rate: float) -> None:
        """
        Apply mutation over an individual. Mutation tries to adjust the solution for more optimisation\n
        Mutation is done as follows:\n
        1. Start at the vertex indexed 1.\n
        2. Determine whether or not to mutate this vertex, if no mutation, go to step 4.\n
        3. If the vertex is not balance, try to change the flow at the vertex by increment/decrement of 1.\
            For better optimisation, we always prioritise incrementing the flow first. If no outflowing edge\
            can be incremented due to reaching the flow capacity. We will try to decrement an inflowing edge.\n
            In case when the vertex is balance, raise NotImplementedError().\n
        4. Repeat step 2 with the next vertex.

        capacity_matrix: A capacity matrix is a 2D-square matrix. The matrix shows the flow\
            capacity from one vertex to another. For any element e_{ij}, it represents the flow from the\
            vertex indexed at i to the vertex indexed at j.\
            For example, at the element at index (0, 1) has a value of 10. That means the flow from the 0th\
            vertex to the 1st vertex is 10.
        mutation_rate: a float representing the rate of mutation. Must be between 0 and 1
        """
        mutation_rate = mutation_rate
        mutated: bool
        # iterating through every vertices except the source and sink
        for i in range(1, len(self.dna) - 1):
            if rand.random() > mutation_rate:
                continue
            mutated = False
            # case where there's more flow coming in => prioritise incrementing outflow
            if self.inflow[i] > self.outflow[i]:
                # iterates through the row at index i
                # basically iterating through the outcoming edges
                edges = []
                for j, val in enumerate(capacity_matrix[i]):
                    if val != 0 and self.dna[i][j] < val:
                        # self.dna[i][j] += 1
                        # mutated = True
                        # break
                        edges.append((i, j))
                if edges:
                    random_edge = rand.choice(edges)
                    self.dna[random_edge[0]][random_edge[1]] += 1
                    continue
                # if no mutation yet, try to decrement the inflow
                # this is done by iterating through the column at index i
                # basically iterating through the incoming edge
                for j, _ in enumerate(capacity_matrix):
                    if capacity_matrix[j][i] != 0 and self.dna[j][i] > 0:
                        # self.dna[j][i] -= 1
                        # break
                        edges.append((j, i))
                random_edge = rand.choice(edges)
                self.dna[random_edge[0]][random_edge[1]] -= 1
            # case where there's more flow going out -> prioritise incrementing the inflow
            elif self.inflow[i] < self.outflow[i]:
                # iterates through the column at index i
                # basically iterating through the incoming edge
                edges = []
                for j, _ in enumerate(capacity_matrix):
                    if capacity_matrix[j][i] != 0 and self.dna[j][i] < capacity_matrix[j][i]:
                        # self.dna[j][i] += 1
                        # mutated = True
                        # break
                        edges.append((j, i))
                if edges:
                    random_edge = rand.choice(edges)
                    self.dna[random_edge[0]][random_edge[1]] += 1
                    continue
                # if no mutation yet, try to decrement the outflow
                # iterates through the row at index i
                # basically iterating through the outcoming edges
                for j, val in enumerate(capacity_matrix[i]):
                    if val != 0 and self.dna[i][j] > 0:
                        # self.dna[i][j] -= 1
                        # break
                        edges.append((i, j))
                random_edge = rand.choice(edges)
                self.dna[random_edge[0]][random_edge[1]] -= 1
            # case where the node is balanced => try to increment both outflow and inflow
            # if either direction is not possible, don't do it!!
            else:
                indices_pool: list[tuple[int, int]] = []
                # iterating through the row at index i (basically the outcoming edges)
                # and see which edges can be incremented, append it to a pool for random selection
                # afterwards choose a random edge to increment
                for j, val in enumerate(capacity_matrix[i]):
                    if self.dna[i][j] < val:
                        indices_pool.append((i, j))
                if not indices_pool:
                    continue
                temp_indices1 = rand.choice(indices_pool)
                indices_pool = []
                # do the same thing but with elments at column i instead (basically the incoming edges)
                for j, row in enumerate(capacity_matrix):
                    if self.dna[j][i] < row[i]:
                        indices_pool.append((j, i))
                if not indices_pool:
                    continue
                temp_indices2 = rand.choice(indices_pool)
                self.dna[temp_indices1[0]][temp_indices1[1]] += 1
                self.dna[temp_indices2[0]][temp_indices2[1]] += 1
        self.UpdateFlowrate()
    def c1(self, partner: 'Individual') -> 'Individual':
        """Perform crossover on two individuals\n
        When doing crossover, we go through each vertices and decide whether we would take the partner's vertex\
        instead. Decision is based on the balance of that vertex as well as its flow.\n
        When choosing between two vertices, we prioritise taking the more balanced vertex, that is the vertex\
        with less excess flow (excess flow is difference between inflow and outflow).\n
        In the case where both vertices have the same balance, we prioritise taking the vertex with higher flow.\
        This comparision can be done on either the inflow or the outflow. This implementation chooses to use the\
        inflow for comparision.\n\n

        When a vertex is chosen to be encoded to the child, every edges that connects the vertex is written into\
        the child's DNA. In the case where the edge has already been written to beforehand. We take the average\
        between the old and new flow.
        partner: an Individual to perform crossover with
        """
        new_dna: list[list[int]] = [[0] * len(self.dna) for _ in range(len(self.dna))]
        chosen: 'Individual'
        ColAssigned: dict[int: bool] = {i: False for i in range(1, len(self.dna))}
        RowAssigned: dict[int: bool] = {i: False for i in range(1, len(self.dna))}
        for i in range(1, len(self.dna) - 1):
            # note: order in which these conditions are checked MATTERS!!!
            # DO NOT TRY TO REFACTOR THIS CONDITION CHECKING!!!
            if (exA := abs(self.inflow[i] - self.outflow[i])) < (exB := abs(partner.inflow[i] - partner.outflow[i])):
                chosen = self.dna
            elif exA > exB:
                chosen = partner.dna
            elif self.inflow[i] > partner.inflow[i]:
                chosen = self.dna
            elif self.inflow[i] < partner.inflow[i]:
                chosen = partner.dna
            else:
                chosen = self.dna if rand.random() < 0.5 else partner.dna
            
            #row assignment (outflow)
            RowAssigned[i] = True
            for j in range(1, len(self.dna)):
                if ColAssigned[j]:
                    new_dna[i][j] = (new_dna[i][j] + chosen[i][j]) // 2
                else:
                    new_dna[i][j] = chosen[i][j]
            #col assigment (inflow)
            ColAssigned[i] = True
            for j in range(len(self.dna) - 1):
                if j == 0 or not RowAssigned[j]:
                    new_dna[j][i] = chosen[j][i]
                else:
                    new_dna[j][i] = (new_dna[j][i] + chosen[j][i]) // 2
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
        new_dna: list[list[int]] = []
        vertices_from_self: int = rand.randint(0, len(self.dna) - 2)
        for i, row in enumerate(self.dna):
            if i <= vertices_from_self:
                new_dna.append(row)
            else:
                new_dna.append(partner.dna[i])

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
        new_dna: list[list[int]] = []
        endpoints: list[int] = rand.sample(range(len(self.dna) - 1), k=2)
        for i, row in enumerate(self.dna):
            if endpoints[0] <= i and i <= endpoints[1]:
                new_dna.append(row)
            else:
                new_dna.append(partner.dna[i])
        child = Individual()
        child.dna = new_dna
        return child
    def __str__(self) -> str:
        result: str = ""
        for row in self.dna:
            result += str(row) + "\n"
        return result
