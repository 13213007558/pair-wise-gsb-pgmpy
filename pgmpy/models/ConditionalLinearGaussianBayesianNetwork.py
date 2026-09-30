from __future__ import annotations

from collections.abc import Hashable, Iterable

import networkx as nx

from pgmpy.base import DAG
from pgmpy.factors.continuous import ConditionalLinearGaussianCPD, LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD


class ConditionalLinearGaussianBayesianNetwork(DAG):
    """
    Bayesian network containing discrete and continuous variables.

    Discrete variables may only have discrete parents. Each continuous variable
    uses a :class:`ConditionalLinearGaussianCPD`, allowing its intercept, linear
    coefficients, and noise standard deviation to depend on its discrete
    parents.

    Parameters
    ----------
    ebunch : input graph, optional
        Edges used to initialize the DAG.
    discrete_variables : iterable[Hashable], optional
        Nodes that should be treated as discrete. Every other graph node is
        treated as continuous.
    """

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        discrete_variables: Iterable[Hashable] | None = None,
    ) -> None:
        super().__init__(ebunch=ebunch)
        self.discrete_variables = list(discrete_variables or [])
        unknown_discrete = set(self.discrete_variables) - set(self.nodes())
        if unknown_discrete:
            raise ValueError(f"Discrete variables not in the model: {sorted(unknown_discrete, key=str)}")
        if len(self.discrete_variables) != len(set(self.discrete_variables)):
            raise ValueError("Discrete variables must be unique.")
        self.cpds = []

    @property
    def continuous_variables(self) -> list[Hashable]:
        return [node for node in self.nodes() if node not in self.discrete_variables]

    @property
    def states(self) -> dict[Hashable, list]:
        state_names = {}
        for cpd in self.cpds:
            if isinstance(cpd, TabularCPD):
                state_names.update(cpd.state_names)
            elif isinstance(cpd, ConditionalLinearGaussianCPD):
                state_names.update(cpd.state_names)
        return state_names

    def add_cpds(self, *cpds: TabularCPD | ConditionalLinearGaussianCPD | LinearGaussianCPD) -> None:
        """Add a discrete table CPD or conditional linear Gaussian CPD."""
        for cpd in cpds:
            if isinstance(cpd, LinearGaussianCPD) and not isinstance(cpd, ConditionalLinearGaussianCPD):
                parents = self.get_parents(cpd.variable) if cpd.variable in self.nodes() else []
                if set(parents) & set(self.discrete_variables):
                    raise ValueError("Continuous nodes with discrete parents require ConditionalLinearGaussianCPD.")
                cpd = ConditionalLinearGaussianCPD(
                    variable=cpd.variable,
                    discrete_evidence=[],
                    continuous_evidence=list(cpd.evidence),
                    parameters={(): {"beta": cpd.beta, "std": cpd.std}},
                )

            if not isinstance(cpd, TabularCPD | ConditionalLinearGaussianCPD):
                raise ValueError("Only TabularCPD and ConditionalLinearGaussianCPD can be added.")

            if cpd.variable not in self.nodes():
                raise ValueError(f"CPD defined on variable not in the model: {cpd.variable}")

            for index, existing_cpd in enumerate(self.cpds):
                if existing_cpd.variable == cpd.variable:
                    self.cpds[index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(
        self, node: Hashable | None = None
    ) -> TabularCPD | ConditionalLinearGaussianCPD | list[TabularCPD | ConditionalLinearGaussianCPD] | None:
        if node is None:
            return self.cpds
        if node not in self.nodes():
            raise ValueError(f"Node not present in the graph: {node}")
        for cpd in self.cpds:
            if cpd.variable == node:
                return cpd
        return None

    def remove_cpds(self, *cpds: TabularCPD | ConditionalLinearGaussianCPD | Hashable) -> None:
        for cpd in cpds:
            if not isinstance(cpd, TabularCPD | ConditionalLinearGaussianCPD):
                cpd = self.get_cpds(cpd)
            self.cpds.remove(cpd)

    def copy(self) -> ConditionalLinearGaussianBayesianNetwork:
        model_copy = ConditionalLinearGaussianBayesianNetwork(
            self.edges(),
            discrete_variables=list(self.discrete_variables),
        )
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        return model_copy

    def check_model(self) -> bool:
        """Check graph type restrictions, CPD coverage, parent sets, and state consistency."""
        if not nx.is_directed_acyclic_graph(self):
            raise ValueError("The graph must be a directed acyclic graph.")

        for node in self.nodes():
            parents = list(self.get_parents(node))
            is_discrete = node in self.discrete_variables
            if is_discrete and any(parent not in self.discrete_variables for parent in parents):
                raise ValueError(f"Discrete node {node} can only have discrete parents.")

            cpd = self.get_cpds(node)
            if cpd is None:
                raise ValueError(f"No CPD associated with {node}")

            if is_discrete:
                if not isinstance(cpd, TabularCPD):
                    raise ValueError(f"Discrete node {node} must use TabularCPD.")
                if set(cpd.variables[1:]) != set(parents):
                    raise ValueError(f"CPD associated with {node} does not have the proper parents.")
                if not cpd.is_valid_cpd():
                    raise ValueError(f"CPD for {node} is not normalized.")
            else:
                if not isinstance(cpd, ConditionalLinearGaussianCPD):
                    raise ValueError(f"Continuous node {node} must use ConditionalLinearGaussianCPD.")
                discrete_parents = set(parents).intersection(self.discrete_variables)
                continuous_parents = set(parents).difference(self.discrete_variables)
                if set(cpd.discrete_evidence) != discrete_parents:
                    raise ValueError(
                        f"Discrete parents of {node} do not match its CPD: "
                        f"{set(cpd.discrete_evidence)} != {discrete_parents}"
                    )
                if set(cpd.continuous_evidence) != continuous_parents:
                    raise ValueError(f"Continuous parents of {node} do not match its CPD.")

        for cpd in self.cpds:
            if isinstance(cpd, TabularCPD):
                for parent in cpd.variables[1:]:
                    parent_cpd = self.get_cpds(parent)
                    if parent_cpd.cardinality[0] != cpd.state_names[parent].__len__():
                        raise ValueError(f"The cardinality of {parent} does not match in its child CPD.")
                    if parent_cpd.state_names[parent] != cpd.state_names[parent]:
                        raise ValueError(f"The state names of {parent} do not match in its child CPD.")
            else:
                for parent in cpd.discrete_evidence:
                    parent_cpd = self.get_cpds(parent)
                    if parent_cpd.state_names[parent] != cpd.state_names[parent]:
                        raise ValueError(f"The state names of {parent} do not match in its child CPD.")

        return True
