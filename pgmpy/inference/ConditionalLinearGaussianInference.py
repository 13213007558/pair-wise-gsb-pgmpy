from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from numbers import Real

import networkx as nx
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from pgmpy.factors.continuous import GaussianMixture
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference.base import Inference
from pgmpy.models import ConditionalLinearGaussianBayesianNetwork, LinearGaussianBayesianNetwork


class ConditionalLinearGaussianInference(Inference):
    """
    Exact inference for conditional linear Gaussian Bayesian networks.

    The algorithm enumerates complete discrete configurations. For each one it
    conditions the induced joint Gaussian on the continuous evidence. Posterior
    Gaussian components are returned explicitly, so multimodal posteriors are
    never replaced by a single moment-matched Gaussian.
    """

    def __init__(self, model: ConditionalLinearGaussianBayesianNetwork):
        if not isinstance(model, ConditionalLinearGaussianBayesianNetwork):
            raise TypeError("model must be a ConditionalLinearGaussianBayesianNetwork.")
        super().__init__(model)

    def query(
        self,
        variables: list[Hashable],
        evidence: dict[Hashable, Hashable | float] | None = None,
        joint: bool = True,
    ):
        """
        Compute an exact posterior query.

        Parameters
        ----------
        variables : list
            Queried discrete and/or continuous variables.
        evidence : dict, optional
            Observed state names for discrete variables and numeric values for
            continuous variables.
        joint : bool, default=True
            If True, return the joint posterior. A query containing continuous
            variables is a :class:`GaussianMixture`; a discrete-only query is a
            :class:`DiscreteFactor`. If False, return a dict of marginals.
        """
        evidence = dict(evidence or {})
        variables = list(variables)

        if not variables:
            raise ValueError("variables must contain at least one variable.")
        unknown = (set(variables) | set(evidence)) - set(self.model.nodes())
        if unknown:
            raise ValueError(f"Variables not in the model: {sorted(unknown, key=str)}")
        overlap = set(variables) & set(evidence)
        if overlap:
            raise ValueError(f"Variables cannot be both queried and observed: {sorted(overlap, key=str)}")

        for variable, value in evidence.items():
            if variable in self.model.discrete_variables:
                if value not in self.model.states[variable]:
                    raise ValueError(f"Unknown state {value!r} for discrete variable {variable!r}.")
            elif not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"Evidence for continuous variable {variable!r} must be a finite number.")

        discrete_variables = self.model.discrete_variables
        query_discrete = [var for var in variables if var in self.model.discrete_variables]
        query_continuous = [var for var in variables if var not in self.model.discrete_variables]
        observed_discrete = {var: value for var, value in evidence.items() if var in discrete_variables}
        observed_continuous = {var: float(value) for var, value in evidence.items() if var not in discrete_variables}

        component_records = []
        for assignment in self._enumerate_discrete_assignments():
            if any(assignment[var] != value for var, value in observed_discrete.items()):
                continue

            discrete_probability = self._discrete_probability(assignment)
            if discrete_probability <= 0:
                continue
            log_weight = np.log(discrete_probability)
            mean, covariance, order = self._joint_gaussian(assignment)
            index_by_variable = {var: index for index, var in enumerate(order)}

            if observed_continuous:
                observed_order = [var for var in order if var in observed_continuous]
                observed_index = [index_by_variable[var] for var in observed_order]
                observed_values = np.array([observed_continuous[var] for var in observed_order])
                observed_mean = mean[observed_index]
                observed_covariance = covariance[np.ix_(observed_index, observed_index)]
                log_weight += multivariate_normal.logpdf(
                    observed_values,
                    mean=observed_mean,
                    cov=observed_covariance,
                    allow_singular=True,
                )

                latent_order = [var for var in order if var not in observed_continuous]
                latent_index = [index_by_variable[var] for var in latent_order]
                latent_observed_covariance = covariance[np.ix_(latent_index, observed_index)]
                conditional_mean = mean[latent_index] + latent_observed_covariance @ np.linalg.solve(
                    observed_covariance,
                    observed_values - observed_mean,
                )
                conditional_covariance = covariance[np.ix_(latent_index, latent_index)] - (
                    latent_observed_covariance @ np.linalg.solve(observed_covariance, latent_observed_covariance.T)
                )
                conditional_covariance = (conditional_covariance + conditional_covariance.T) / 2
                conditional_order = latent_order
            elif query_continuous:
                conditional_mean = mean
                conditional_covariance = covariance
                conditional_order = order
            else:
                conditional_mean = None
                conditional_covariance = None
                conditional_order = []

            if np.isfinite(log_weight):
                component_records.append(
                    {
                        "log_weight": float(log_weight),
                        "mean": conditional_mean,
                        "covariance": conditional_covariance,
                        "order": conditional_order,
                        "assignment": assignment,
                    }
                )

        if not component_records:
            raise ValueError("The evidence has zero probability under the model.")
        log_total = logsumexp([record["log_weight"] for record in component_records])

        if not joint:
            return self._marginals(component_records, log_total, variables)

        if query_continuous:
            weights, means, covariances, discrete_states = self._select_continuous(
                component_records, query_continuous, query_discrete
            )
            return GaussianMixture(
                weights=weights,
                means=means,
                covariances=covariances,
                variables=query_continuous,
                discrete_states=discrete_states if query_discrete else None,
            )

        return self._discrete_factor(component_records, log_total, query_discrete)

    def _enumerate_discrete_assignments(self) -> list[dict[Hashable, Hashable]]:
        states = [self.model.states[var] for var in self.model.discrete_variables]
        return [dict(zip(self.model.discrete_variables, values)) for values in product(*states)]

    def _discrete_probability(self, assignment: dict[Hashable, Hashable]) -> float:
        probability = 1.0
        for var in self.model.discrete_variables:
            cpd = self.model.get_cpds(var)
            probability *= float(cpd.get_value(**{parent: assignment[parent] for parent in cpd.variables}))
        return probability

    def _joint_gaussian(self, assignment: dict[Hashable, Hashable]) -> tuple[np.ndarray, np.ndarray, list[Hashable]]:
        continuous_edges = [
            (parent, child)
            for parent, child in self.model.edges()
            if parent not in self.model.discrete_variables and child not in self.model.discrete_variables
        ]
        gaussian_model = LinearGaussianBayesianNetwork(continuous_edges)
        gaussian_model.add_nodes_from(self.model.continuous_variables)

        for var in nx.topological_sort(gaussian_model):
            cpd = self.model.get_cpds(var)
            discrete_state = tuple(assignment[parent] for parent in cpd.discrete_evidence)
            gaussian_model.add_cpds(cpd.to_linear_gaussian_cpd(discrete_state))

        mean, covariance = gaussian_model.to_joint_gaussian()
        order = list(nx.topological_sort(gaussian_model))
        return mean, covariance, order

    def _discrete_factor(self, component_records, log_total, query_discrete) -> DiscreteFactor:
        query_states = [self.model.states[var] for var in query_discrete]
        values = np.zeros([len(states) for states in query_states], dtype=float)

        for record in component_records:
            index = tuple(self.model.states[var].index(record["assignment"][var]) for var in query_discrete)
            values[index] += np.exp(record["log_weight"] - log_total)

        return DiscreteFactor(
            variables=query_discrete,
            cardinality=[len(states) for states in query_states],
            values=values,
            state_names={var: self.model.states[var] for var in query_discrete},
        )

    def _marginals(self, component_records, log_total, variables):
        result = {}
        for variable in variables:
            if variable in self.model.discrete_variables:
                result[variable] = self._discrete_factor(component_records, log_total, [variable])
            else:
                weights, means, covariances, _ = self._select_continuous(component_records, [variable], [], log_total)
                result[variable] = GaussianMixture(
                    weights=weights,
                    means=means,
                    covariances=np.asarray(covariances),
                    variables=[variable],
                )
        return result

    @staticmethod
    def _select_continuous(component_records, query_continuous, query_discrete, log_total=None):
        if log_total is None:
            log_total = logsumexp([record["log_weight"] for record in component_records])
        weights = [np.exp(record["log_weight"] - log_total) for record in component_records]
        means = []
        covariances = []
        discrete_states = []

        for record in component_records:
            index_by_variable = {var: index for index, var in enumerate(record["order"])}
            query_index = [index_by_variable[var] for var in query_continuous]
            means.append(record["mean"][query_index])
            covariances.append(record["covariance"][np.ix_(query_index, query_index)])
            discrete_states.append({var: record["assignment"][var] for var in query_discrete})

        return weights, np.asarray(means), np.asarray(covariances), discrete_states
