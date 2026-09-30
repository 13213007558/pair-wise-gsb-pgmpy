from __future__ import annotations

from collections.abc import Hashable

import numpy as np

from pgmpy.factors.base import BaseFactor


class GaussianMixture(BaseFactor):
    """
    A finite mixture of multivariate Gaussian distributions.

    Parameters
    ----------
    weights : array-like
        Normalized component weights.
    means : np.ndarray
        Component means with shape ``(n_components, n_variables)``.
    covariances : np.ndarray
        Component covariance matrices with shape
        ``(n_components, n_variables, n_variables)``.
    variables : list[Hashable]
        Continuous variables in the column/row order used by means and
        covariances.
    discrete_states : list[dict]
        Discrete assignment labeling each component.
    """

    def __init__(
        self,
        weights,
        means,
        covariances,
        variables: list[Hashable],
        discrete_states: list[dict[Hashable, Hashable]] | None = None,
    ) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.means = np.asarray(means, dtype=float)
        self.covariances = np.asarray(covariances, dtype=float)
        self.variables = list(variables)
        self.discrete_states = list(discrete_states or [])
        self.discrete_variables = list(self.discrete_states[0].keys()) if self.discrete_states else []

        n_components = len(self.weights)
        if self.means.shape != (n_components, len(self.variables)):
            raise ValueError("means shape does not match weights and variables.")
        if self.covariances.shape != (n_components, len(self.variables), len(self.variables)):
            raise ValueError("covariances shape does not match weights and variables.")
        if len(self.discrete_states) not in (0, n_components):
            raise ValueError("discrete_states must contain one assignment for every mixture component.")
        if np.any(self.weights < 0):
            raise ValueError("weights must be non-negative.")

        self.normalize()

    def normalize(self) -> GaussianMixture:
        total = self.weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Weights must have a positive finite total.")
        self.weights = self.weights / total
        return self

    def marginalize(self, variables: list[Hashable], inplace: bool = True) -> GaussianMixture:
        """Marginalize selected continuous variables while retaining all components."""
        indices = [self.variables.index(var) for var in variables]
        keep = [index for index in range(len(self.variables)) if index not in indices]
        means = self.means[:, keep]
        covariances = self.covariances[np.ix_(np.arange(len(self.weights)), keep, keep)]

        if inplace:
            self.means = means
            self.covariances = covariances
            self.variables = [self.variables[index] for index in keep]
            return self

        return GaussianMixture(
            weights=self.weights.copy(),
            means=means,
            covariances=covariances,
            variables=[self.variables[index] for index in keep],
            discrete_states=[state.copy() for state in self.discrete_states],
        )

    def discrete_marginal(self, variable: Hashable) -> dict[Hashable, float]:
        """Aggregate component weights for one discrete variable."""
        marginal = {}
        for weight, state in zip(self.weights, self.discrete_states):
            state_value = state[variable]
            marginal[state_value] = marginal.get(state_value, 0.0) + float(weight)
        return marginal

    def variances(self) -> np.ndarray:
        """Return the marginal variance of a one-dimensional mixture component."""
        if len(self.variables) != 1:
            raise ValueError("variances is only defined for a one-dimensional Gaussian mixture.")
        return self.covariances[:, 0, 0]

    def copy(self) -> GaussianMixture:
        return GaussianMixture(
            weights=self.weights.copy(),
            means=self.means.copy(),
            covariances=self.covariances.copy(),
            variables=list(self.variables),
            discrete_states=[state.copy() for state in self.discrete_states],
        )

    def __str__(self) -> str:
        return (
            f"GaussianMixture over {self.variables} with {len(self.weights)} components; "
            f"weights={np.round(self.weights, 4).tolist()}"
        )

    def __repr__(self) -> str:
        return f"<{self} at {hex(id(self))}>"
