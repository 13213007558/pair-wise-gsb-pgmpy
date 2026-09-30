from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from numbers import Real

import numpy as np

from pgmpy.factors.base import BaseFactor
from pgmpy.factors.continuous.LinearGaussianCPD import LinearGaussianCPD


class ConditionalLinearGaussianCPD(BaseFactor):
    r"""
    Conditional linear Gaussian distribution.

    For continuous variable :math:`Y`, discrete parents :math:`D`, and continuous
    parents :math:`X`, each discrete configuration defines:

    .. math::

        p(Y \mid X=x, D=d) = \mathcal{N}(\beta_{0,d} + \beta_{1,d} x_1 +
        \cdots; \sigma_d).

    Parameters
    ----------
    variable : Hashable
        Continuous variable defined by this CPD.
    discrete_evidence : list[Hashable]
        Discrete parent variables.
    continuous_evidence : list[Hashable]
        Continuous parent variables.
    parameters : dict[tuple[Hashable, ...], dict]
        Mapping from a tuple of discrete parent states to ``{"beta": ...,
        "std": ...}``. A scalar key is accepted when there is one discrete
        parent. Beta is ``[intercept, coefficient...]`` in continuous-parent
        order.
    state_names : dict[Hashable, list]
        State names for each discrete parent in the same order used by
        ``discrete_evidence`` and parameter keys.
    """

    def __init__(
        self,
        variable: Hashable,
        discrete_evidence: list[Hashable] | None,
        continuous_evidence: list[Hashable] | None,
        parameters: dict[tuple[Hashable, ...] | Hashable, dict],
        state_names: dict[Hashable, list] | None = None,
    ) -> None:
        try:
            hash(variable)
        except TypeError as e:
            raise ValueError(f"`variable` must be hashable. Got {type(variable).__name__}") from e

        self.variable = variable
        self.discrete_evidence = list(discrete_evidence or [])
        self.continuous_evidence = list(continuous_evidence or [])
        self.evidence = self.discrete_evidence + self.continuous_evidence
        self.variables = [variable] + self.evidence
        self.state_names = dict(state_names or {})

        if len(set(self.variables)) != len(self.variables):
            raise ValueError("Variable and evidence names must be unique.")

        missing_states = set(self.discrete_evidence) - set(self.state_names)
        if missing_states:
            raise ValueError(f"Missing state names for discrete evidence: {sorted(missing_states, key=str)}")

        for evidence_var in self.discrete_evidence:
            states = self.state_names[evidence_var]
            if len(states) == 0 or len(states) != len(set(states)):
                raise ValueError(f"States for {evidence_var} must be a non-empty list of unique values.")

        self.parameters = {}
        for key, parameter in parameters.items():
            normalized_key = key if isinstance(key, tuple) else (key,)
            if len(normalized_key) != len(self.discrete_evidence):
                raise ValueError(
                    f"Expected parameter key of length {len(self.discrete_evidence)}. Got {normalized_key}."
                )

            for evidence_var, state in zip(self.discrete_evidence, normalized_key):
                if state not in self.state_names[evidence_var]:
                    raise ValueError(f"Unknown state {state!r} for discrete evidence {evidence_var!r}.")

            beta = np.asarray(parameter.get("beta"), dtype=float)
            std = parameter.get("std")
            expected_beta_length = len(self.continuous_evidence) + 1
            if beta.shape != (expected_beta_length,):
                raise ValueError(
                    f"Beta for {normalized_key} must have shape ({expected_beta_length},). Got {beta.shape}."
                )
            if not np.all(np.isfinite(beta)):
                raise ValueError(f"Beta for {normalized_key} must contain finite values.")
            if isinstance(std, bool) or not isinstance(std, Real) or not np.isfinite(std) or std <= 0:
                raise ValueError(f"std for {normalized_key} must be a positive number. Got {std!r}.")

            self.parameters[normalized_key] = {"beta": beta, "std": float(std)}

        expected_keys = set(product(*(self.state_names[var] for var in self.discrete_evidence)))
        missing_parameters = expected_keys - set(self.parameters)
        if missing_parameters:
            raise ValueError(f"Missing parameters for discrete configurations: {sorted(missing_parameters, key=str)}")

        extra_parameters = set(self.parameters) - expected_keys
        if extra_parameters:
            raise ValueError(f"Unexpected discrete configurations: {sorted(extra_parameters, key=str)}")

    def get_parameters(self, discrete_state: tuple[Hashable, ...] | Hashable) -> tuple[np.ndarray, float]:
        """Return beta and std for one discrete-parent configuration."""
        key = discrete_state if isinstance(discrete_state, tuple) else (discrete_state,)
        parameter = self.parameters[key]
        return parameter["beta"], parameter["std"]

    def to_linear_gaussian_cpd(self, discrete_state: tuple[Hashable, ...] | Hashable) -> LinearGaussianCPD:
        """Return the pure linear Gaussian CPD selected by a discrete configuration."""
        beta, std = self.get_parameters(discrete_state)
        return LinearGaussianCPD(
            variable=self.variable,
            beta=beta,
            std=std,
            evidence=list(self.continuous_evidence),
        )

    def is_valid_cpd(self) -> bool:
        return len(self.parameters) == np.prod(
            [len(self.state_names[var]) for var in self.discrete_evidence],
            dtype=int,
        )

    def copy(self) -> ConditionalLinearGaussianCPD:
        return ConditionalLinearGaussianCPD(
            variable=self.variable,
            discrete_evidence=list(self.discrete_evidence),
            continuous_evidence=list(self.continuous_evidence),
            parameters={
                key: {"beta": parameter["beta"].copy(), "std": parameter["std"]}
                for key, parameter in self.parameters.items()
            },
            state_names={var: list(states) for var, states in self.state_names.items()},
        )

    def __str__(self) -> str:
        return (
            f"P({self.variable} | {', '.join(map(str, self.evidence))}) = "
            f"ConditionalLinearGaussian({len(self.parameters)} configurations)"
        )

    def __repr__(self) -> str:
        return f"<ConditionalLinearGaussianCPD: {self} at {hex(id(self))}>"
