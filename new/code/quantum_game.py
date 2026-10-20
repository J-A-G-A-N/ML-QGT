import numpy as np
import random
from itertools import product


class QuantumGame:

    def __init__(self, N_players: int, gamma: float, restricted: bool):
        """
        Args:
            N_players : Number of players in game
            gamma: engtangle variable
            restircted: decidas the domain of the strategy space
        """
        self.N_players = N_players
        self.gamma = gamma
        self.restricted = restricted
        self.initial_state = np.asarray(
            [1.0] + [0.0] * (2**self.N_players - 1), dtype=float
        )

    def entangler(self):
        """
        Creates the entangle operator
        """
        identity = np.eye(2**self.N_players)
        pauli_x = np.array(np.flip(identity, 0))
        self.J = np.matrix(
            np.cos(self.gamma / 2) * identity + 1j * np.sin(self.gamma / 2) * pauli_x
        )
        self.J_dagger = self.J.H

    def strategy_operator(self, param: np.ndarray):
        """Build and store the 2×2 strategy operator from three parameters.
        Args:
            param: Array-like NumPy array with shape (3,), containing
                (theta, alpha, beta) in radians.

        Returns:
            The resulting 2×2 complex-valued strategy operator, also stored
            as self.U
        """

        if param.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {param.shape}")

        theta, alpha, beta = param

        if not (0 <= theta <= np.pi):
            raise ValueError("Require 0 <= theta <= pi")

        if self.restricted:
            if not (0 <= alpha <= np.pi / 2 and beta == 0.0):
                raise ValueError("Require 0 <= alpha <= np.pi / 2, beta == 0")
        else:
            if not (0 <= alpha < 2 * np.pi and 0 <= beta < 2 * np.pi):
                raise ValueError("Require 0 <= alpha, beta < 2*pi")

        A = np.exp(1j * alpha) * np.cos(theta / 2)
        B = 1j * np.exp(1j * beta) * np.sin(theta / 2)
        C = 1j * np.exp(-1j * beta) * np.sin(theta / 2)
        D = np.exp(-1j * alpha) * np.cos(theta / 2)

        self.U = np.array(
            [[A, B], [C, D]],
            dtype=np.complex128,
        )

        return self.U

    def compute_payoff(self, Us: np.ndarray, payoffs: np.ndarray) -> np.ndarray:

        d = 2**self.N_players
        entangled_state = self.initial_state @ self.J

        Us = np.asarray(Us, dtype=np.complex128)
        if Us.ndim == 3:
            Us = Us[None, ...]
        results = np.empty((len(Us), self.N_players), dtype=float)

        for k, profile in enumerate(Us):
            state = np.asarray(entangled_state).reshape(-1).copy()

            for player, U in enumerate(profile):
                tensor = state.reshape((2,) * self.N_players)
                tensor = np.moveaxis(tensor, player, 0)
                tensor = np.tensordot(U, tensor, axes=(1, 0))
                tensor = np.moveaxis(tensor, 0, player)
                state = tensor.reshape(d)

            de_entangled_state = state @ self.J_dagger
            de_entangled_state = np.asarray(de_entangled_state).reshape(-1)

            probabilities = np.abs(de_entangled_state) ** 2
            results[k] = payoffs @ probabilities
        return results


class Gen:
    @staticmethod
    def gen_rand_init(restricted: bool) -> Tuple[float, float, float]:
        if restricted:
            theta = np.pi * random.random()
            alpha = random.uniform(0, np.pi / 2)
            beta = 0.0
        else:
            theta = np.pi * random.random()
            alpha = 2.0 * np.pi * random.random()
            beta = 2.0 * np.pi * random.random()
        return theta, alpha, beta


class Payoffs:
    def __init__(self):
        pass

    def generate_diner_payoffs(
        self, N, cheap_cost, expensive_cost, cheap_value, expensive_value
    ) -> np.ndarray:
        """
        Actions: 0 = cheap meal, 1 = expensive meal.

        Returns
        -------
        outcomes : (2**N, N)
            Every possible action profile.
        A : (N, 2**N)
            Payoff matrix, one row per player.
        """
        outcomes = np.array(list(product([0, 1], repeat=N)), dtype=int)

        # Cost incurred by each diner
        costs = np.where(outcomes == 0, cheap_cost, expensive_cost)

        # Total bill for each outcome
        total_bill = costs.sum(axis=1)

        # Private value of each diner's meal
        values = np.where(outcomes == 0, cheap_value, expensive_value)

        # Each diner pays an equal share of the total bill
        payoff_per_outcome = values - total_bill[:, None] / N

        # Convert from (2**N, N) to (N, 2**N)
        A = payoff_per_outcome.T

        return outcomes, A

    def prisoner_dilemma_payoff(self) -> np.ndarray:
        R = 3.0
        T = 5.0
        S = 0.0
        P = 1.0
        alice_payoff = [R, S, T, P]
        bob_payoff = [R, T, S, P]
        return np.asarray([alice_payoff, bob_payoff])


class Adam:
    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2=0.999,
        eps: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = 0.0
        self.v = 0.0
        self.t = 0.0

    def step(self, strategy_params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if (np.logical_or(self.m, self.v)).all() is None:
            self.m = np.zeros_like(strategy_params, dtype=float)
            self.v = np.zeros_like(strategy_params, dtype=float)
        self.t += 1

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * gradient**2

        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)

        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return strategy_params + update


class QuantumGameLearner:
    def __init__(
        self,
        N_players: int,
        gamma: float,
        restricted: bool,
        learning_rate: float = 0.01,
        finite_diff_step: float = 0.01,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.restricted = restricted
        self.finite_diff_step = finite_diff_step
        self.N_players = N_players

        self.optimizers = [Adam(learning_rate=learning_rate) for _ in range(N_players)]

        self.qg = QuantumGame(
            self.N_players, gamma=self.gamma, restricted=self.restricted
        )
        self.qg.entangler()

    def _payoff_from_params(
        self, params: np.ndarray, payoff_matrix: np.ndarray
    ) -> np.ndarray:

        operator = np.stack([self.qg.strategy_operator(row) for row in params])
        return self.qg.compute_payoff(operator, payoff_matrix)[0]

    def grad(
        self, strategies_params: np.ndarray, payoff_matrix: np.ndarray
    ) -> np.ndarray:
        params = np.asarray(strategies_params, dtype=float)

        if params.shape != (self.N_players, 3):
            raise ValueError(f"strategies_params must have shape{(self.N_player,3)}")

        gradients = np.zeros_like(params)
        h = self.finite_diff_step

        for player in range(self.N_players):
            dimensions = (0, 1) if self.restricted else (0, 1, 2)

            for dim in dimensions:

                plus, minus = params.copy(), params.copy()
                plus[player][dim] += h
                minus[player][dim] -= h

                if self.restricted and dim == 1:
                    plus[player, 2] += h
                    minus[player, 2] -= h

                plus = self._project(plus)
                minus = self._project(minus)

                payoff_plus = self._payoff_from_params(plus, payoff_matrix)[player]
                payoff_minus = self._payoff_from_params(minus, payoff_matrix)[player]
                gradients[player][dim] = (payoff_plus - payoff_minus) / (2.0 * h)

                if self.restricted and dim == 1:
                    gradients[player, 2] = gradients[player, 1]

        return gradients

    def _project(self, params: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=float).copy()

        p[:, 0] = np.clip(p[:, 0], 0.0, np.pi)

        if self.restricted:
            upper = np.nextafter(np.pi / 2, 0.0)
            p[:, 1] = np.clip(p[:, 1], 0.0, upper)
            p[:, 2] = 0.0
        else:
            p[:, 1] %= 2.0 * np.pi
            p[:, 2] %= 2.0 * np.pi

        return p

    def train(
        self,
        initial_params: np.ndarray,
        payoff_matrix: np.ndarray,
        iterations: int = 10_000,
        verbose: int = 100,
    ):
        if iterations < 1:
            raise ValueError("Iterations must be at least 1")

        params = np.asarray(initial_params, dtype=float)

        if params.shape != (self.N_players, 3):
            raise ValueError(f"initial_params must have shape {(self.N_players, 3)}")

        payoff_matrix = np.asarray(payoff_matrix, dtype=float)

        if payoff_matrix.shape != (self.N_players, 2**self.N_players):
            raise ValueError(
                f"payoff_matrix must have shape "
                f"{(self.N_players, 2**self.N_players)}"
            )

        params = self._project(params)

        strategy_history = np.empty((iterations + 1, self.N_players, 3), dtype=float)

        payoff_history = np.empty((iterations + 1, self.N_players), dtype=float)

        gradient_history = np.empty((iterations, self.N_players, 3), dtype=float)

        strategy_history[0] = params

        payoff_history[0] = self._payoff_from_params(params, payoff_matrix)

        for iteration in range(iterations):
            gradients = self.grad(params, payoff_matrix)
            gradient_history[iteration] = gradients

            updated = params.copy()

            # Update each player's strategy using their own optimizer.
            for player in range(self.N_players):
                updated[player] = self.optimizers[player].step(
                    params[player], gradients[player]
                )

            # Project and record the complete strategy profile once.
            params = self._project(updated)

            strategy_history[iteration + 1] = params
            payoff_history[iteration + 1] = self._payoff_from_params(
                params, payoff_matrix
            )

            if verbose and (iteration + 1) % verbose == 0:
                print(
                    f"Iteration: {iteration + 1}: "
                    f"payoffs: {payoff_history[iteration + 1]}"
                )

        return {
            "strategies_history": strategy_history,
            "payoff_history": payoff_history,
            "gradient_history": gradient_history,
        }
