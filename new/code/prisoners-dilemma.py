import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    import matplotlib.style
    import marimo as mo
    from typing import List, Tuple, Callable
    from itertools import product
    from quantum_game import QuantumGame, QuantumGameLearner, Adam, Payoffs, Gen

    return Gen, Payoffs, QuantumGame, QuantumGameLearner, mo, np, plt, sp


@app.cell
def _(Gen, Payoffs, QuantumGame, QuantumGameLearner, np):
    N_val = 3
    gamma = np.pi / 2
    restricted = True
    iteration = 5000
    print(f"Restricted: {restricted}")
    qg = QuantumGame(N_players=N_val, gamma=gamma, restricted=restricted)
    qg.entangler()

    learner = QuantumGameLearner(N_players=N_val, gamma=gamma, restricted=restricted)
    init_strategies = np.asarray([ np.asarray( Gen.gen_rand_init(restricted=restricted) , dtype=float, ) for i in range(N_val) ])
    print(init_strategies)

    PC = Payoffs()

    outcomes, payoff_matrix = PC.generate_diner_payoffs(N=N_val, cheap_cost=2, expensive_cost=4, cheap_value=5, expensive_value=7)
    return (
        N_val,
        init_strategies,
        iteration,
        learner,
        payoff_matrix,
        qg,
        restricted,
    )


@app.cell
def _(init_strategies, iteration, learner, payoff_matrix):
    result = learner.train(
        init_strategies, payoff_matrix, iterations=iteration, verbose=1000
    )
    return (result,)


@app.cell
def _(result):
    final_strategies = result["strategies_history"][-1]
    strategies_history = result["strategies_history"].transpose(1, 2, 0)
    payoff_history = result["payoff_history"].T
    return final_strategies, payoff_history, strategies_history


@app.cell
def _(final_strategies, init_strategies, iteration, restricted):
    print("Restricted:",restricted)
    print("Intial Strategy")
    for m,val in enumerate(init_strategies):
        theta, alpha, beta = val
        print(f"{m + 1}: Theta: {theta} | Alpha: {alpha} | Beta: {beta}  ")

    print(f"Final Strategy after Iterations: {iteration}")
    for m,val in enumerate(final_strategies):
        theta, alpha, beta = val
        print(f"{m + 1}: Theta: {theta} | Alpha: {alpha} | Beta: {beta}  ")
    return


@app.cell
def _(Gen, np, restricted):

    def best_response(
        strategies,
        qg,
        payoffs,
        eps=1e-6,
        max_iter=100_000,
        seed=None,
    ):
        """
        Monte Carlo epsilon-Nash equilibrium check.

        strategies: (N_players, 3) array of (theta, alpha, beta)
        qg: initialized QuantumGame
        payoffs: payoff matrix accepted by qg.compute_payoff()
        """

        rng = np.random.default_rng(seed)
        strategies = np.asarray(strategies, dtype=float)

        N = qg.N_players
        best_strategies = strategies.copy()

        # Operators and current payoffs
        operators = np.array([
            qg.strategy_operator(s) for s in strategies
        ])
        current_payoffs = qg.compute_payoff(operators, payoffs)[0]

        best_payoffs = current_payoffs.copy()
        regrets = np.zeros(N)

        for player in range(N):
            best_value = current_payoffs[player]
            best_strategy = strategies[player].copy()

            for _ in range(max_iter):
                candidate = np.asarray(
                    Gen.gen_rand_init(restricted),
                dtype=float,
                )

                test_operators = operators.copy()
                test_operators[player] = qg.strategy_operator(candidate)

                test_payoffs = qg.compute_payoff(
                    test_operators, payoffs
                )[0]

                if test_payoffs[player] > best_value:
                    best_value = test_payoffs[player]
                    best_strategy = candidate.copy()

            best_payoffs[player] = best_value
            best_strategies[player] = best_strategy
            regrets[player] = max(
                0.0, best_value - current_payoffs[player]
            )

        return (
            np.max(regrets) <= eps,
            regrets,
            best_strategies,
            current_payoffs,
            best_payoffs,
        )

    return (best_response,)


@app.cell
def _(best_response, final_strategies, np, payoff_matrix, qg):
    is_ne, regrets, best_strategies, current_payoffs, best_payoffs = (
        best_response(
            final_strategies,
            qg,
            payoff_matrix,
            eps=1e-8,
           max_iter=20_000,
        )
    )

    print("Current payoffs:", current_payoffs)
    print("Best-response payoffs:", best_payoffs)
    print("Player regrets:", regrets)
    print("Maximum regret:", np.max(regrets))
    print("Epsilon-Nash equilibrium:", is_ne)
    print("Best-response strategies:\n", best_strategies)
    return


@app.cell(hide_code=True)
def _(final_strategies, init_strategies, iteration, mo, restricted):
    initial_table = "\n".join(
        f"| {m + 1} | {theta:.6f} | {alpha:.6f} | {beta:.6f} |"
        for m, (theta, alpha, beta) in enumerate(init_strategies)
    )

    final_table = "\n".join(
        f"| {m + 1} | {theta:.6f} | {alpha:.6f} | {beta:.6f} |"
        for m, (theta, alpha, beta) in enumerate(final_strategies)
    )

    mo.md(f"""
    ## Game Configuration

    - **Restricted strategy space:** `{restricted}`
    - **Iterations:** `{iteration}`

    ### Restricted Strategy Space

    \\[
    \\Omega_{{\\mathrm{{restricted}}}} =
    \\left\\{{(\\theta,\\alpha,\\beta)\\;\\middle|\\;
    \\theta\\in[0,\\pi],\\;
    \\alpha\\in\\left[0,\\frac{{\\pi}}{{2}}\\right],\\;
    \\beta=0
    \\right\\}}
    \\]

    ### Initial Strategies

    | Player | $\\theta$ | $\\alpha$ | $\\beta$ |
    |:---:|---:|---:|---:|
    {initial_table}

    ### Final Strategies after Optimization

    | Player | $\\theta$ | $\\alpha$ | $\\beta$ |
    |:---:|---:|---:|---:|
    {final_table}
    """)
    return


@app.cell
def _(N_val, final_strategies, mo, np, qg, sp):
    outputs = []
    for k in range(N_val):
        M = np.round(qg.strategy_operator(final_strategies[k]), 10) + 0
        outputs.append(
            mo.md(rf"$$ \text{{Player {k+1}}} = {sp.latex(sp.nsimplify(sp.Matrix(M)))} $$")
        )

    mo.vstack(outputs)   # must be the last expression in the cell 
    return


@app.cell
def _(N_val, payoff_history, plt):
    N_plot = 9999
    fig0 = plt.figure(figsize = (6,4))
    for i in range(N_val):
        plt.plot(payoff_history[i][:N_plot],label=f"Player {i+1}")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Payoff")
    plt.title("Payoff Evolution")
    plt.show()
    fig0.savefig("Payoff Evolution.png", dpi=300, bbox_inches="tight")
    return


@app.cell
def _(np, plt, strategies_history):
    # Shape: (player, values, iteration)
    iterations = np.arange(strategies_history.shape[2])
    variable_names = [r"$\theta$", r"$\alpha$", r"$\beta$"]

    fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=True)

    for j, ax in enumerate(axes):
        for player in range(strategies_history.shape[0]):
            ax.plot(
                iterations,
                strategies_history[player, j, :],
                label=f"Player {player + 1}"
            )

        ax.set_ylabel(variable_names[j] + " (rad)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration")

    fig.suptitle("Strategy Evolution")
    fig.tight_layout()
    fig.savefig("Strategy Evolution.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
