"""Smoke test: 2-player game, find NE via BR dynamics."""
import time
from game.game import make_n_player_game
from game.utility import aware_player


def main():
    players = [
        aware_player("P1", B=1.0, C=1.0, eps_max=32.0),
        aware_player("P2", B=1.0, C=1.0, eps_max=32.0),
    ]
    game = make_n_player_game(
        players=players,
        group_sizes=[5_000, 5_000],
        expected_batch_size=128,
        steps=int(5 * 10_000 / 128),  # ~390 steps
        actions=(2.0, 4.0, 8.0, 16.0, 32.0),
        delta=1e-12,
    )
    t0 = time.time()
    history = game.best_response_dynamics(start=(8.0, 8.0), max_iter=10)
    print(f"BR dynamics: {history}  ({time.time()-t0:.1f}s)")
    print()
    t0 = time.time()
    ne = game.find_pure_nash_equilibria()
    print(f"Pure NE: {ne}  ({time.time()-t0:.1f}s, |cache|={len(game._eval_cache)})")
    for p in ne[:3]:
        ev = game.evaluate(p)
        print(f"  eps={p} sigma={ev.sigma:.3f} A={ev.advantages.tolist()} u={ev.utilities.tolist()}")


if __name__ == "__main__":
    main()
