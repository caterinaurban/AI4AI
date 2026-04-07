#!/usr/bin/env python3
"""
Demo: bound propagation and local robustness verification.

Supported models:
  toy    -- mpri1.py          (2 inputs,   2 outputs, ReLU, hand-crafted)
  bcw    -- bcw/model.py      (9 inputs,   2 outputs, ReLU, Breast Cancer Wisconsin)
  mnist  -- mnist-net_256x2.onnx (784 inputs, 10 outputs, ReLU, ONNX)

Supported domains:
  interval  -- Box / Interval domain
  symbolic  -- Symbolic (affine) domain
  deeppoly  -- DeepPoly domain            [default]
  product   -- Symbolic x DeepPoly reduced product

Supported tasks:
  propagate -- Run bound propagation and display output bounds
  verify    -- Check local robustness (postcondition = class at center point)
  compare   -- Run propagate with every domain and compare precision
  all       -- Run propagate + verify             [default]

Usage examples:
  python demo.py
  python demo.py --model toy --epsilon 0.2 --domain interval
  python demo.py --model bcw --row 2 --epsilon 0.05 --task verify
  python demo.py --model toy --epsilon 0.15 --task compare
"""

import argparse
import os
import sys

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, '')
sys.path.insert(0, SRC)

from abstract_domains.abstract_domain import AbstractDomain
from abstract_domains.symbolic_domain  import SymbolicDomain
from abstract_domains.deeppoly_domain  import DeepPolyDomain
from abstract_domains.product_domain   import ProductDomain
from engine.verify import bound, verify, Outcome
from frontend.mirror import Activations
from frontend.python2mirror import python2mirror

# ── domain registry ───────────────────────────────────────────────────────────
DOMAINS = {
    'interval': AbstractDomain,
    'symbolic': SymbolicDomain,
    'deeppoly': DeepPolyDomain,
    'product':  None,   # constructed differently (see make_domain)
}

DOMAIN_LABELS = {
    'interval': 'Interval',
    'symbolic': 'Symbolic',
    'deeppoly': 'DeepPoly',
    'product':  'Symbolic×DeepPoly',
}


def make_domain(name: str, ranges: dict):
    if name == 'product':
        return ProductDomain(ranges, domains=[SymbolicDomain, DeepPolyDomain])
    return DOMAINS[name](ranges)


# ── model loaders ─────────────────────────────────────────────────────────────

def load_toy():
    """Toy 2-input network (mpri1.py). Returns (mirror, center_point)."""
    path = os.path.join(SRC, 'models', 'mpri', 'mpri1.py')
    mirror = python2mirror(path)
    center = {'x00': 0.5, 'x01': 0.5}
    return mirror, center


def load_bcw(row: int = 0):
    """BCW network + one row from the test set. Returns (mirror, center_point, label)."""
    import pandas as pd
    path   = os.path.join(SRC, 'models', 'bcw', 'model.py')
    x_path = os.path.join(SRC, 'models', 'bcw', 'bcw_Xtest.csv')
    y_path = os.path.join(SRC, 'models', 'bcw', 'bcw_ytest.csv')

    mirror = python2mirror(path)
    X = pd.read_csv(x_path, header=None)
    y = pd.read_csv(y_path, header=None)

    n_rows = len(X)
    if row >= n_rows:
        raise ValueError(f"Row {row} out of range (BCW test set has {n_rows} rows).")

    center = {f'x0{c}': float(X.iloc[row, c]) for c in range(X.shape[1])}
    label  = int(y.iloc[row, 0])
    return mirror, center, label


def load_mnist(row: int = 0):
    """MNIST ONNX network + one row. Returns (mirror, center_point, label)."""
    import pandas as pd
    from frontend.onnx2mirror import onnx2mirror

    mirror = onnx2mirror(os.path.join(SRC, 'models', 'mnist', 'mnist-net_256x2.onnx'))

    X = pd.read_csv(os.path.join(SRC, 'models', 'mnist', 'mnist_Xtest.csv'), header=None)
    y = pd.read_csv(os.path.join(SRC, 'models', 'mnist', 'mnist_ytest.csv'), header=None)

    n_rows = len(X)
    if row >= n_rows:
        raise ValueError(f"Row {row} out of range (MNIST test set has {n_rows} rows).")

    center = {f'x0{c}': float(X.iloc[row, c]) for c in range(X.shape[1])}
    label  = int(y.iloc[row, 0])
    return mirror, center, label


def load_model(model_name: str, row: int):
    """Load a model and return (mirror, center, label_or_None)."""
    if model_name == 'toy':
        mirror, center = load_toy()
        return mirror, center, None
    elif model_name == 'bcw':
        return load_bcw(row)
    elif model_name == 'mnist':
        return load_mnist(row)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: toy, bcw, adult, mnist.")


# ── input region construction ─────────────────────────────────────────────────

def make_ranges(center: dict, epsilon: float) -> dict:
    """Build an epsilon-ball (interval) around the center point, clipped to [0, 1]."""
    return {
        var: (max(0.0, v - epsilon), min(1.0, v + epsilon))
        for var, v in center.items()
    }


# ── display helpers ───────────────────────────────────────────────────────────

def class_name(output_var: str, mirror, label: int = None) -> str:
    """Translate an output variable name to a human-readable class string."""
    if output_var in ('?', '⊥'):
        return output_var
    idx = mirror.outputs.index(output_var)
    s = f"class {idx}"
    if label is not None and idx == label:
        s += " (ground truth)"
    return s


def print_header(title: str):
    print()
    print('=' * 60)
    print(f"  {title}")
    print('=' * 60)


def print_bounds_summary(mirror, final_state, label=None):
    """Print a table of output bounds and indicate the predicted class."""
    print("\n  Output bounds:")
    for out in mirror.outputs:
        lo, hi = final_state.bounds[out]
        idx = mirror.outputs.index(out)
        print(f"    {out}  (class {idx}):  [{lo:+.4f},  {hi:+.4f}]")

    found = final_state.outcome(mirror.outputs, log=False)
    if found == '⊥':
        print("\n  Result: INFEASIBLE (precondition is empty)")
    elif found == '?':
        print("\n  Result: AMBIGUOUS — output bounds overlap, cannot determine class")
    else:
        print(f"\n  Result: class {mirror.outputs.index(found)} ({found}) is provably the maximum")
        if label is not None:
            match = "CORRECT" if mirror.outputs.index(found) == label else "INCORRECT"
            print(f"  Ground-truth label: {label}  →  {match}")
    return found


# ── tasks ─────────────────────────────────────────────────────────────────────

def run_propagate(mirror, ranges: dict, domain_name: str, label=None):
    print_header(f"Bound propagation  [{DOMAIN_LABELS[domain_name]}]")
    print(f"\n  Input region ({len(ranges)} variables, ε applied):")
    for var, (lo, hi) in list(ranges.items())[:6]:
        print(f"    {var}: [{lo:.4f}, {hi:.4f}]")
    if len(ranges) > 6:
        print(f"    ... ({len(ranges) - 6} more variables)")

    initial = make_domain(domain_name, ranges)
    final, activated, deactivated, found = bound(mirror, initial)

    print(f"\n  ReLU neurons:  {len(mirror.activations)} total  |"
          f"  {len(activated)} active  |  {len(deactivated)} inactive  |"
          f"  {len(mirror.activations) - len(activated) - len(deactivated)} unstable")

    print_bounds_summary(mirror, final, label)
    return found


def run_verify(mirror, center: dict, ranges: dict, domain_name: str, label=None):
    print_header(f"Local robustness verification  [{DOMAIN_LABELS[domain_name]}]")

    # Determine the postcondition: the class predicted at the center point
    point_ranges = {var: (v, v) for var, v in center.items()}
    point_initial = make_domain(domain_name, point_ranges)
    _, _, _, point_pred = bound(mirror, point_initial)

    if point_pred in ('?', '⊥'):
        print(f"\n  Cannot determine prediction at center point ({point_pred}). Aborting.")
        return

    postcondition = mirror.outputs.index(point_pred)
    print(f"\n  Center-point prediction: class {postcondition} ({point_pred})", end="")
    if label is not None:
        match = " (correct)" if postcondition == label else " (WRONG)"
        print(match, end="")
    print()
    print(f"  Postcondition to verify: the network predicts class {postcondition} "
          f"for ALL inputs in the epsilon-ball")

    initial = make_domain(domain_name, ranges)
    result = verify(mirror, initial, postcondition)

    status_str = {
        Outcome.Verified:     "VERIFIED   -- the network is locally robust at this point",
        Outcome.Unknown:      "UNKNOWN    -- the analysis is inconclusive (over-approximation)",
        Outcome.Infeasible:   "INFEASIBLE -- the input region is empty",
        Outcome.Counterexample: "COUNTEREXAMPLE -- a violation was found",
    }
    print(f"\n  Outcome: {status_str[result]}")
    return result


def run_compare(mirror, ranges: dict, label=None):
    print_header("Domain comparison")
    print(f"\n  Input region: {len(ranges)} variables")

    results = {}
    for name in DOMAINS:
        initial = make_domain(name, ranges)
        final, activated, deactivated, found = bound(mirror, initial)

        # Measure output interval widths as a proxy for precision (smaller = more precise)
        widths = [final.bounds[o][1] - final.bounds[o][0] for o in mirror.outputs]
        avg_width = sum(widths) / len(widths)

        results[name] = (found, len(activated), len(deactivated), avg_width)

    # Print comparison table
    total = len(mirror.activations)
    col = 18
    header = f"  {'Domain':<{col}}  {'Prediction':<12}  {'Active':>6}  {'Inactive':>8}  {'Avg output width':>18}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))
    for name, (found, act, deact, width) in results.items():
        pred = f"class {mirror.outputs.index(found)}" if found not in ('?', '⊥') else found
        print(f"  {DOMAIN_LABELS[name]:<{col}}  {pred:<12}  {act:>6}  {deact:>8}  {width:>18.6f}")

    print(f"\n  Total ReLU neurons: {total}")
    print("  (Smaller average output width = more precise analysis)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bound propagation and local robustness verification demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model',   default='toy',
                        choices=['toy', 'bcw', 'adult', 'mnist'],
                        help="Neural network to analyse (default: toy)")
    parser.add_argument('--domain',  default='deeppoly',
                        choices=list(DOMAINS.keys()),
                        help="Abstract domain to use (default: deeppoly)")
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help="Perturbation radius around the center point (default: 0.1)")
    parser.add_argument('--row',     type=int,   default=0,
                        help="Row index in the test CSV (for bcw/adult/mnist, default: 0)")
    parser.add_argument('--task',    default='all',
                        choices=['propagate', 'verify', 'compare', 'all'],
                        help="Task to run (default: all = propagate + verify)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nModel:    {args.model}")
    print(f"Domain:   {DOMAIN_LABELS[args.domain]}")
    print(f"Epsilon:  {args.epsilon}")
    if args.model != 'toy':
        print(f"Row:      {args.row}")
    print(f"Task:     {args.task}")

    # Load model and data
    mirror, center, label = load_model(args.model, args.row)

    print(f"\nNetwork:  {len(mirror.inputs)} inputs, "
          f"{len(mirror.outputs)} outputs, "
          f"{len(mirror.activations)} ReLU neurons, "
          f"{len(mirror.layers)} layers")
    if label is not None:
        print(f"Label:    {label}")

    ranges = make_ranges(center, args.epsilon)

    # Dispatch tasks
    if args.task in ('propagate', 'all'):
        run_propagate(mirror, ranges, args.domain, label)

    if args.task in ('verify', 'all'):
        run_verify(mirror, center, ranges, args.domain, label)

    if args.task == 'compare':
        run_compare(mirror, ranges, label)

    print()


if __name__ == '__main__':
    main()
