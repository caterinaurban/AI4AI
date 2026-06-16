#!/usr/bin/env python3
"""
Demo: bound propagation and local robustness verification.

Supported models:
  handcrafted -- models/handcrafted/*.py  (hand-crafted nets; pick one with --net, default mpri1)
  bcw         -- bcw/model.py      (9 inputs,   2 outputs, ReLU, Breast Cancer Wisconsin)
  mnist       -- mnist-net_256x2.onnx (784 inputs, 10 outputs, ReLU, ONNX)

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
  python demo.py --model handcrafted --net mpri3 --epsilon 0.2 --domain interval
  python demo.py --model bcw --row 2 --epsilon 0.05 --task verify
  python demo.py --model handcrafted --epsilon 0.15 --task compare
  python demo.py --model handcrafted --net mpri4 --input-min -1 --input-max 1
  python demo.py --model handcrafted --range 0.2 0.8 --task propagate
  python demo.py --model handcrafted --var-range x00 0.1 0.3 --var-range x01 0.4 0.9 --task propagate
  python demo.py --model handcrafted --range 0.2 0.8 --point x00 0.3 --task all
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

def list_handcrafted_nets() -> list:
    """Base names of every hand-crafted network available in models/handcrafted/."""
    handcrafted_dir = os.path.join(SRC, 'models', 'handcrafted')
    return sorted(f[:-3] for f in os.listdir(handcrafted_dir) if f.endswith('.py'))


HANDCRAFTED_NETS = list_handcrafted_nets()


def load_handcrafted(net: str = 'mpri1'):
    """Hand-crafted network from models/handcrafted/<net>.py. Returns mirror."""
    if net not in HANDCRAFTED_NETS:
        raise ValueError(f"Unknown handcrafted net '{net}'. Choose from: {', '.join(HANDCRAFTED_NETS)}.")
    path = os.path.join(SRC, 'models', 'handcrafted', f'{net}.py')
    return python2mirror(path)


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


def load_model(model_name: str, row: int, net: str = 'mpri1'):
    """Load a model and return (mirror, default_point, label_or_None).

    default_point is None for handcrafted nets (no associated data point; the
    midpoint of --input-min/--input-max is used as the default in main()), or
    the loaded data row for bcw/mnist.
    """
    if model_name == 'handcrafted':
        return load_handcrafted(net), None, None
    elif model_name == 'bcw':
        return load_bcw(row)
    elif model_name == 'mnist':
        return load_mnist(row)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: handcrafted, bcw, adult, mnist.")


# ── input region construction ─────────────────────────────────────────────────

def make_ranges(center: dict, epsilon: float, input_range: tuple = (0.0, 1.0)) -> dict:
    """Build an epsilon-ball (interval) around the center point, clipped to input_range."""
    lo, hi = input_range
    return {
        var: (max(lo, v - epsilon), min(hi, v + epsilon))
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
    print(f"\n  Input region ({len(ranges)} variables):")
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


def run_verify(mirror, point: dict, ranges: dict, domain_name: str, label=None):
    print_header(f"Local robustness verification  [{DOMAIN_LABELS[domain_name]}]")

    # Determine the postcondition: the class predicted at the reference point
    point_ranges = {var: (v, v) for var, v in point.items()}
    point_initial = make_domain(domain_name, point_ranges)
    _, _, _, point_pred = bound(mirror, point_initial)

    if point_pred in ('?', '⊥'):
        print(f"\n  Cannot determine prediction at the reference point ({point_pred}). Aborting.")
        return

    postcondition = mirror.outputs.index(point_pred)
    print(f"\n  Reference-point prediction: class {postcondition} ({point_pred})", end="")
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
    parser.add_argument('--model',   default='handcrafted',
                        choices=['handcrafted', 'bcw', 'adult', 'mnist'],
                        help="Neural network to analyse (default: handcrafted)")
    parser.add_argument('--net',     default='mpri1',
                        choices=HANDCRAFTED_NETS,
                        help=f"Hand-crafted network from models/handcrafted/ to use when "
                             f"--model handcrafted (default: mpri1). "
                             f"Available: {', '.join(HANDCRAFTED_NETS)}")
    parser.add_argument('--input-min', type=float, default=0.0,
                        help="Lower bound of the default input range for handcrafted models "
                             "(default: 0.0). Used for --epsilon/verify, and as the propagate/"
                             "compare range unless --range/--var-range is given.")
    parser.add_argument('--input-max', type=float, default=1.0,
                        help="Upper bound of the default input range for handcrafted models "
                             "(default: 1.0). Used for --epsilon/verify, and as the propagate/"
                             "compare range unless --range/--var-range is given.")
    parser.add_argument('--range', nargs=2, type=float, metavar=('LO', 'HI'), default=None,
                        help="Directly set the same [LO, HI] input range for every variable, "
                        "for propagate/compare (handcrafted models only). Bypasses "
                        "--input-min/--input-max for those tasks.")
    parser.add_argument('--var-range', nargs=3, metavar=('NAME', 'LO', 'HI'), action='append',
                        dest='var_ranges', default=[],
                        help="Override the propagate/compare input range for a single named "
                        "variable (repeatable, handcrafted models only), "
                        "e.g. --var-range x00 0.1 0.3.")
    parser.add_argument('--point', nargs=2, metavar=('NAME', 'VALUE'), action='append',
                        dest='points', default=[],
                        help="Override the value of a named input variable at the reference "
                        "point used for --epsilon / verify (repeatable), e.g. --point x00 0.3. "
                        "Defaults to the center of the input range (or the loaded data row "
                        "for bcw/mnist).")
    parser.add_argument('--domain',  default='deeppoly',
                        choices=list(DOMAINS.keys()),
                        help="Abstract domain to use (default: deeppoly)")
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help="Perturbation radius around the reference point (default: 0.1)")
    parser.add_argument('--row',     type=int,   default=0,
                        help="Row index in the test CSV (for bcw/adult/mnist, default: 0)")
    parser.add_argument('--task',    default='all',
                        choices=['propagate', 'verify', 'compare', 'all'],
                        help="Task to run (default: all = propagate + verify)")
    args = parser.parse_args()
    if args.input_min >= args.input_max:
        parser.error("--input-min must be smaller than --input-max")
    if args.range is not None and args.range[0] >= args.range[1]:
        parser.error("--range LO must be smaller than HI")
    var_ranges = []
    for name, lo_s, hi_s in args.var_ranges:
        lo, hi = float(lo_s), float(hi_s)
        if lo >= hi:
            parser.error(f"--var-range {name}: LO must be smaller than HI")
        var_ranges.append((name, lo, hi))
    args.var_ranges = var_ranges
    args.points = [(name, float(value)) for name, value in args.points]
    return args


def main():
    args = parse_args()
    explicit_ranges = args.model == 'handcrafted' and (args.range is not None or args.var_ranges)

    print(f"\nModel:    {args.model}")
    if args.model == 'handcrafted':
        print(f"Net:      {args.net}")
        print(f"Input range: [{args.input_min}, {args.input_max}]")
    print(f"Domain:   {DOMAIN_LABELS[args.domain]}")
    print(f"Epsilon:  {args.epsilon}")
    if args.model != 'handcrafted':
        print(f"Row:      {args.row}")
    print(f"Task:     {args.task}")

    # Load model and data
    mirror, default_point, label = load_model(args.model, args.row, args.net)

    print(f"\nNetwork:  {len(mirror.inputs)} inputs, "
          f"{len(mirror.outputs)} outputs, "
          f"{len(mirror.activations)} ReLU neurons, "
          f"{len(mirror.layers)} layers")
    if label is not None:
        print(f"Label:    {label}")

    bounds = (args.input_min, args.input_max) if args.model == 'handcrafted' else (0.0, 1.0)
    if default_point is None:
        lo, hi = bounds
        default_point = {var: (lo + hi) / 2.0 for var in mirror.inputs}

    # The reference point for --epsilon / verify: --point overrides, else the default center
    point = dict(default_point)
    for name, value in args.points:
        if name not in mirror.inputs:
            raise ValueError(f"Unknown input variable '{name}' for --point. "
                              f"Network inputs: {', '.join(mirror.inputs)}.")
        point[name] = value
    verify_ranges = make_ranges(point, args.epsilon, bounds)

    # Ranges for propagate/compare. For handcrafted models, this is the full [input-min,
    # input-max] box by default, or --range/--var-range if given. bcw/mnist have no such
    # "full box" notion, so they keep using the epsilon-ball around the loaded data point.
    if args.model == 'handcrafted':
        if explicit_ranges:
            base = tuple(args.range) if args.range is not None else bounds
            ranges = {var: base for var in mirror.inputs}
            for name, lo, hi in args.var_ranges:
                if name not in mirror.inputs:
                    raise ValueError(f"Unknown input variable '{name}' for --var-range. "
                                      f"Network inputs: {', '.join(mirror.inputs)}.")
                ranges[name] = (lo, hi)
        else:
            ranges = {var: bounds for var in mirror.inputs}
    else:
        ranges = verify_ranges

    # Dispatch tasks
    if args.task in ('propagate', 'all'):
        run_propagate(mirror, ranges, args.domain, label)

    if args.task in ('verify', 'all'):
        run_verify(mirror, point, verify_ranges, args.domain, label)

    if args.task == 'compare':
        run_compare(mirror, ranges, label)

    print()


if __name__ == '__main__':
    main()
