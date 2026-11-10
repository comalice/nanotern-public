"""
Synthetic symbolic-reasoning corpus generator (oracle-verified).

Purpose (PLAN pieces 13/14, DECISIONS.md D11):
- a representative slice of symbolic text for BPE tokenizer training
  (--render mix), so symbolic corpora tokenize efficiently later;
- paired "good views" (symbolic expression <-> independent prose rendering
  of the SAME problem) for the piece-14 JEPA experiment;
- later: full curriculum corpora + retention probes (piece 13 arms).

Design cribbed from neural-set-union/experiments/symbolic_sets/corpus.py
(reimplemented here; that repo's engineering generator has no split/balance
guarantees either — those come when pieces 13/14 get their own protocols).

Every example is oracle-checked with plain Python sets at generation time.
No torch dependency: this runs anywhere, fast.

Usage:
  python -m nanotern.symbolic --n-examples 20000 --out /tmp/symbolic_slice.txt
  python -m nanotern.symbolic --n-examples 100 --render pairs --out /tmp/pairs.jsonl
"""
import argparse
import json
import random
from dataclasses import dataclass

# Word pools: natural-ish nouns for elements/sets. Deliberately mixed with
# synthetic identifiers (n<digits>) so the BPE sees both regimes.
ELEMENT_WORDS = (
    "ash cedar birch elm oak pine maple willow alder rowan hazel yew holly "
    "alum borax cobalt nickel pewter zinc copper tin lead iron quartz mica "
    "falcon heron ibis lark merlin osprey raven swift wren kite finch grebe "
    "basil sage thyme mint dill chive tansy rue sorrel cress lovage borage "
    "anvil brace chisel mallet auger plane rasp file level square plumb gauge"
).split()
SET_WORDS = (
    "grove orchard thicket copse stand hollow ridge meadow heath moor glen "
    "alloy ingot solder flux ore seam lode vein matte slag bloom dross "
    "flock covey brood clutch roost murmuration kettle cast drift rafter "
    "bed border patch row trench furrow flat coldframe nursery plot"
).split()

OPS_SET = ("union", "intersection", "difference")
PROSE_OP = {
    "union": ["combined with", "merged with", "joined together with", "pooled with"],
    "intersection": ["shared with", "in common with", "overlapping with", "both in"],
    "difference": ["excluding", "with ... removed", "minus", "leaving out"],
}


@dataclass(frozen=True)
class SymbolicProblem:
    universe: tuple          # element symbols
    names: tuple             # set names
    sources: tuple           # membership bit-rows, one per set name
    query: tuple             # nested (op, a, b) tree; int leaf = source index
    answer: frozenset        # oracle answer (element indices)

    def members(self, bits):
        return [s for s, b in zip(self.universe, bits) if b]

    def expr(self, node):
        if isinstance(node, int):
            return self.names[node]
        op, a, b = node
        return f"{op}({self.expr(a)},{self.expr(b)})"

    def symbolic_view(self):
        lines = ["universe=[" + ",".join(self.universe) + "]"]
        for name, row in zip(self.names, self.sources):
            lines.append(name + "={" + ",".join(self.members(row)) + "}")
        lines.append("query=" + self.expr(self.query))
        return "\n".join(lines)

    def prose_view(self, rng):
        """An independent rendering of the SAME problem (good view, piece 14).
        Different surface: prose, reordered declarations, varied phrasing."""
        idx = list(range(len(self.names)))
        rng.shuffle(idx)
        sentences = []
        for i in idx:
            members = self.members(self.sources[i])
            verb = rng.choice(("contains", "holds", "is made up of", "consists of"))
            if members:
                listing = ", ".join(members[:-1]) + (" and " + members[-1] if len(members) > 1 else "")
                sentences.append(f"The {rng.choice(('collection', 'group', 'set'))} {self.names[i]} {verb} {listing}.")
            else:
                sentences.append(f"The set {self.names[i]} is empty.")

        def prose_expr(node):
            if isinstance(node, int):
                return self.names[node]
            op, a, b = node
            left, right = prose_expr(a), prose_expr(b)
            if op == "difference" and rng.random() < 0.5:
                return f"{left} with {right} removed"
            if op == "intersection" and rng.random() < 0.5:
                return f"the items both in {left} and in {right}"
            return f"the {op} of {left} and {right}"

        sentences.append(f"Compute {prose_expr(self.query)}.")
        return " ".join(sentences)

    def answer_view(self):
        answer = [self.universe[i] for i in sorted(self.answer)]
        return "answer={" + ",".join(answer) + "}"


def _oracle(sources, query):
    sets = [frozenset(i for i, b in enumerate(row) if b) for row in sources]

    def evaluate(node):
        if isinstance(node, int):
            return sets[node]
        op, a, b = node
        left, right = evaluate(a), evaluate(b)
        if op == "union":
            return left | right
        if op == "intersection":
            return left & right
        if op == "difference":
            return left - right
        raise ValueError(f"unknown op {op}")

    return evaluate(query)


def generate(rng, universe=8, sources=3, operations=3, word_names=True):
    """One random problem with an exact oracle answer."""
    if word_names and rng.random() < 0.7:
        symbols = tuple(rng.sample(ELEMENT_WORDS, universe))
        names = tuple(rng.sample(SET_WORDS, sources))
    else:
        ids = rng.sample(range(1_000_000), universe + sources)
        symbols = tuple(f"n{i}" for i in ids[:universe])
        names = tuple(f"s{i}" for i in ids[universe:])
    rows = tuple(tuple(rng.randrange(2) for _ in range(universe)) for _ in range(sources))

    def tree(n):
        if n == 0:
            return rng.randrange(sources)
        left = rng.randrange(n)
        a, b = tree(left), tree(n - left - 1)
        if a == b:
            # union(x,x)/intersection(x,x) = x teaches nothing; difference is fine
            op = "difference"
        else:
            op = rng.choice(OPS_SET)
        return (op, a, b)

    query = tree(operations)
    return SymbolicProblem(symbols, names, rows, query, _oracle(rows, query))


def render_doc(problem, rng, style):
    if style == "symbolic":
        return problem.symbolic_view()
    if style == "prose":
        return problem.prose_view(rng)
    if style == "qa":
        return problem.symbolic_view() + "\n" + problem.answer_view()
    if style == "mix":
        return render_doc(problem, rng, rng.choice(("symbolic", "prose", "qa")))
    raise ValueError(style)


def main():
    p = argparse.ArgumentParser(description="Generate oracle-verified symbolic corpora")
    p.add_argument("--n-examples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--universe", type=int, default=8)
    p.add_argument("--sources", type=int, default=3)
    p.add_argument("--operations", type=int, default=3)
    p.add_argument("--render", choices=["symbolic", "prose", "qa", "mix", "pairs"],
                   default="mix",
                   help="'pairs' writes JSONL {symbolic, prose, answer} for piece 14")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    n_empty = 0
    with open(args.out, "w") as f:
        for _ in range(args.n_examples):
            prob = generate(rng, args.universe, args.sources, args.operations)
            assert prob.answer == _oracle(prob.sources, prob.query)  # paranoia
            n_empty += (len(prob.answer) == 0)
            if args.render == "pairs":
                f.write(json.dumps({
                    "symbolic": prob.symbolic_view(),
                    "prose": prob.prose_view(rng),
                    "answer": prob.answer_view(),
                }) + "\n")
            else:
                f.write(render_doc(prob, rng, args.render) + "\n\n")
    print(f"Wrote {args.n_examples} examples ({args.render}) to {args.out}; "
          f"empty-answer fraction: {n_empty / args.n_examples:.3f}")


if __name__ == "__main__":
    main()
