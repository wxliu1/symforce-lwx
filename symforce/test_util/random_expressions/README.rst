This module provides randomly sampled expression trees, which are useful primarily for testing and
benchmarking purposes.

The intent is to take a given set of

1) Leaf symbols, such as x_0, x_1, ..., x_10
2) Leaf constants, such as -2, -1, ..., 2
3) N-ary operators, such as ``lambda x: -x``, ``lambda x, y: x ** y``, or ``lambda x: sin(x)``

And generate expressions trees sampled uniformly from the set of all expression trees of a given
size that can be formed from these primitives.  There are a couple of details here.  First, size is
defined as total number of ops.  In practice, this is an approximate target, since SymPy and
SymEngine consolidate some ops on construction and hitting an exact target while accounting for
this is challenging.  Second by "uniformly sampled" we can mean either

1) Uniformly sampled over tree *structures*, and then uniformly sampled over operators and leaves
   once the structure is fixed, or
2) Uniformly sampled over all trees.

These are different since the number of unary and binary ops available will make some structures
more probable than others with definition 2.  We take option 1 by default, this is controlled by
the parameters ``p1`` and ``p2``.

The methodology and implementation here is based on "Deep Learning for Symbolic Mathematics", Appendix C:

https://arxiv.org/abs/1912.01412
