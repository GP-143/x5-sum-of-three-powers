# x5-sum-of-three-powers
C++ algorithms for searching integer solutions to x^5 + y^5 + z^5 = n.
# x^5 + y^5 + z^5 = n, 0 < n < 100, integers with |x|,|y|,|z| < 10^7  (searched with RANGE = 9,999,999)
# Every line below was re-checked with exact big-integer arithmetic.
# (s,z) candidates examined in total: 1166469125
#
# INFINITE FAMILIES (any integer a; any of the three slots; not enumerated):
#   n = 1  : (a, -a, 1)
#   n = 32 : (a, -a, 2)
#
# ALL OTHER SOLUTIONS (sorted x<=y<=z), format: n x y z
2 0 1 1
3 1 1 1
12 -17 13 16
30 -1 -1 2
31 -1 0 2
33 0 1 2
34 1 1 2
63 -1 2 2
64 0 2 2
65 1 2 2
96 2 2 2
#
# No solution exists (in this range) for any other n in 1 to 99 apart from the families above.
