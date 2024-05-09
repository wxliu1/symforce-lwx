from timeit import default_timer as clock
from sage.all import var
var("x y z w")
e = (x+y+z+w)**60
t1 = clock()
g = e.expand()
t2 = clock()
print("Total time:", t2-t1, "s")
