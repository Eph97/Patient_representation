import matplotlib.pyplot as plt
from errorv2.py import *

v0 = 1
v1 = 1
# p = 0.02
# y = [errorMenWom10test(p, x, 0.5,3) for x in options]
y = [justMen(p, v0,v1)[0] for p in options[1:]]
plt.plot(options[1:], y)
plt.plot(options[1:], options[1:], color="red")
plt.ylabel("Trial Proportion")
plt.ylabel("Population Proportion")
plt.show()
