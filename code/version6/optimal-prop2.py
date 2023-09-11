import matplotlib.pyplot as plt
import numpy as np
from errorv6 import *

# p = 0.02
options = np.linspace(0.0, 1.0, 101)
# y = [errorMenWom10test(p, x, 0.5,3) for x in options]

fig,ax =plt.subplots(3,3, sharex=True, sharey=True)





p_1 = 0.5
p_2 = 0.25
p_3 = 0.25
test = [test_sol(p_1,p_2,p_3,x,round((1-x)/2,3), round((1-x)/2,3)) for x in options]
ax[0,0].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[0,0].plot(options[:], test)
ax[0,0].axvline(x=p_1)
min_ind =test.index(min(test))
ax[0,0].axvline(x=options[min_ind], color='red')

test = [test_sol(p_1,p_2,p_3,round((1-x)/2,3),x, round((1-x)/2,3)) for x in options]
min_ind =test.index(min(test))
ax[1,0].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[1,0].plot(options[:], test)
ax[1,0].axvline(x=p_1)
ax[1,0].axvline(x=options[min_ind], color='red')


test = [test_sol(p_1,p_2,p_3,round((1-x)/2,3), round((1-x)/2,3), x) for x in options]
min_ind =test.index(min(test))
ax[2,0].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[2,0].plot(options[:], test)
ax[2,0].axvline(x=p_1)
ax[2,0].axvline(x=options[min_ind], color='red')


p_1 = 0.25
p_2 = 0.5
p_3 = 0.25
test = [test_sol(p_1,p_2,p_3,x,round((1-x)/2,3), round((1-x)/2,3)) for x in options]
min_ind =test.index(min(test))
ax[0,1].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[0,1].plot(options[:], test)
ax[0,1].axvline(x=p_1)
ax[0,1].axvline(x=options[min_ind], color='red')

test = [test_sol(p_1,p_2,p_3,round((1-x)/2,3),x, round((1-x)/2,3)) for x in options]
min_ind =test.index(min(test))
ax[1,1].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[1,1].plot(options[:], test)
ax[1,1].axvline(x=p_1)
ax[1,1].axvline(x=options[min_ind], color='red')


test = [test_sol(p_1,p_2,p_3,round((1-x)/2,3), round((1-x)/2,3), x) for x in options]
min_ind =test.index(min(test))
ax[2,1].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[2,1].plot(options[:], test)
ax[2,1].axvline(x=p_1)
ax[2,1].axvline(x=options[min_ind], color='red')


p_1 = 0.25
p_2 = 0.25
p_3 = 0.5
test = [test_sol(p_1,p_2,p_3,x,round((1-x)/2,3), round((1-x)/2,3)) for x in options]
min_ind =test.index(min(test))
ax[0,2].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[0,2].plot(options[:], test)
ax[0,2].axvline(x=p_1)
ax[0,2].axvline(x=options[min_ind], color='red')

test = [test_sol(p_1,p_2,p_3,round((1-x)/2,3),x, round((1-x)/2,3)) for x in options]
min_ind =test.index(min(test))
ax[1,2].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[1,2].plot(options[:], test)
ax[1,2].axvline(x=p_1)
ax[1,2].axvline(x=options[min_ind], color='red')


test = [test_sol(p_1,p_2,p_3,round((1-x)/2,3), round((1-x)/2,3), x) for x in options]
min_ind =test.index(min(test))
ax[2,2].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[2,2].plot(options[:], test)
ax[2,2].axvline(x=p_1)
ax[2,2].axvline(x=options[min_ind], color='red')

plt.show()
