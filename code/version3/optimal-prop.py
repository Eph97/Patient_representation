import matplotlib.pyplot as plt
import numpy as np
from errorv3 import *

# p = 0.02
options = np.linspace(0.0, 1.0, 51)
# y = [errorMenWom10test(p, x, 0.5,3) for x in options]



fig,ax =plt.subplots(3,3, sharex=True, sharey=True)

y0 = [opt_prop(p, 1,1, -0.5)[0] for p in options[1:]]
ax[0,0].title.set_text("Var(Women) = 1 Var(men) = 1\n(1,1,-0.5)")
ax[0,0].plot(options[1:], y0)
ax[0,0].plot(options[1:], options[1:], color="red")

y1 = [opt_prop(p, 1,1, -0.75)[0] for p in options[1:]]
ax[0,1].title.set_text("Var(Women)=1 Var(men)=0.5\n(1,1,-0.75)")
ax[0,1].plot(options[1:], y1)
ax[0,1].plot(options[1:], options[1:], color="red")

y2 = [opt_prop(p, 1,1, -1)[0] for p in options[1:]]
ax[0,2].title.set_text("Var(Women)=1 Var(men)=0\n(1,1,-1)")
ax[0,2].plot(options[1:], y2)
ax[0,2].plot(options[1:], options[1:], color="red")


y3 = [opt_prop(p, 1,1, -0.25)[0] for p in options[1:]]
ax[1,0].title.set_text("Var(Women)=1 Var(men)=1.5\n(1,1,-0.25)")
ax[1,0].plot(options[1:], y3)
ax[1,0].plot(options[1:], options[1:], color="red")

y4 = [opt_prop(p, 1,1, 0)[0] for p in options[1:]]
ax[1,1].title.set_text("Var(Women)=1 Var(men)=2\n(1,1,0)")
ax[1,1].plot(options[1:], y4)
ax[1,1].plot(options[1:], options[1:], color="red")


y6 = [opt_prop(p, 1,1, 0.5)[0] for p in options[1:]]
ax[2,0].title.set_text("Var(Women)=1 Var(men)=3\n(1,1,0.5)")
ax[2,0].plot(options[1:], y6)
ax[2,0].plot(options[1:], options[1:], color="red")

y7 = [opt_prop(p, 1,1, 1)[0] for p in options[1:]]
ax[2,1].title.set_text("Var(Women)=1 Var(men) = 0\n(1,1,1)")
ax[2,1].plot(options[1:], y7)
ax[2,1].plot(options[1:], options[1:], color="red")


y8 = [opt_prop(p, 1,2, -1)[0] for p in options[1:]]
ax[2,2].title.set_text("Var(women)=1 Var(men)=1\n(1,2,-1)")
ax[2,2].plot(options[1:], y8)
ax[2,2].plot(options[1:], options[1:], color="red")

plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
fig.supxlabel("Population Proportion")
fig.supylabel("Trial Proportion")
plt.show()
