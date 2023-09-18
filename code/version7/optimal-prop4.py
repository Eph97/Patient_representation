import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import numpy as np
import pandas as pd
from errorv7 import *

# p = 0.02
options = np.linspace(0.0, 1.0, 51)
# y = [errorMenWom10test(p, x, 0.5,3) for x in options]



def find_min_index1(p_1, p_2, p_3):
    temp = [test_sol(p_1,p_2,p_3,x,round((1-x)/2,3), round((1-x)/2,3)) for x in options]
    min_ind =temp.index(min(temp))
    return min_ind


def find_min_index2(p_1, p_2, p_3):
    temp = [test_sol(p_1,p_2,p_3,x,x, round((1-2*x),3)) for x in options[:51]]
    min_ind =temp.index(min(temp))
    return min_ind

# test1 = [find_min_index1(x,round((1-x)/2,3), round((1-x)/2,3)) for x in options]
# test2 = [find_min_index2(x,x, round((1-2*x),3)) for x in options[:51]]


 
k = 3 # number of groups
gamma_mean = np.ones(k) *0.0
gamma_cov=np.identity(k)
i = 0; j = 1
gamma_cov[i,j] = gamma_cov[j,i] = 0.7
# gamma_cov[2,2] = 1
gamma_cov[1,1] = 1
gamma_cov[0,0] = 0.9
var_cov = gamma_cov
p1 = 2/9
p2 = 3/9
p3 = 4/9

assert gamma_cov[i,j]**2 <= gamma_cov[i,i]*gamma_cov[j,j]

p = [p1, p2, p3]
domain =[[ i,j, (1-i-j) ] for i in options for j in options]

domain_clean = np.array([x_bar for x_bar in domain if (x_bar[-1] >= 0)])

error = [sol_var_cov_p(p, gamma_cov, x_bar) for x_bar in domain_clean]

min_ind = error.index(min(error))


df = pd.DataFrame(domain_clean, columns = ['x_1', 'x_2', 'x_3'])
df['error'] = error

print(df.iloc[min_ind])

# find row with minimum error in df


fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111, projection='3d')
x = df['x_1']
y = df['x_2']
z = df['error']

ax.plot_trisurf(df['x_1'], df['x_2'], df['error'], cmap='viridis', edgecolor='none');
# label axis
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('error')
ax.set_title(f"Error surface for 3 groups p_1={p[0]}, p_2={p[1]}, p_3={p[2]}")


plt.show()

