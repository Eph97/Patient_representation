import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
from findMin import C, C_2
from functions import fAndG

np.random.seed(366)
# np.random.seed(365)
# np.random.seed(36)
# create dataframe with row ind numbering from 0 to 100,000
num_ind = 1000
num_sims = 1000
num_groups = 2
gamma_mean = np.ones(num_groups) *0.0
gamma_cov=np.identity(num_groups)
# gamma_cov=[[1,0],[0,1]]
error_df = pd.DataFrame()

# for p in np.linspace(0.1, 0.9, 9):
# for p in np.linspace(0.1, 0.9, 1):
options = np.linspace(0.0, 1.0, 101)
# for i in range(3):

p_0 = 1
x_bar_0 = 1
# p_1 = 0.4
# x_bar_1 = 0.4

p_1_opt = [0.4, 0.6, 0.75]
gamma_actual = np.random.multivariate_normal(mean=gamma_mean, cov=gamma_cov, size=1)
gammas = np.random.multivariate_normal(mean=gamma_mean, cov=gamma_cov, size=num_sims)
sample_gamma_mean = gammas.mean(axis=0)
vals = []

for p_1 in p_1_opt:
    prop = np.array([p_0, p_1])
    x = np.random.binomial(1, prop, size=(num_ind, num_groups))
    for x_bar_1 in options:
        actual = pd.DataFrame({'ind': range(num_ind)})
        x_bar = np.array([x_bar_0, x_bar_1])


        # gamma_actual = np.random.normal(0, 1, size=num_groups)
        actual["beta_ATE"] = np.dot(gamma_actual, x_bar)[0]
        actual["beta"] = np.dot(x,gamma_actual.T)

        # number of simulated trials
        betas = np.dot(x, gammas.T)
        beta_ate = np.dot(x_bar, gammas.T)

        # choose from simulate.1 and simulate.0 in proportion
        c_i = np.zeros(num_ind)
        c_i_2 = np.zeros(num_ind)
        for i in range(num_ind):
            # c_i[i] = ((betas[i] - betas[i].mean())*(beta_ate - beta_ate.mean())).sum() / (np.square(beta_ate - beta_ate.mean())).sum()
            # c_i[i] = (np.cov(betas[i], beta_ate)[0][1]/ np.var(beta_ate))
            c_i[i] = C(x[i], x_bar)
            # assert C_2(x[i,1], x_bar[1])[0][0] == C(x[i], x_bar)

        # assert (c_i - c_i_2 < 0.05).all()

        # This isn't quite right. 
        # beta_post = np.dot(x, sample_gamma_mean) + c_i*(actual.beta_ATE - np.dot(x_bar, sample_gamma_mean))
        beta_post = np.dot(x, sample_gamma_mean).reshape(num_ind,1) + c_i.reshape(num_ind,1)*((beta_ate - np.dot(x_bar, sample_gamma_mean)).reshape(1,num_sims))

        error = (np.square(beta_post - betas)).mean()
        # error = (np.square(beta_post - actual.beta)).mean()
        # print(error)
        error_temp = pd.DataFrame({'p_0': prop[0], 'p_1': round(prop[1],3), 'x_bar_0': x_bar[0], 'x_bar_1' : round(x_bar[1],3), 'error': error}, index=[0])
        error_df = pd.concat([error_df, error_temp] , ignore_index=True)
        print(prop, x_bar, np.unique(c_i), np.unique(c_i_2))
        # breakpoint()
    



output2 = error_df.copy()

output_pivot = output2.pivot(index='p_1', columns='x_bar_1', values='error')





def error(p, x_bar_1,a,b):
    error12 = (((a**2).mean()*(x_bar_1**4 - 2*p*x_bar_1**3 + p*x_bar_1**2)) + ((b**2).mean()*(p - 2*x_bar_1*p + x_bar_1**2)) + ((2*a*b).mean()*(2*p*x_bar_1**2 - p*x_bar_1 - x_bar_1**3)))/(1+ 2*x_bar_1**2+x_bar_1**4)
    return error12

def deriv(p, x_bar_1, a,b):
    deriv = (2 *(a* x_bar_1-b) *(2 *a* x_bar_1**2 + a* p*(1 - 3 *x_bar_1 - x_bar_1**2 + x_bar_1**3) + (b)* (p - x_bar_1 + 2* p* x_bar_1 - 3 *p* x_bar_1**2 + x_bar_1**3)))/(1 + x_bar_1**2)**3
    return deriv


def deriv2(p, x, a,b):
    deriv2 = (-4*x/(1 + x**2)**3) * (((a**2).mean()*(x**4 - 2*p*x**3 + p*x**2)) + ((b**2).mean()*(p - 2*x*p + x**2)) + ((2*a*b).mean()*(2*p*x**2 - p*x - x**3))) + (1/(1 + x**2)**2)*(((a**2).mean()*(4*x**3 - 6*p*x**2 + 2*p*x)) + ((b**2).mean()*(2*x - 2*p)) + ((2*a*b).mean()*(4*p*x - p - 3*x**2)))
    return deriv2

def deriv3(p,x,a,b):
    a2 = (a**2).mean()
    a1 = (a).mean()
    b2 = (b**2).mean()
    b1 = (b).mean()
    deriv3 = (-2) * ((a2*p*x**3- a2*2*p*x**4) + (b2*p*x - b2*2*(x**2)*p + b2*x**3) + (a1*b1*4*p*x**3 - a1*b1*2*p*x**2 - 2*a1*b1*x**4)) + \
            (2*a2*x**3 - 3*a2*p*x**2 + p*a2*x) + (b2*x - b2*p) + (a1*b1*4*p*x - a1*b1*p - 3*a1*b1*x**2) + \
    (a2*p*x**3- 3*a2*p*x**4 ) + (b2*x**3 - b2*p*x**2) + (4*a1*b1*p*x**2 - a1*b1*p*x**2 - 3*a1*b1*x**4)

    return deriv3

def deriv4(p,x,a,b):
    a2 = (a**2).mean()
    a1 = (a).mean()
    b2 = (b**2).mean()
    b1 = (b).mean()
    # deriv4 = 4* a2*p*x**4 -2*a2*p*x**3  - 2*b2*p*x + 4*b2*p*x**2 - 2*b2*x**3 - 8*a1*b1*p*x**3 + 4* a1*b1*p*x**2 + 4*a1*b1*x**4 + \
    #         (2*a2*x**3 - 3*a2*p*x**2 + p*a2*x) + (b2*x - b2*p) + (4*a1*b1*p*x - a1*b1*p - 3*a1*b1*x**2) + \
    #         (a2*p*x**3- 3*a2*p*x**4 ) + (b2*x**3 - b2*p*x**2) + (4*a1*b1*p*x**2 - a1*b1*p*x**2 - 3*a1*b1*x**4)
    # reduced
    # deriv4 = (x**4)*(4*a2*p + 4*a1*b1 - 3*a2*p - 3*a1*b1) + (x**3)*(2*a2 - 2*a2*p - 2*b2 - 8*a1*b1*p + a2*p + b2) + \
    #         (x**2)*(4*b2*p + 4*a1*b1*p - 3*a2*p - 3*a1*b1 - b2*p + 4*a1*b1*p - a1*b1*p) + x*(p*a2 - 2*b2*p + b2 + 4*a1*b1*p) - (b2*p+ a1*b1*p)
    deriv4 = (x**4)*(a2*p + a1*b1) + (x**3)*(2*a2 - a2*p - b2 - 8*a1*b1*p ) + \
            (x**2)*(3*b2*p + 7*a1*b1*p - 3*a2*p - 3*a1*b1) + x*(p*a2 - 2*b2*p + b2 + 4*a1*b1*p) - (b2*p+ a1*b1*p)
    return deriv4

def anal_min(p):
    one = - (np.sqrt(5*p**2 - 2*p + 1) - p+1)/(2*p)
    two = (p+ np.sqrt(5*p**2 - 2*p + 1) - 1)/(2*p)
    return np.array([one,two])

fig,ax =plt.subplots(3,3)
ax[0,0].plot(options, output_pivot.iloc[0])
ax[0,0].axvline(x=p_1_opt[0])
min_ind =output_pivot.iloc[0].to_list().index(min(output_pivot.iloc[0]))
ax[0,0].axvline(x=output_pivot.columns[min_ind], color='red')
# add text of exact value of min(output_pivot.iloc[0])
ax[0,0].text(0.5, 0.5, str(output_pivot.columns[min_ind]), fontsize=12)

ax[1,0].plot(options, output_pivot.iloc[1])
ax[1,0].axvline(x=p_1_opt[1])
min_ind =output_pivot.iloc[1].to_list().index(min(output_pivot.iloc[1]))
ax[1,0].axvline(x=output_pivot.columns[min_ind], color='red')
ax[1,0].text(0.5, 0.5, str(output_pivot.columns[min_ind]), fontsize=12)

ax[2,0].plot(options, output_pivot.iloc[2])
ax[2,0].axvline(x=p_1_opt[2])
min_ind =output_pivot.iloc[2].to_list().index(min(output_pivot.iloc[2]))
ax[2,0].axvline(x=output_pivot.columns[min_ind], color='red')
ax[2,0].text(0.5, 0.5, str(output_pivot.columns[min_ind]), fontsize=12)

for ind, p_1 in enumerate(p_1_opt):
    p = p_1
    # B = (gamma_actual - sample_gamma_mean).T
    B = (gammas - sample_gamma_mean).T
    prop = np.array([p_0, p_1])

    x = np.random.binomial(1, prop, size=(num_ind, num_groups))
    a = B[0]
    b = B[1]

    props = np.linspace(0,1,101)

    y_error = [error(p, prop, a, b) for prop in props]
    y_deriv1 = [deriv(p, prop, a, b).mean() for prop in props]
    y_deriv2 = [deriv2(p, prop, a, b) for prop in props]
    y_deriv4 = [deriv4(p, prop, a, b) for prop in props]
    y_deriv5 = np.array([[fAndG(B.var(axis=1), np.array([1,prop]), x_i)[0] for prop in props] for x_i in x]).mean(axis=0)

    # assert (np.array(y_deriv1) - np.array(y_deriv4) < 0.001).all()
    y_deriv = y_deriv4.copy()

    # find closest value to zero in 
    pos_y_error = [abs(x) for x in y_error]
    min_index_error = pos_y_error.index(min(pos_y_error))

    pos_y_deriv = [abs(x) for x in y_deriv]
    min_index_deriv = pos_y_deriv.index(min(pos_y_deriv))

    pos_y_deriv1 = [abs(x) for x in y_deriv1]
    min_index_deriv1 = pos_y_deriv1.index(min(pos_y_deriv1))

    pos_y_deriv4 = [abs(x) for x in y_deriv4]
    min_index_deriv4 = pos_y_deriv4.index(min(pos_y_deriv4))

    assert min_index_deriv1 == min_index_deriv4






    ax[ind,1].plot(props, y_error)
    ax[ind,1].axvline(x=props[min_index_error])
        
    ax[ind,2].plot(props, y_deriv)
    ax[ind,2].axvline(x=props[min_index_deriv])



plt.show()

# plot anal_min over options

# anal_min_g= [max(anal_min(x)) for x in props[10:]]
# # plot anal_min_g against props
# plt.plot(props[10:], anal_min_g)
# plt.show()
