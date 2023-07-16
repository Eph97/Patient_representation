import numpy as np
import pandas as pd
import cmath
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from findMin import C, C_2, C_cov
# from errorv2 import deriv, splitderiv2, errorMenWom2, errorMenWom3
from errorv3 import *


# np.random.seed(366)
# np.random.seed(365)
# np.random.seed(36)
# create dataframe with row ind numbering from 0 to 100,000
num_ind = 10000
num_sims = 1000
num_groups = 2
gamma_mean = np.ones(num_groups) *0.0
# gamma_mean[0] = 1.0
gamma_cov=np.identity(num_groups)
gamma_cov[0,1] = gamma_cov[1,0] = 0.0
# gamma_cov=[[1,0],[0,1]]
error_df = pd.DataFrame()
error_val_df = pd.DataFrame()
deriv_df = pd.DataFrame()

# for p in np.linspace(0.1, 0.9, 9):
# for p in np.linspace(0.1, 0.9, 1):
# options = np.linspace(0.0, 1.0, 101)
options = np.linspace(0.0, 1.0, 51)
# for i in range(3):

# p_1 = 0.4
# x_bar_1 = 0.4

p_1_opt = [0.25, 0.5, 0.75]
gammas = np.random.multivariate_normal(mean=gamma_mean, cov=gamma_cov, size=num_sims)
# gammas[:,1] += gammas[:,0]
gammas[:,1] = gammas[:,1] - gammas[:,0]
sample_gamma_mean = gammas.mean(axis=0)
# var = gammas.var(axis=0)
g0 = gammas[:,0]
g1 = gammas[:,1]
cov_mat = np.cov(g0, g1)
cov = cov_mat[0,1]
v0 = cov_mat[0,0]
v1 = cov_mat[1,1]
max_diff = -np.inf

for p_1 in p_1_opt:
    # prop = np.array([p_0, p_1])
    p_0 = 1
    x = np.zeros((num_ind, 2))
    x[:,1] = np.random.binomial(1, p_1, size=num_ind)
    x[:,0] = 1
    for x_bar_1 in options:
        actual = pd.DataFrame({'ind': range(num_ind)})
        x_bar = np.array([1, x_bar_1])
        # number of simulated trials
        betas = np.dot(x, gammas.T)
        beta_ate = np.dot(x_bar, gammas.T)

        # choose from simulate.1 and simulate.0 in proportion
        c_i = np.zeros(num_ind)
        c_i_2 = np.zeros(num_ind)
        for i in range(num_ind):
            # c_i[i] = (np.cov(betas[i], beta_ate)[0][1]/ np.var(beta_ate))
            # c_i[i] = C(x[i]*np.sqrt(var), x_bar*np.sqrt(var))
            c_i[i] = C_cov(x[i, 1], x_bar[1], v0, v1, cov)
            # assert C_2(x[i,1], x_bar[1])[0][0] == C(x[i], x_bar)

        # This isn't quite right. 
        beta_post = np.dot(x, sample_gamma_mean).reshape(num_ind,1) + c_i.reshape(num_ind,1)*((beta_ate - np.dot(x_bar, sample_gamma_mean)).reshape(1,num_sims))

        error = (np.square(beta_post - betas)).mean()
        # error_val = error2(x[:,1],x_bar_1, g0, g1)
        # error_val = errorMis3(x[:,1],x_bar_1, g0, g1)
        # error_val = errorMenWom4(x[:,1],x_bar_1, g0, g1)
        error_val = errorMenWom5(p_1,x_bar_1, v0, v1, cov)
        # error_val = errorMenWom7(p_1,x_bar_1, g0, g1)
        print(error, error_val, np.abs(error_val - error).round(5))
        # assert (np.abs(error_val - error) < 0.005)

        # deriv = deriv5(p_1, x_bar_1,g0, g1)
        # deriv = deriv7(p_1, x_bar_1,g0, g1)
        # deriv_val = deriv(p_1, x_bar_1)
        deriv_val = splitderiv2(p_1, x_bar_1, v0, v1, cov)
        # deriv = deriv10(p_1, x_bar_1)
        # deriv_val2 = deriv5(p_1, x_bar_1,g0, g1)
        # assert (np.abs(deriv - deriv_val2) < 0.005)
        # print(error, error_val)
        # assert (np.abs(error - error_val) < 0.005)
        # print(deriv, deriv_val2)
        # print(error)
        error_temp = pd.DataFrame({'p_0': p_0, 'p_1': round(p_1,3), 'x_bar_0': x_bar[0], 'x_bar_1' : round(x_bar[1],3), 'error': error}, index=[0])
        error_df = pd.concat([error_df, error_temp] , ignore_index=True)

        error_temp_anal = pd.DataFrame({'p_0': p_0, 'p_1': round(p_1,3), 'x_bar_0': x_bar[0], 'x_bar_1' : round(x_bar[1],3), 'error': error_val}, index=[0])
        error_val_df = pd.concat([error_val_df, error_temp_anal] , ignore_index=True)

        deriv_temp = pd.DataFrame({'p_0': p_0, 'p_1': round(p_1,3), 'x_bar_0': x_bar[0], 'x_bar_1' : round(x_bar[1],3), 'deriv': deriv_val}, index=[0])
        deriv_df = pd.concat([deriv_df, deriv_temp] , ignore_index=True)

        diff = np.abs(error_val - error).round(5)
        if diff > max_diff:
            diff_data = pd.DataFrame({'p_0': p_0, 'p_1': round(p_1,3), 'x_bar_0': x_bar[0], 'x_bar_1' : round(x_bar[1],3), 'diff': diff}, index=[0])
            max_diff = diff
            print("new max diff", max_diff)

        print(x.mean(axis=0), x_bar) # np.unique(c_i) , np.unique(c_i_2))

output2 = error_df.copy()
output_pivot = output2.pivot(index='p_1', columns='x_bar_1', values='error')

output_anal = error_val_df.pivot(index='p_1', columns='x_bar_1', values='error')
output_deriv = deriv_df.pivot(index='p_1', columns='x_bar_1', values='deriv')

fig,ax =plt.subplots(3,3)
ax[0,0].plot(options, output_pivot.iloc[0])
ax[0,0].axvline(x=p_1_opt[0], color="blue")
min_ind =output_pivot.iloc[0].to_list().index(min(output_pivot.iloc[0]))
ax[0,0].axvline(x=output_pivot.columns[min_ind], color='red')

min_ind =output_deriv.iloc[0].abs().to_list().index(min(output_deriv.iloc[0].abs()))
ax[0,1].plot(options, output_anal.iloc[0])
ax[0,1].axvline(x=output_deriv.columns[min_ind], color='red')
ax[0,1].axvline(x=p_1_opt[0], color="blue")
ax[0,2].plot(options, output_deriv.iloc[0])
ax[0,2].axhline(y=0, color='blue')
ax[0,0].title.set_text("Simulated Error p = "+ str(p_1_opt[0]))
ax[0,1].title.set_text("Analytic Erorr")
ax[0,2].title.set_text("derivative and critical points")

ax[1,0].plot(options, output_pivot.iloc[1])
ax[1,0].axvline(x=p_1_opt[1])
min_ind =output_pivot.iloc[1].to_list().index(min(output_pivot.iloc[1]))
ax[1,0].axvline(x=output_pivot.columns[min_ind], color='red')
# ax[1,1].plot(options, output_anal.iloc[1])
ax[1,2].plot(options, output_deriv.iloc[1])

min_ind =output_deriv.iloc[1].abs().to_list().index(min(output_deriv.iloc[1].abs()))
ax[1,1].plot(options, output_anal.iloc[1])
ax[1,1].axvline(x=output_deriv.columns[min_ind], color='red')
ax[1,1].axvline(x=p_1_opt[1], color="blue")
ax[1,2].plot(options, output_deriv.iloc[1])
ax[1,2].axhline(y=0, color='blue')
ax[1,0].title.set_text("Simulated Error p = "+ str(p_1_opt[1]))
ax[1,1].title.set_text("Analytic Erorr")
ax[1,2].title.set_text("derivative and critical points")



ax[2,0].plot(options, output_pivot.iloc[2])
ax[2,0].axvline(x=p_1_opt[2])
min_ind =output_pivot.iloc[2].to_list().index(min(output_pivot.iloc[2]))
ax[2,0].axvline(x=output_pivot.columns[min_ind], color='red')
# ax[2,0].text(0.5, 0.5, str(output_pivot.columns[min_ind]), fontsize=12)
# ax[2,1].plot(options, output_anal.iloc[2])
# ax[2,1].plot(options, output_deriv.iloc[2])

min_ind =output_deriv.iloc[2].abs().to_list().index(min(output_deriv.iloc[2].abs()))
ax[2,1].plot(options, output_anal.iloc[2])
# ax[2,1].plot(options, output_deriv.iloc[2])
ax[2,1].axvline(x=output_deriv.columns[min_ind], color='red')
ax[2,1].axvline(x=p_1_opt[2], color="blue")
ax[2,2].plot(options, output_deriv.iloc[2])
ax[2,2].axhline(y=0, color='blue')
ax[2,0].title.set_text("Simulated Error p = "+ str(p_1_opt[2]))
ax[2,1].title.set_text("Analytic Erorr")
ax[2,2].title.set_text("derivative and critical points")

plt.show()

