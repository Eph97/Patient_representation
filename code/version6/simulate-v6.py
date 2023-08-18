import numpy as np
import pandas as pd
import cmath
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
from update import *
# from errorv2 import deriv, splitderiv2, errorMenWom2, errorMenWom3
from errorv6 import *


# np.random.seed(366)
# np.random.seed(365)
# np.random.seed(36)
# create dataframe with row ind numbering from 0 to 100,000
num_ind = 10000
num_sims = 10000
k = 2 # number of groups
num_interactions = 2**k
gamma_mean = np.ones(num_interactions) *0.0
# gamma_mean[0] = 1.0
gamma_cov=np.identity(num_interactions)
gamma_cov[0,1] = gamma_cov[1,0] = 0.0
# gamma_cov=[[1,0],[0,1]]
error_df = pd.DataFrame()
error_val_df = pd.DataFrame()
deriv_df = pd.DataFrame()

options = np.linspace(0.0, 1.0, 10)


p_opt = [0.25, 0.5, 0.75]
gammas = np.random.multivariate_normal(mean=gamma_mean, cov=gamma_cov, size=num_sims)
# gammas[:,1] += gammas[:,0]
sample_gamma_mean = gammas.mean(axis=0)
var = gammas.var(axis=0)
v0 = var[0]
v1 = var[1]
v2 = var[2]
# cov = 0
g0 = gammas[:,0]
g1 = gammas[:,1]
g2 = gammas[:,2]
g3 = gammas[:,3]
max_diff = -np.inf
p_1 = 0.5
for p_0 in p_opt:
    for p_1 in p_opt:
        for p_2 in p_opt:
            for p_3 in p_opt:
                # prop = np.array([p_0, p_1])
                x = np.zeros((num_ind, num_interactions))
                x[:,0] = np.random.binomial(1, p_0, size=num_ind)
                x[:,1] = np.random.binomial(1, p_1, size=num_ind)
                x[:,2] = np.random.binomial(1, p_2, size=num_ind)
                x[:,3] = np.random.binomial(1, p_3, size=num_ind)

                x.mean(axis=0)

                for x_bar_0 in options:
                    for x_bar_1 in options:
                        for x_bar_2 in options:
                            for x_bar_3 in options:
                                x_bar = np.array( [x_bar_0, x_bar_1, x_bar_2, x_bar_3])
                                # number of simulated trials
                                betas = np.dot(x, gammas.T)
                                beta_ate = np.dot(x_bar, gammas.T)

                                # choose from simulate.1 and simulate.0 in proportion
                                c_i = np.zeros(num_ind)
                                for i in range(num_ind):
                                    c_i[i] = (np.cov(betas[i], beta_ate)[0][1]/ np.var(beta_ate))
                                beta_post = np.dot(x, sample_gamma_mean).reshape(num_ind,1) + c_i.reshape(num_ind,1)*((beta_ate - np.dot(x_bar, sample_gamma_mean)).reshape(1,num_sims))

                                error = (np.square(beta_post - betas)).mean()
                                error_temp = pd.DataFrame({'p_0': round(p_0,3), 'p_1': round(p_1,3),'x_bar_0': round(x_bar[0],3), 'x_bar_1' : round(x_bar[1],3), 'error': error}, index=[0])
                                error_df = pd.concat([error_df, error_temp] , ignore_index=True)


                                print(x.mean(axis=0), x_bar, error) # np.unique(c_i) , np.unique(c_i_2))

# output2 = error_df.copy()
# output_pivot = output2.pivot(index='p_1', columns='x_bar_1', values='error')

# output_anal = error_val_df.pivot(index='p_1', columns='x_bar_1', values='error')
# output_deriv = deriv_df.pivot(index='p_1', columns='x_bar_1', values='deriv')

# fig,ax =plt.subplots(3,3)
# ax[0,0].plot(options, output_pivot.iloc[0])
# ax[0,0].axvline(x=p_1_opt[0], color="blue")
# min_ind =output_pivot.iloc[0].to_list().index(min(output_pivot.iloc[0]))
# ax[0,0].axvline(x=output_pivot.columns[min_ind], color='red')

# min_ind =output_deriv.iloc[0].abs().to_list().index(min(output_deriv.iloc[0].abs()))
# ax[0,1].plot(options, output_anal.iloc[0])
# ax[0,1].axvline(x=output_deriv.columns[min_ind], color='red')
# ax[0,1].axvline(x=p_1_opt[0], color="blue")
# ax[0,2].plot(options, output_deriv.iloc[0])
# ax[0,2].axhline(y=0, color='blue')
# ax[0,0].title.set_text("Simulated Error p = "+ str(p_1_opt[0]))
# ax[0,1].title.set_text("Analytic Erorr")
# ax[0,2].title.set_text("derivative and critical points")

# ax[1,0].plot(options, output_pivot.iloc[1])
# ax[1,0].axvline(x=p_1_opt[1])
# min_ind =output_pivot.iloc[1].to_list().index(min(output_pivot.iloc[1]))
# ax[1,0].axvline(x=output_pivot.columns[min_ind], color='red')
# # ax[1,1].plot(options, output_anal.iloc[1])
# ax[1,2].plot(options, output_deriv.iloc[1])

# min_ind =output_deriv.iloc[1].abs().to_list().index(min(output_deriv.iloc[1].abs()))
# ax[1,1].plot(options, output_anal.iloc[1])
# ax[1,1].axvline(x=output_deriv.columns[min_ind], color='red')
# ax[1,1].axvline(x=p_1_opt[1], color="blue")
# ax[1,2].plot(options, output_deriv.iloc[1])
# ax[1,2].axhline(y=0, color='blue')
# ax[1,0].title.set_text("Simulated Error p = "+ str(p_1_opt[1]))
# ax[1,1].title.set_text("Analytic Erorr")
# ax[1,2].title.set_text("derivative and critical points")



# ax[2,0].plot(options, output_pivot.iloc[2])
# ax[2,0].axvline(x=p_1_opt[2])
# min_ind =output_pivot.iloc[2].to_list().index(min(output_pivot.iloc[2]))
# ax[2,0].axvline(x=output_pivot.columns[min_ind], color='red')
# # ax[2,0].text(0.5, 0.5, str(output_pivot.columns[min_ind]), fontsize=12)
# # ax[2,1].plot(options, output_anal.iloc[2])
# # ax[2,1].plot(options, output_deriv.iloc[2])

# min_ind =output_deriv.iloc[2].abs().to_list().index(min(output_deriv.iloc[2].abs()))
# ax[2,1].plot(options, output_anal.iloc[2])
# # ax[2,1].plot(options, output_deriv.iloc[2])
# ax[2,1].axvline(x=output_deriv.columns[min_ind], color='red')
# ax[2,1].axvline(x=p_1_opt[2], color="blue")
# ax[2,2].plot(options, output_deriv.iloc[2])
# ax[2,2].axhline(y=0, color='blue')
# ax[2,0].title.set_text("Simulated Error p = "+ str(p_1_opt[2]))
# ax[2,1].title.set_text("Analytic Erorr")
# ax[2,2].title.set_text("derivative and critical points")

# plt.show()




result = error_df.groupby(['p_0', 'p_1']).apply(lambda group: group.nsmallest(3, 'error'))
result.reset_index(drop=True, inplace=True)

# error_df.to_csv("error_df.csv", index=False)

# test = pd.read_csv("error_df.csv")
