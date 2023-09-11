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
num_ind = 1000
num_sims = 10000
k = 3 # number of groups
# num_interactions = 2**k
gamma_mean = np.ones(k) *0.0
# gamma_mean[0] = 1.0
gamma_cov=np.identity(k)
gamma_cov[0,1] = gamma_cov[1,0] = 0.0
# gamma_cov=[[1,0],[0,1]]
error_df = pd.DataFrame()
error_val_df = pd.DataFrame()
deriv_df = pd.DataFrame()

options = np.linspace(0.0, 1.0, 101)


p0 = [0.1]
p1 = [0.3]
p2 = [0.6]

# p_opt = [0.33,0.34]
# p_opt = [0.25, 0.5, 0.75]
gammas = np.random.multivariate_normal(mean=gamma_mean, cov=gamma_cov, size=num_sims)
# gammas[:,1] += gammas[:,0]
sample_gamma_mean = gammas.mean(axis=0, keepdims=True)
var = gammas.var(axis=0)
v0 = var[0]
v1 = var[1]
v2 = var[2]
# cov = 0
# g0 = gammas[:,0]
# g1 = gammas[:,1]
# g2 = gammas[:,2]
max_diff = -np.inf
for p_0 in p0:
    for p_1 in p1:
        for p_2 in p2:
            print(np.abs(p_0 + p_1 + p_2 - 1))
            if np.abs(p_0 + p_1 + p_2 - 1) > 0.0001:
                # print("skipping (from p)")
                continue
            # prop = np.array([p_0, p_1])
            x = np.zeros((num_ind, k))
            # x[:,0] = np.random.binomial(1, p_0, size=num_ind)
            # x[:,1] = np.random.binomial(1, p_1, size=num_ind)
            # x[:,2] = np.random.binomial(1, p_2, size=num_ind)
            x[0:int(num_ind*p_0),0] = 1
            x[int(num_ind*p_0):int(num_ind*(p_0+p_1)), 1] = 1
            x[int(num_ind*(p_0+p_1)):, 2] = 1

            x.mean(axis=0)

            for x_bar_0 in options:
                for x_bar_1 in options:
                    for x_bar_2 in options:
                        if np.abs(x_bar_0 + x_bar_1 + x_bar_2 - 1) > 0.0001:
                            # print("skipping (from x_bar)")
                            continue
                        x_bar = np.array( [x_bar_0, x_bar_1, x_bar_2]).reshape(1,-1)
                        # number of simulated trials
                        betas = np.dot(x, gammas.T)
                        beta_ate = np.dot(x_bar, gammas.T)

                        # choose from simulate.1 and simulate.0 in proportion
                        c_i = np.zeros(num_ind)
                        c_i  =np.dot(x, x_bar.T)/np.dot(x_bar, x_bar.T)
                        # c_2i = np.zeros(num_ind)
                        # for i in range(num_ind):
                        #     c_2i[i] = (np.cov(betas[i], beta_ate)[0][1]/ np.var(beta_ate))
                            # c_i[i] = C(betas[i], beta_ate)
                        # beta_post = np.dot(x, sample_gamma_mean.T).reshape(num_ind,1) + c_i.reshape(num_ind,1)*((beta_ate - np.dot(sample_gamma_mean, x_bar.T)).reshape(1,num_sims))
                        # bpMinb = beta_post - betas
                        # error3 = (np.square(bpMinb)).mean()

                        error = MSE(x, x_bar, c_i, gammas)
                        error2 = anal_error(x, x_bar, gammas)
                        error_test = test(x, x_bar, gammas)
                        diff = np.abs(error - error2)
                        diff2 = np.abs(error - error_test)
                        # diff2 = np.abs(error - error3)

                        print(error, error_test, diff2)
                        # assert diff2 < 0.0001
                        # assert diff2 < 0.0001

                        trial_props = x_bar[0]
                        error_temp = pd.DataFrame({'p_1': round(p_0,3), 'p_2': round(p_1,3),'p_3': round(p_2,3),'x_bar_1': round(trial_props[0],3), 'x_bar_2' : round(trial_props[1],3), 'x_bar_3' : round(trial_props[2],3), 'error': error}, index=[0])
                        error_df = pd.concat([error_df, error_temp] , ignore_index=True)


                        print(x.mean(axis=0), x_bar, error) # np.unique(c_i) , np.unique(c_i_2))

# output2 = error_df.copy()
# output_pivot = output2.pivot(index='p_1', columns='x_bar_1', values='error')


# error_df.to_csv("error_df-final.csv", index=False)
# error_df = pd.read_csv("error_df-final.csv")

# result = error_df.groupby(['p_1', 'p_2', 'p_3']).apply(lambda group: group.nsmallest(3, 'error'))
# result.reset_index(drop=True, inplace=True)

# error_df = error_df.drop(columns="Unnamed: 0")

# select where p_1 = 0.25, p_2 = 0.25, p_3 = 0.5
graph_1 = error_df[(error_df['p_1'] == 0.1) & (error_df['p_2'] == 0.3) & (error_df['p_3'] == 0.6)]
graph_2 = error_df[(error_df['p_1'] == 0.33) & (error_df['p_2'] == 0.34) & (error_df['p_3'] == 0.33)]
graph_3 = error_df[(error_df['p_1'] == 0.33) & (error_df['p_2'] == 0.33) & (error_df['p_3'] == 0.34)]


# error_curve = [select(i) for i in graph_1['x_bar_1'].unique()]

# select graph_1 for indices where error_curve is not None
# graph_1 = graph_1[graph_1['x_bar_1'].isin(graph_1['x_bar_1'].unique()[~np.isnone(error_curve)])]

# error_curve1 = [x for x in error_curve if x is not None]


fig,ax =plt.subplots(3,3, sharex=True, sharey=True)

p_1 = graph_1['p_1'].values[0]
p_2 = graph_1['p_2'].values[0]
p_3 = graph_1['p_3'].values[0]
test0 = graph_1.loc[(graph_1.x_bar_2 == round((1-graph_1.x_bar_1)/2,3)) & (graph_1.x_bar_3 == round((1-graph_1.x_bar_1)/2,3)), ].reset_index(drop=True)
ax[0,0].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[0,0].plot(test0['x_bar_1'], test0['error'].values)
ax[0,0].axvline(x=p_1)
min_ind =test0.error.to_list().index(min(test0.error))
ax[0,0].axvline(x=test0.loc[min_ind, "x_bar_1"], color='red')


test0 = graph_1.loc[(graph_1.x_bar_1 == round((1-graph_1.x_bar_2)/2,3)) & (graph_1.x_bar_3 == round((1-graph_1.x_bar_2)/2,3)), ].reset_index(drop=True)
ax[1,0].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[1,0].plot(test0['x_bar_2'], test0['error'].values)
ax[1,0].axvline(x=p_2)
min_ind =test0.error.to_list().index(min(test0.error))
ax[1,0].axvline(x=test0.loc[min_ind, "x_bar_2"], color='red')


test0 = graph_1.loc[(graph_1.x_bar_1 == round((1-graph_1.x_bar_3)/2,3)) & (graph_1.x_bar_2 == round((1-graph_1.x_bar_3)/2,3)), ].reset_index(drop=True)
ax[2,0].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[2,0].plot(test0['x_bar_3'], test0['error'].values)
ax[2,0].axvline(x=p_3)
min_ind =test0.error.to_list().index(min(test0.error))
ax[2,0].axvline(x=test0.loc[min_ind, "x_bar_3"], color='red')


p_1 = graph_2['p_1'].values[0]
p_2 = graph_2['p_2'].values[0]
p_3 = graph_2['p_3'].values[0]
test1 = graph_2.loc[(graph_2.x_bar_2 == round((1-graph_2.x_bar_1)/2,3)) & (graph_2.x_bar_3 == round((1-graph_2.x_bar_1)/2,3)), ].reset_index(drop=True)
ax[0,1].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[0,1].plot(test1['x_bar_1'], test1['error'].values)
ax[0,1].axvline(x=p_1)
min_ind =test1.error.to_list().index(min(test1.error))
ax[0,1].axvline(x=test1.loc[min_ind, "x_bar_1"], color='red')

test1 = graph_2.loc[(graph_2.x_bar_1 == round((1-graph_2.x_bar_2)/2,3)) & (graph_2.x_bar_3 == round((1-graph_2.x_bar_2)/2,3)), ].reset_index(drop=True)
ax[1,1].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[1,1].plot(test1['x_bar_2'], test1['error'].values)
ax[1,1].axvline(x=p_2)
min_ind =test1.error.to_list().index(min(test1.error))
ax[1,1].axvline(x=test1.loc[min_ind, "x_bar_2"], color='red')


test1 = graph_2.loc[(graph_2.x_bar_1 == round((1-graph_2.x_bar_3)/2,3)) & (graph_2.x_bar_2 == round((1-graph_2.x_bar_3)/2,3)), ].reset_index(drop=True)
ax[2,1].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[2,1].plot(test1['x_bar_3'], test1['error'].values)
ax[2,1].axvline(x=p_3)
min_ind =test1.error.to_list().index(min(test1.error))
ax[2,1].axvline(x=test1.loc[min_ind, "x_bar_3"], color='red')


p_1 = graph_3['p_1'].values[0]
p_2 = graph_3['p_2'].values[0]
p_3 = graph_3['p_3'].values[0]
test2 = graph_3.loc[(graph_3.x_bar_2 == round((1-graph_3.x_bar_1)/2,3)) & (graph_3.x_bar_3 == round((1-graph_3.x_bar_1)/2,3)), ].reset_index(drop=True)
ax[0,2].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[0,2].plot(test2['x_bar_1'], test2['error'].values)
ax[0,2].axvline(x=p_1)
min_ind =test2.error.to_list().index(min(test2.error))
ax[0,2].axvline(x=test2.loc[min_ind, "x_bar_1"], color='red')

test2 = graph_3.loc[(graph_3.x_bar_1 == round((1-graph_3.x_bar_2)/2,3)) & (graph_3.x_bar_3 == round((1-graph_3.x_bar_2)/2,3)), ].reset_index(drop=True)
ax[1,2].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[1,2].plot(test2['x_bar_2'], test2['error'].values)
ax[1,2].axvline(x=p_2)
min_ind =test2.error.to_list().index(min(test2.error))
ax[1,2].axvline(x=test2.loc[min_ind, "x_bar_2"], color='red')

test2 = graph_3.loc[(graph_3.x_bar_1 == round((1-graph_3.x_bar_3)/2,3)) & (graph_3.x_bar_2 == round((1-graph_3.x_bar_3)/2,3)), ].reset_index(drop=True)
ax[2,2].title.set_text(f"p_1 = {p_1}, p_2 = {p_2},  p_3 = {p_3}")
ax[2,2].plot(test2['x_bar_3'], test2['error'].values)
ax[2,2].axvline(x=p_3)
min_ind =test2.error.to_list().index(min(test2.error))
ax[2,2].axvline(x=test2.loc[min_ind, "x_bar_3"], color='red')

plt.show()


