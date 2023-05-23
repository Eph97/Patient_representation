import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# create dataframe with row ind numbering from 0 to 100,000
num_ind = 100000

prop = 0.5
actual = pd.DataFrame({'ind': range(num_ind)})

# draw x from bernoulli distribution with p = 0.5
actual['x'] = np.random.binomial(1, prop, size=num_ind)

simulate = actual.copy()
actual['gamma'] = np.repeat(np.random.uniform(0, 1, size=1), num_ind)
actual["beta_ATE"] = actual.gamma.mean() * actual.x.mean()
actual["beta"] = actual.gamma * actual.x


# simulate['x'] = np.random.binomial(1, prop, size=num_ind)
# number of simulated trials
num_sims = 1000
gammas = np.zeros(num_sims)

for i in range(num_sims):
    temp = pd.DataFrame()
    # draw gamma from uniform distribution with p = 0.5
    # simulate['gamma'] = np.random.uniform(0, 1, size=num_ind)
    gammas[i] = np.random.uniform(0, 1, size=1)
    simulate['gamma'] = np.repeat(gammas[i], num_ind)
    beta_ATE_i = "beta_ATE_" + str(i)
    beta_i = "beta_" + str(i)
    temp[beta_i] = simulate.gamma * simulate.x
    temp[beta_ATE_i] = simulate.gamma.mean() * simulate.x.mean()
    simulate = pd.concat([simulate, temp], axis=1)


# Path: simulate.py
# df['beta_ATE'] = df.filter(regex='beta_ATE').mean(axis=1)
# df['beta'] = df.filter(regex='beta_').mean(axis=1)
beta_ATE_mean = simulate.filter(regex='beta_ATE').mean(axis=1)
beta_mean = simulate.filter(regex='beta_[0-9]+').mean(axis=1)

# Variance of beta_ATE
# df['var_beta_ATE'] = df.filter(regex='beta_ATE').var(axis=1)
# var_beta_ATE = simulate.filter(regex='beta_ATE').var(axis=1)

# covariance of beta_ATE and beta by row
beta_ATEs =  simulate.filter(regex='beta_ATE')
betas = simulate.filter(regex='beta_[0-9]+')


# numpy linear regression

# generate sample data of x and y for linear regression
# y = betas.iloc[0].values.reshape(-1, 1)
# x = beta_ATEs.iloc[0].values.reshape(-1, 1)

c_i = np.zeros(num_ind)
for i in range(num_ind):
    y = betas.iloc[i].values.reshape(-1, 1)
    x = beta_ATEs.iloc[i].values.reshape(-1, 1)
    c_i[i] = ((x - x.mean())*(y - y.mean())).sum() / (np.square(x - x.mean())).sum()
    # model = LinearRegression(fit_intercept=True)
    # reg = model.fit(x, y)
    # c_i[i] = reg.coef_
    # c_i[i] = np.linalg.lstsq(np.vstack([x.T, np.ones(200)]).T , y, rcond=None)[0][0]

# np.linalg.lstsq(np.vstack([x.T, np.ones(200)]).T , y)[0][0]

# add c_i to dataframe
actual['c_i'] = c_i

# This isn't quite right. 
actual['beta_post'] = (actual.x - actual.c_i * actual.x.mean()) * gammas.mean()

error = ((actual.beta_post - actual.beta_ATE)**2).mean()
print(error)

# get average gamma if x = 1
# actual[actual.x == 1].c_i.mean()
