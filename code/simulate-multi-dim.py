import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# create dataframe with row ind numbering from 0 to 100,000
num_ind = 100000
num_sims = 1000
num_groups = 2
gamma_mean = np.ones(num_groups) * 0.0
gamma_cov=np.identity(num_groups)
# gamma_cov=[[1,0],[0,1]]
error_df = pd.DataFrame()

for p in np.linspace(0.1, 0.9, 9):
    print(p)
    prop = [p, 1-p]
    actual = pd.DataFrame({'ind': range(num_ind)})

    # draw x from bernoulli distribution with p = 0.5
    x = np.random.binomial(1, prop, size=(num_ind,num_groups))

    simulate = actual.copy()
    # actual['gamma'] = np.repeat(np.random.uniform(0, 1, size=1), num_ind)
    # gamma_actual = np.random.normal(0, 1, size=num_groups)
    gamma_actual = np.random.multivariate_normal(mean=gamma_mean, cov=gamma_cov, size=1)
    actual["beta_ATE"] = np.dot(gamma_actual, x.mean(axis=0))[0]
    actual["beta"] = np.dot(x,gamma_actual.T)

    # simulate['x'] = np.random.binomial(1, prop, size=num_ind)
    # number of simulated trials
    gammas = np.zeros((num_sims, num_groups))

    for i in range(num_sims):
        temp = pd.DataFrame()
        # draw gamma from uniform distribution with p = 0.5
        # simulate['gamma'] = np.random.uniform(0, 1, size=num_ind)
        # gammas[i] = np.random.normal(0, 1, size=num_groups)
        gammas[i] = np.random.multivariate_normal(mean=gamma_mean, cov=gamma_cov, size=1)
        # simulate['gamma'] = np.repeat(gammas[i], num_ind)
        beta_ATE_i = "beta_ATE_" + str(i)
        beta_i = "beta_" + str(i)
        temp[beta_i] = np.dot(x, gammas[i])
        temp[beta_ATE_i] = np.dot(x.mean(axis=0), gammas[i])
        simulate = pd.concat([simulate, temp], axis=1)

    print("finished simulating")

    # Path: simulate.py
    # beta_ATE_mean = simulate.filter(regex='beta_ATE').mean(axis=1)
    # beta_mean = simulate.filter(regex='beta_[0-9]+').mean(axis=1)

    # covariance of beta_ATE and beta by row
    beta_ATEs =  simulate.filter(regex='beta_ATE')
    betas = simulate.filter(regex='beta_[0-9]+')


    c_i = np.zeros(num_ind)
    for i in range(num_ind):
        beta_samples = betas.iloc[i].values.reshape(-1, 1) # y
        beta_ate_samples = beta_ATEs.iloc[i].values.reshape(-1, 1) # x
        # try:
        c_i[i] = ((beta_samples - beta_samples.mean())*(beta_ate_samples - beta_ate_samples.mean())).sum() / (np.square(beta_samples - beta_samples.mean())).sum()
        # except RuntimeWarning:
            # print("index", i)

    # add c_i to dataframe
    actual['c_i'] = c_i

    # This isn't quite right. 
    actual['beta_post'] = np.dot((x - np.dot(c_i.reshape(-1,1), x.mean(axis=0).reshape(1,num_groups))), gammas.mean(axis=0)) + c_i * actual.beta_ATE


    error = (np.square(actual.beta_post - actual.beta_ATE)).mean()
    print(error)
    error_temp = pd.DataFrame({'p': p, 'error': error}, index=[0])
    error_df = pd.concat([error_df, error_temp] , ignore_index=True)
    print(error_df)

