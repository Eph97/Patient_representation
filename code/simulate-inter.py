import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# create dataframe with row ind numbering from 0 to 100,000
num_ind = 100000
num_sims = 1000
num_groups = 2
gamma_mean = np.ones(num_groups) * 0.0
gamma_cov=np.identity(num_groups)
# gamma_cov=[[1,0],[0,1]]
error_df = pd.DataFrame()

# for p in np.linspace(0.1, 0.9, 9):
# for p in np.linspace(0.1, 0.9, 1):
options = np.linspace(0.0, 1.0, 11)
# for i in range(3):

np.random.seed(365)
p_0 = 1
x_bar_0 = 1
# p_1 = 0.4
# x_bar_1 = 0.4
for p_1 in options:
    for x_bar_1 in options:
        sampled_inds = np.random.choice(num_ind, size=3, replace=False)
        print(p_0, p_1, x_bar_0, x_bar_1)
        # p_0, p_1, x_bar_0, x_bar_1 =  np.random.choice()
        prop = [p_0, p_1]
        x_bar = [x_bar_0, x_bar_1]
        actual = pd.DataFrame({'ind': range(num_ind)})

        # draw x from bernoulli distribution with p = 0.5
        x = np.random.binomial(1, prop, size=(num_ind,num_groups))

        simulate = actual.copy()
        # actual['gamma'] = np.repeat(np.random.uniform(0, 1, size=1), num_ind)
        # gamma_actual = np.random.normal(0, 1, size=num_groups)
        gamma_actual = np.random.multivariate_normal(mean=gamma_mean, cov=gamma_cov, size=1)
        actual["beta_ATE"] = np.dot(gamma_actual, x_bar)[0]
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
            temp[beta_ATE_i] = np.dot(x_bar, gammas[i])
            simulate = pd.concat([simulate, temp], axis=1)

        print("finished simulating")

        # Path: simulate.py
        # beta_ATE_mean = simulate.filter(regex='beta_ATE').mean(axis=1)
        # beta_mean = simulate.filter(regex='beta_[0-9]+').mean(axis=1)

        # covariance of beta_ATE and beta by row
        beta_ATEs =  simulate.filter(regex='beta_ATE').iloc[0]
        betas = simulate.filter(regex='beta_[0-9]+')


        c_i = np.zeros(num_ind)
        for i in range(num_ind):
            beta_samples = betas.iloc[i].values.reshape(-1, 1) # y
            beta_ate_samples = beta_ATEs.values.reshape(-1, 1) # x
            # denom = (np.square(beta_ate_samples - beta_ate_samples.mean())).sum()
            c_i[i] = ((beta_samples - beta_samples.mean())*(beta_ate_samples - beta_ate_samples.mean())).sum() / (np.square(beta_ate_samples - beta_ate_samples.mean())).sum()

        # add c_i to dataframe
        actual['c_i'] = c_i

        # This isn't quite right. 
        actual['beta_post'] = np.dot((x - (c_i.reshape(-1,1) * x_bar)), gammas.mean(axis=0)) + c_i * actual.beta_ATE


        error = (np.square(actual.beta_post - actual.beta_ATE)).mean()
        print(error)
        error_temp = pd.DataFrame({'p_0': prop[0], 'p_1': prop[1], 'x_bar_0': x_bar[0], 'x_bar_1' : x_bar[1], 'error': error}, index=[0])
        error_df = pd.concat([error_df, error_temp] , ignore_index=True)
        print(error_df)

        # Sampled beta_ATE and 3 betas
        fig, axs = plt.subplots(2, 2)
        axs[0,0].hist(beta_ATEs.values, bins=100)
        axs[0,0].set_title('beta_ATE')

        index = sampled_inds[0]
        axs[0,1].hist(betas.iloc[index], bins=100)
        axs[0,1].set_title('beta_' + str(sampled_inds[0]))

        index = sampled_inds[1]
        axs[1,0].hist(betas.iloc[index], bins=100)
        axs[1,0].set_title('beta_' + str(sampled_inds[1]))

        index = sampled_inds[2]
        axs[1,1].hist(betas.iloc[index], bins=100)
        axs[1,1].set_title('beta_' + str(sampled_inds[2]))
        # plt.hist(betas.iloc[sampled_inds[3]], bins=1000)
        # plt.show()
        plt.savefig("../plots/beta_ATE_hist-intercept-" + str(p_1) + "-" + str(x_bar_1) + ".png")


# create dataframe with column equal to x and column equal to c_i

# test = pd.DataFrame({'x_0': x[:,0], 'x_1': x[:,1], 'c_i': c_i})

# collpase test by c_i
# test.groupby('c_i').mean()

# np.unique(c_i, return_counts=True)

# drop rows if p_0 + p_1 > 1
# todays date
# date = datetime.datetime.now().strftime("%Y-%m-%d")
# error_df.to_csv("../data/error_df-" + date + ".csv", index=False)

# test = pd.read_csv("../data/error_df-" + date + ".csv")
# test[(test.p_0 + test.p_1 <= 1)&(test.x_bar_0 + test.x_bar_1 <= 1)].sort_values(by='error').head(10)


# plot histogram of c_i

