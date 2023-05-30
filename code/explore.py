import pandas as pd

output = pd.read_csv('output.csv')

# sort by p_1 and x_bar_1
output.sort_values(by=['p_1', 'x_bar_1'], inplace=True)

# reset index
output.reset_index(drop=True, inplace=True)

output.sort_values(by=['error'], inplace=True)

output.drop(columns=['p_0', 'x_bar_0'], inplace=True)

# pivot so that each row is a different p_1 and each column is a different x_bar_1
output_pivot = output.pivot(index='p_1', columns='x_bar_1', values='error')

output_pivot.to_csv('output_pivot.csv')
