# draw several 
import pandas as pd
import matplotlib.pyplot as plt

output = pd.read_csv('output4.csv')

# sort by p_1 and x_bar_1
# output.sort_values(by=['p_1', 'x_bar_1'], inplace=True)
# output.to_csv('output.csv', index=False)

# reset index
# output.reset_index(drop=True, inplace=True)

# output.sort_values(by=['error'], inplace=True)


output.drop(columns=['p_0', 'x_bar_0'], inplace=True)

# round output columns x_bar_1 and p_1 to two decimal places
output['x_bar_1'] = output['x_bar_1'].round(2)
output['p_1'] = output['p_1'].round(2)

# pivot so that each row is a different p_1 and each column is a different x_bar_1
output_pivot = output.pivot(index='p_1', columns='x_bar_1', values='error')

# output_pivot.to_csv('output_pivot.csv', index=False)

# output_pivot = pd.read_csv('output_pivot.csv')

# for i in range(len(output_pivot)):

# add vertical line at x_bar_1 = 0.5
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(output_pivot.iloc[0])
axs[0,0].plot(output_pivot.iloc[1])
# axs[0,0].plot(output_pivot.iloc[2])
# axs[0,0].plot(output_pivot.iloc[3])
# axs[0,0].plot(output_pivot.iloc[4])
# axs[0,0].plot(output_pivot.iloc[5])
axs[0,0].legend(output_pivot.index[0:6])
axs[0,0].set_xlabel('x_bar_1')
# axs[0,1].set_title("error vs. x_bar_1 for different p_1")
# add legend to plot where each line is a different p_1

axs[0,1].plot(output_pivot.iloc[6])
axs[0,1].plot(output_pivot.iloc[7])
# axs[0,1].plot(output_pivot.iloc[8])
# axs[0,1].plot(output_pivot.iloc[9])
# axs[0,0].plot(output_pivot.iloc[10])
# axs[0,0].plot(output_pivot.iloc[11])
axs[0,1].legend(output_pivot.index[6:12])
axs[0,1].set_xlabel('x_bar_1')
axs[0,1].set_title("error vs. x_bar_1 for different p_1")

axs[1,0].plot(output_pivot.iloc[12])
axs[1,0].plot(output_pivot.iloc[13])
# axs[1,0].plot(output_pivot.iloc[14])
# axs[1,0].plot(output_pivot.iloc[15])
# axs[1,0].plot(output_pivot.iloc[16])
axs[1,0].legend(output_pivot.index[12:17])
axs[1,0].set_xlabel('x_bar_1')
axs[1,0].set_title("error vs. x_bar_1 for different p_1")

axs[1,1].plot(output_pivot.iloc[16])
axs[1,1].plot(output_pivot.iloc[17])
# axs[1,1].plot(output_pivot.iloc[18])
# axs[1,1].plot(output_pivot.iloc[19])
# axs[1,1].plot(output_pivot.iloc[20])
axs[1,1].legend(output_pivot.index[16:22])
axs[1,1].set_xlabel('x_bar_1')
axs[1,1].set_title("error vs. x_bar_1 for different p_1")
plt.show()
