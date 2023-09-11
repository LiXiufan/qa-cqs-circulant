import matplotlib.pyplot as plt
N_list = list(range(2, 9))
T_list = [6, 2, 17, 31, 34, 64, 128]
plt.title("Figure: Truncated Threshold - System Size", fontsize=10)
plt.plot(N_list, T_list, 'g-', linewidth=2.5, label="Truncated Threshold - System Size")
lgd = plt.legend()  # NB different 'prop' argument for legend
# lgd = plt.legend(fontsize=20) # NB different 'prop' argument for legend
lgd.set_title("Legend")
plt.xticks(N_list, N_list)
plt.xlabel("System Size '\Xi'", fontsize=10)
plt.ylabel("Truncated Threshold", fontsize=10)
plt.show()
