import matplotlib.pyplot as plt


def produce_plot(train_list, validation_list, plot_name, experiment_name, learning_rate, epochs):
	"""

	:param train_list:
	:param validation_list:
	:param plot_name:
	:type plot_name: str
	:param experiment_name:
	:param learning_rate:
	:param epochs:
	:return:
	"""
	epoch_range = range(1, len(train_list) + 1)
	# plotting the line 1 points
	plt.plot(epoch_range, train_list, label="Train")
	# plotting the line 2 points
	plt.plot(epoch_range, validation_list, label="Validation")
	plt.xlabel('Epochs')
	# Set the y axis label of the current axis.
	plt.ylabel('Value')
	# Set a title of the current axes.
	plt.title(f'{plot_name} per epoch [LR={str(learning_rate)}]')
	# show a legend on the plot
	plt.legend()
	# Display a figure.
	plot_name_mn = plot_name.lower().replace(' ', '_')
	plot_file = f"./plot_results/{experiment_name}_{plot_name_mn}_{str(learning_rate)}_epochs_{str(epochs)}.png"
	plt.savefig(plot_file)
	print(f"Saved plot in {plot_file}")
	plt.close()
