import matplotlib.pyplot as plt


def plot_losses(train_list, validation_list, learning_rate, epochs):
	epoch_range = range(1, len(train_list) + 1)
	# plotting the line 1 points
	plt.plot(epoch_range, train_list, label="Train")
	# plotting the line 2 points
	plt.plot(epoch_range, validation_list, label="Validation")
	plt.xlabel('Epochs')
	# Set the y axis label of the current axis.
	plt.ylabel('Losses')
	# Set a title of the current axes.
	plt.title(f'Plots of losses per epoch [LR={str(learning_rate)}]')
	# show a legend on the plot
	plt.legend()
	# Display a figure.
	plot_file = f"./plot_results/losses_lr_{str(learning_rate)}_epochs_{str(epochs)}.png"
	plt.savefig(plot_file)
	print(f"Saved plot in {plot_file}")
	plt.close()
