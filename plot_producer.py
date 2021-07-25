import sys

from options.train_options import TestOptions
from util.plots import plot_losses
import pandas as pd

if __name__ == '__main__':
	opt = TestOptions().parse()
	lr = opt.lr
	epochs = opt.niter
	name = opt.name
	train_file = f"./loss_results/{name}_training_losses_lr_{lr}_epochs_{epochs}.csv"
	validation_file = f"./loss_results/{name}_validation_losses_lr_{lr}_epochs_{epochs}.csv"
	train_data = pd.read_csv(train_file)['L1'].tolist()
	val_data = pd.read_csv(validation_file)['L1'].tolist()
	plot_losses(train_data, val_data,name, lr, epochs)
