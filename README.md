### Record
Module for recording experiments. Use to store results of a single experiment or multiple experiments. Logically separate model, logic and data. Ease pipeline training over hyperparameter sweep or multiple models by aggregating parameters with immutable Records and decouple visualization logic from training logic.

### Reproducibility
[The Machine Learning Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)

### Usage
Create a client e.g. `main.py`:

```
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from record import Record
import matplotlib.pyplot as plt


def visualize(record):
	plt.plot(record['losses'])
	plt.title('Training Loss')
	plt.show()


def train(model_list, epochs, learning_rate, record):
	loss = F.nll_loss
	losses = []

	for i, model in enumerate(model_list):
		record.add(f'model_{i}', [p.tolist() for p in model.parameters()])
		for e in range(epochs):
			input = torch.randn(3, 5)
			target = torch.tensor([1, 0, 4])
			l = F.nll_loss(F.log_softmax(input, dim=1), target)
			losses.append(l.detach().tolist())

	record.add('losses', losses)


def main():

	# Save argparse parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Learning rate default: 0.01')
	parser.add_argument('--epochs', '-e', type=int, default=1, help='Training epochs default: 1')
	args = parser.parse_args()

	# Create a Record and save args
	experiment = dt.today().strftime("%b-%d-%Y")
	rec = Record(database='experiments', collection=experiment)
	rec.extend(args)

	# Or save dictionaries
	rec.extend({'desc': 'Example of how to use Record'})

	model_list = [nn.Linear(2,3), nn.Linear(2,3)]
	train(model_list, args.epochs, args.learning_rate, rec)

	# Save the record to get the ID
	rec_id = rec.save()

	# Retrieve the record and visualize it
	visualize(rec.get(rec_id))


if __name__ == '__main__':
	main()
```

Run your client:
```
python main.py
```

View experiments in MongoDB shell
```
db['Nov-27-2020'].find({}).pretty()
```

