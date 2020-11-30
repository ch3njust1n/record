### Record
Module for recording experiments. Use to store results of a single experiment or multiple experiments. Logically separate model, logic and data. Ease pipeline training over hyperparameter sweep or multiple models by aggregating parameters with immutable Records and decouple visualization logic from training logic.

### Reproducibility
[The Machine Learning Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)


### Installation
```
python -m pip install --upgrade git+https://github.com/ch3njust1n/record
```

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
from pprint import pprint
import numpy as np


'''
Visualize all training losses
inputs:
record (Record) Record object containing training data
'''
def visualize(_id, experiment):
	record = Record(_id=_id, database='experiments', collection=experiment)

	for model in record['models'].values():
		plt.plot(model['losses'], c=model['color'], label=model['label'])

	plt.title('Training Losses')
	plt.legend(loc='upper right')
	plt.show()


def train(model_list, epochs, learning_rate, record):
	loss = F.nll_loss
	all_models = {}

	for i, model in enumerate(model_list):
		# collect ith model's parameters
		losses = []
		model_key = f'model_{i}'
		model_data = {'params': [p.tolist() for p in model.parameters()]}

		for e in range(epochs):
			inp = torch.randn(3, 5)
			tgt = torch.tensor([1, 0, 4])
			l = F.nll_loss(F.log_softmax(inp, dim=1), tgt)
			losses.append(l.detach().tolist())

		# collect ith model's losses
		model_data['losses'] = losses
		model_data['color'] = np.random.rand(3,).tolist()
		model_data['label'] = model_key

		# store ith model's data
		all_models[model_key] = model_data
	
	# add all model data to record
	record.update(all_models, key='models')


def main():

	# Save argparse parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Learning rate default: 0.01')
	parser.add_argument('--epochs', '-e', type=int, default=10, help='Training epochs default: 1')
	args = parser.parse_args()

	# Create a Record and save args
	experiment = dt.today().strftime("%b-%d-%Y")
	rec = Record(database='experiments', collection=experiment)
	rec.update(args)

	# Or save dictionaries
	rec.update({'desc': 'Example of how to use Record'})

	model_list = [nn.Linear(2,3), nn.Linear(2,3)]
	train(model_list, args.epochs, args.learning_rate, rec)

	# Retrieve the record and visualize it
	visualize(rec._id, experiment)


if __name__ == '__main__':
	main()
```

#### Example Record
```
{
	"_id" : ObjectId("5fc2e0416f47c33abb406d28"),
	"python" : "3.8.5",
	"machine" : "x86_64",
	"processor" : "i386",
	"os" : "posix",
	"os_name" : "Darwin",
	"os_ver" : "17.7.0",
	"memory" : "4 GB",
	"storage" : "112 GB",
	"user" : "justin",
	"gpus" : [ ],
	"learning_rate" : 0.01,
	"epochs" : 10,
	"desc" : "Example of how to use Record",
	"models" : {
		"model_0" : {
			"params" : [
				[
					[
						0.18339073657989502,
						0.642157256603241
					],
					[
						0.40168291330337524,
						-0.010262012481689453
					],
					[
						-0.2690439820289612,
						0.16178512573242188
					]
				],
				[
					0.6983875632286072,
					-0.49078208208084106,
					0.2637331485748291
				]
			],
			"losses" : [
				2.0151028633117676,
				2.535722017288208,
				1.5341914892196655,
				2.4506757259368896,
				2.2694995403289795,
				2.6964614391326904,
				1.6367357969284058,
				1.6002613306045532,
				1.3573970794677734,
				1.1234484910964966
			],
			"color" : [
				0.6314553138423108,
				0.801809036714752,
				0.04997341760055618
			],
			"label" : "model_0"
		},
		"model_1" : {
			"params" : [
				[
					[
						0.4508076310157776,
						0.3764287829399109
					],
					[
						-0.438532292842865,
						-0.22260430455207825
					],
					[
						-0.6677511930465698,
						-0.3983822762966156
					]
				],
				[
					-0.6998584270477295,
					-0.02827918529510498,
					-0.3253032863140106
				]
			],
			"losses" : [
				1.8863905668258667,
				1.4876238107681274,
				1.4960846900939941,
				2.15963077545166,
				1.5538129806518555,
				2.233182668685913,
				1.670056700706482,
				1.169686198234558,
				1.0809082984924316,
				1.5914429426193237
			],
			"color" : [
				0.8655748005072542,
				0.5731576250720261,
				0.19201749680491897
			],
			"label" : "model_1"
		}
	}
}
```

Example output
![Plotting multiple losses from above code](https://github.com/ch3njust1n/record/blob/dev/assets/example_plot.png)


### Save ConfigParser
```
config = configparser.ConfigParser()
config.read(os.path.join('config', args.config+'.ini'))
rec.update(config)
```

Run your client:
```
python main.py
```

View experiments in MongoDB shell
```
db['Nov-27-2020'].find({}).pretty()
```

Delete collection
```
db['Nov-29-2020'].drop()
```
