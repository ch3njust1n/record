### Record
Module for recording experiments. Use to store results of a single experiment or multiple experiments. Logically separate model, logic and data. Ease pipeline training over hyperparameter sweep or multiple models by aggregating parameters with immutable Records and decouple visualization logic from training logic.

### Reproducibility
[The Machine Learning Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)

### Usage
Create a client e.g. `main.py`:

```
import argparse
from record import Record

def main():
	# Create a Record
	rec = Record()
	param = 'world'

	# Save an individual value
	rec.add('hello', param)

	# Save argparse parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', type=str)
	args = parser.parse_args()

	rec.add('a', args)
	rec.extend(args)

	# Or save dictionaries
	rec.extend({'my': 'dictionary'})

	print(rec.get('5e48b3a036f5e8a712e0e070'))


if __name__ == '__main__':
	main()
```

Run your client:
```
python main.py
```

