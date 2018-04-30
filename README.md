# NLP

## Structure
- A1_model_correctness_testing: Files from TAs to check the model format
- data:
- eval: Model perplexity evaluation
- models: See models/model.py for an interface that all language models implement. These models are selected by command line parameters, trained by a sequence of samples, then print output corresponding to the spec.
- tst:
- utils: Constants/file loading/data handling
- main.py: Parsing instructions
- commands.txt: Sequence of commands to run to test the language model
- run_model.sh: Concatenates commands.txt into the model with a hard-coded seed and model
