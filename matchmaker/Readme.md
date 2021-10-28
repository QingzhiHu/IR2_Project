# Matchmaker Runners

We have the following main points of running experiments in the matchmaker library:

- **[train.py](train.py)** Main training runner for both re-ranking & retrieval training of all models supported by matchmaker.

- **[eval.py](eval.py)** Allows to evaluate a re-ranking model, embedded with quantitative analysis along with ecai20_tk.py, neuralIR_encoder.py. 

- **[pre-train.py](pre-train.py)** Allows to pre-train model weights on an mlm or other self-supervised pre-training task.

- **[qualitative.py](qualitative.py)** Main script for qualitative analysis.

