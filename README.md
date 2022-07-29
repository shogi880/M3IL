# M3IL：Multi-Modal Meta Imitation Learning (under review)


### Code Design

This section is for people who wish to extend the framework.

The code in designed in a pipelined fashion, where there are a list of
consumers that takes in a dictionary of inputs (from a previous consumer)
and then outputs a combined dictionary of the inputs and outputs of that
consumer.
For example:

```python
a = GeneratorConsumer(...)
b = TaskEmbedding(...)
c = MarginLoss(...)
d = Control(...)
e = ImitationLoss(...)
consumers = [a, b, c, d, e]
p = Pipeline(consumers)
```

```bash
0. prepare the xml asserts of mujoco.
1. run generator_instructions.py using bert-as-service
2. bash mill.sh 0.0000001 1 (lambda_embedding rate=1e-7, seed=1)
```
