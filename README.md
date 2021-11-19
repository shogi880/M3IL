# M3IL：Multi-Modal Meta Imitation Learning

### Running Paper Experiments

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

This allows the TecNet to be built in a modular way. For example, if one
wanted to do use a prototypical loss rather than a margin loss, then one would
only need to swap out one of these consumers. 

```
```
