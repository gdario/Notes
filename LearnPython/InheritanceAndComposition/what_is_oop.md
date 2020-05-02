# What is Object-Oriented Programming

Objects in a program are the "nouns", and they can have properties and behaviors. Behaviors can refer the properties.

- Classes are used to create objects.
- Classes define a *type*.
- To create an object from a class we *instantiate* the class.

Software design can take longer than coding. The most complex thing is how to make the various object interact with each other.

Properties are called `attributes`. There are two types:

1. `instance attributes` are unique to each object created.
2. `class attributes` are the same for every instance.

In the example below, `name` and `age` are instance attributes, while `species` is a class attribute.

```python
class Dog:
    species = 'mammal'

    def __init__(self, name, age):
        self.name = name
        self.age = age
```
