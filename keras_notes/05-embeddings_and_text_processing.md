# Embeddings and Text-Processing Functions

## TODO

1. Understand when and why Embedding layers are used as inputs to LSTM layers.
2. Understand if you can share weights between two embedding layers. More
   precisely: imagine you have two inputs, `input1` and `input2`. These are
   connected to two Embedding Layers `embedding1` and `embedding2`. Ideally,
   these two embedding should learn the same weights, so it would be nice to
   have them `synchronized` between the two inputs.
3. Understand *sequence masking*.

## Introduction

In this section we start exploring how Embedding Layers and RNNs work. They
often appear together, but they don't necessarily have to. The main
applications are sequence classification, i.e. predicting the class of an input
sequence, and sequence-to-sequence processing. This is, for example, used in
automatic translation. One example of sequence classification, for which a full
implementation is available in the `examples` folder, is the "IMDB movie review
sentiment classification problem". In this example, each review is a vector of
words. Words have been mapped to integers reflecting how frequent the word is.
The most frequent word will have value 1, the second most frequent word will
have value 2 and so on.

Keras has a number of text processing utilities. These are powerful, but not
extensively documented. A gentle introduction to Embeddings can be found in
[Deep Learning wiht
Keras](https://www.safaribooksonline.com/library/view/deep-learning-with/9781787128422/a52db3cc-cc09-457f-b704-aac9ec3a001f.xhtml).
In this book the goal is building an Embedding with a vocabulary of 5000 (the
5000 most frequent words), an embedding size of 300 (this is the size of the
resulting embedding vector) and a window size of 1. This last parameter means
that the *context* around an input word consists of the word to its left and
the word to its right. In other words this value is the *skip* value of the
*skip-gram*, and it is symmetrical around the input word.

Note that this type of model requires *two* embedding layers: one for the input
word and one for the context. The two embedding layers have the same shape, and
they are combined with a dot product. This results in one single value that is
used as an input to a dense layer with a sigmoid activation. If the value is
above 0.5, it will be flattened to 1. If it is less than 0.5, it will be
flattened to zero. Below we show the code for the implementation.

```python
from keras.layers import Merge
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential

vocab_size = 5000embed_size = 300

# Create a model for the input word
word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size,
                         embeddings_initializer="glorot_uniform",
                         input_length=1))
word_model.add(Reshape((embed_size, )))

# Create a model for the context
context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size,
                  embeddings_initializer="glorot_uniform",
                  input_length=1))
context_model.add(Reshape((embed_size,)))

# Combine the two embeddings into one
model = Sequential()
model.add(Merge([word_model, context_model], mode="dot"))
model.add(Dense(1, init="glorot_uniform", activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer="adam")
```

The loss function is "mean_squared_error". If I understand correctly the cosine
similarity between the two embedding vectors (rather than the dot product,
which requires vector normalization), is compared with the class labels. Two
co-occurring words should have a cosine similarity close to 1 and should
therefore be close to the class label. Similarly for two non co-occurring
words.

## What is the real dimension of inputs and output is an Embedding?

Let's consider the following example. We have a set of protein sequences. We
consider only the proteins with up to 1000 AAs. The vocabulary size is 21,
corresponding to the 21 AAs. In order to make proteins of different lengths
comparable we introduce sequence padding, so that each sequence is of length
1000. If we turn each protein into a one-hot-encoded array, each protein will
be represented by a 1000 x 21 array.


## Understanding Keras utility functions

### The `preprocessing.text` module

The `preprocessing.text` module contains (at the time of writing) the following
functions:

1. `text_to_word_sequence`
2. `one_hot`
3. `Tokenizer` (this is a class, not a function)

Let's consider a simple but not completely trivial example:

```python
from keras.preprocessing import text, sequence

my_texts = [
    'This is a very simple text',
    'This is a somewhat more elaborate text',
    'This one is a confusingly verbose text with a lot of terms'
    ]

# Create an instance of the Tokenizer class
tokenizer = text.Tokenizer()

# Train the instance on our text
tokenizer.fit_on_texts(my_texts)

# Show the word index
print(tokenizer.word_index)

# Show the word count
print(tokenizer.word_count)

# Convert the whole text to a sequence of word indexes
tokenizer.texts_to_sequences(my_text) # Note the brackets!
```

The `Tokenizer` class is a key component for every text processing operation.
It reads the whole corpus, produces a mapping between words and integers,
counts the occurrences of each word, so that we can order the mapping based on
frequency. It has a number of methods that make text manipulation much simpler:

1. `fit_on_texts(texts)`: reads a *list* of texts and learns the mapping. After running this command the tokenizer instance has new attributes like `word_index` and `word_counts`.
2. `text_to_sequences(texts)`: given a *list* of texts, it returns a list of
   integer sequences based on the mapping learned by the tokenizer. There is
   also a generator version of this method, that does not return the whole
   output in one go.
3. `texts_to_matrix(texts)`: this method must be used with care. It takes a
   list of input texts, and for each text it return a vector with the same
   length as the vocabulary size. These vectors are arranged into a matrix of
   size `len(texts), num_words`. However, if a word occurs one or multiple times in a text, this will make no difference, and the presence of the word will be indicated by a `1`. In other words, the frequency of the word in a given text is ignored.

The `text_to_word_sequence(text)` function takes a text (i.e. a string, not a
list) and returns a list of words separated by the value of the `split`
argument, which defaults to a white-space.

```python
from keras.preprocessing import text

my_text = 'My neighbour Totoro'
my_seq = text_to_word_sequence(my_text)
# Returns ['my', 'neighbour', 'totoro']
```

The `one_hot(text)` function can be easily misleading. What this function does
is to take a text and to one-hot encoded it into a list of word indexes in a
vocabulary. In other words, if our vocabulary contains `n_words` words, the
encoding will be a vector of length `n_words` with ones in the indexes of the
words present in the text. This is the same as in `texts_to_matrix`, in that it
ignores the multiple occurrences of words in the same text.


### The `preprocessing.sequence` module

The `preprocessing.sequence` module contains (at the time of writing) the
following functions:

1. `pad_sequences`
2. `skipgrams`
3. `make_sampling_table`

