# search

Learning to search: A learning algorithm that learns a learning algorithm.

<p align="center"><img src="/result.png?raw=true" width="750"></p>

## Why?

Search is a basis of discovery. Any intelligent agent should be able to perform search --- randomly picking up a solution and evaluating it. An agent that successfully learns how to search should be able to find answer in a given input space. 

Suppose that we want to classify an image using a neural network that has been trained to search. Instead of providing a bunch of data-label pairs, the network only requires a minimum amount of data examples from each label class --- one for each should be enough. It is then tasked to sample configurations and evaluate those configurations in the example spaces; in other word, the network will mentally rotate, translate, or scale the image until it finds a configuration that fits the one of the previously memorized example in each class. The best of all class is the result we seek. 

This technique has the characteristics of both unsupervised learning and reinforcement learning. It does not require data-label examples to train like supervised learning, and trial-and-error is the heart of it just like reinforcement learning, though not as hard to train. All it needs are samples for each class, and it will try to match the input to the samples.

## Approach

tl,dr; an LSTM with policy-gradient.
Please refer to my blog for more detail.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* TensorFlow r1.0
* OpenCV
* numpy
* matplotlib

### Usage

```
usage: python main.py [-h] [--plot] [--test] [--cont] [--rate RATE] [--pg] [--dg]

optional arguments:
  -h, --help   show this help message and exit
  --plot       run plot
  --test       run test
  --cont       continue mode
  --rate RATE  learning rate
  --pg         policy gradient
  --dg         disconnected gradient
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
