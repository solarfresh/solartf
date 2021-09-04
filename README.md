# Introduction

This is a module based on tensorflow, and collect pipelines of various SOTA models. The purpose of this project is to conduct experiments conveniently. Moreover, it is going to be composed of main components including the data management, the model management, and the knowledge management.

# Data Management

Outcomes of machine learning algorithms are highly depend on input data. Once the data is biased, models are usually results in different stories. For example, equivariance is a good inductive bias for deep convolutional networks. The group equivariance in convolutional neural networks is an inductive bias that helps in the generalisation of the networks. It reduces the sample complexity by exploiting symmetries in the networks [1]. Therefore, functions of data management are preventing inductive biases, reducing data complexity, and standardizing collecting data such that experiemts are reproducible.

# Model Management

Machine learning algorithms are widely adopted in broaden domains nowadays. Algorithms and neural network archetectures are collected to convert images, the text and the sound into useful information. Additionally, model configurations are also record in order to improve experiments adaptively. Dynamical hyperparameter optimization or neural architecture optimization will be also implemented after bunch of experiments are conducted.

# Knowledge Management

Only is data collection or model building not enough to productize AI solutions until it can connect to domain knowledge and satisfy user needs. People usually start from asking questions after the observation. Next, several hypotheses are proposed to overcome challenges. Finally, information should be record and updated frequently, hence the searchable information yields solid references to avoid us falling into pitfalls.

# References

1. Taco S. Cohen, Max Welling, Group Equivariant Convolutional Networks, [arXiv:1602.07576](https://arxiv.org/abs/1602.07576).
