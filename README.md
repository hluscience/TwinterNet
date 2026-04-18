# Neural Networks Guided Interaction Detection in Multiple Data Views

Welcome to the official implementation of our paper, "Neural Networks Guided Interaction Detection in Multiple Data Views." This repository contains a PyTorch-based framework for detecting and analyzing feature interactions between two data views.

### Components

#### Neural Network Models:
- **TwinterNet.py**: Implementation of the TwinterNet architecture designed for detecting complex feature interactions between two data views.
- **BetweenView_Interaction_detection.py**: Handles interaction detection specifically between different views, integral to the TwinterNet model.

#### Supporting Methods:
- **Other Methods**: A collection of other methods provided in the "other methods" folder, offering comparisons for interaction detection.

#### Simulations and Utilities:
- **Simulations**: Contains scripts to run various simulations for evaluating the performance of the TwinterNet model and other methods.
- **utils.py**: A set of utility functions that streamline data preprocessing, interaction detection, and evaluation tasks.

### Demo:
A user-friendly PyTorch demo is included, which provides a step-by-step guide to applying the TwinterNet model to your own interaction detection tasks.

### Getting Started:
1. **Clone the Repository**: Start by cloning the repository to your local machine.
2. **Run the Demo**: Follow the instructions in the demo to see how the TwinterNet model can be used for interaction detection in a two-view dataset.
3. **Explore the Code**: Dive into the `TwinterNet.py` and `BetweenView_Interaction_detection.py` files to understand the architecture and methodology.
4. **Run Simulations**: Use the scripts in the `simulations` folder to test the model under various conditions.

### Simulation Settings and Network Specifications

The following outlines the simulation configurations, model architectures, and training parameters used across TwinterNet and baseline methods in our experiments.

#### TwinterNet Architecture

- **Between-View Subnetworks**:
  - Each \( X_i\text{-}Z \) network begins with an interaction layer of **10 nodes**.
  - Followed by **two hidden layers**, each with **10 nodes**.
  - Applies to both configurations: \( p = 100, q = 20 \) and \( p = 500, q = 20 \).

- **Within-View Networks**:
  - When \( p = 500 \): the \( X \)-view network uses hidden layers: `[50, 30, 5]`
  - When \( q = 20 \): the \( Z \)-view network uses hidden layers: `[8, 5, 3]`

#### Baseline Architectures

- **MLP and paraACE**:
  - For \( p = 100, q = 20 \): `[100, 50, 20]`
  - For \( p = 500, q = 20 \): `[350, 150, 50]`

- **AutoInt**:
  - For \( p = 100, q = 20 \):
    - 3 attention blocks  
    - 2 attention heads  
    - Block shape: `[64, 64, 64]`
  - For \( p = 500, q = 20 \):
    - 4 attention blocks  
    - 4 attention heads  
    - Block shape: `[128, 128, 128, 128]`

#### Training Configuration (All Neural Methods)

- **Activation Function**: ReLU for all hidden layers
- **Optimizer**: Adam
  - TwinterNet, AutoInt, MLP:
    - Learning rate: `5e-3`
    - L1 regularization: `5e-5`
  - paraACE:
    - Adam optimizer with betas `(0.9, 0.99)`
- **Early Stopping**: Triggered after 10 epochs without validation improvement
- **Epochs**: 100
- **Batch Size**: 100
- **Hardware**: Training runs on 8 CPUs (AMD EPYC 7B12)
