# Deep Q Learning for Aggregate Computing Program Scheduling

This repository showcases an experiment on Deep Q Learning applied to schedule collective computation, specifically for aggregate computing programs.

## Structure
The system consists of N nodes, each executing a local aggregate program $P$ and governed by a scheduling policy $\pi$.
Both the program $P$ and the policy $\pi$ are identical for all nodes, creating a homogeneous system.
To store the nodes' experiences, a central replay buffer $D$ is utilized.
A central *agent* takes charge of training the scheduling policies of the nodes.

## Dynamics
Each node constructs a local state $s$ by observing its neighbourhood and the local output of the aggregate program $P$.
The window, with a fixed size of $w$, comprises the last $w$ output of a node.
The local state $s$ aids the node in selecting an action $a$ based on the scheduling policy $\pi$.
The action $a$ determines the next wake-up time for the node.
Following each execution of program $P$, the node receives a reward $r$ and stores the trajectory $(s, a, r, s')$ in the replay buffer $D$.
The global learner, responsible for training the nodes' scheduling policies, samples a batch of trajectories $(s, a, r, s')$ from the replay buffer $D` and utilizes them to train the scheduling policy $\pi$.

## Deep Q Learning Mode
The scheduling policy $\pi$ is trained using a Deep Q Learning algorithm.
I experimented with two state spaces:
- **State space 1**: The state space comprises the last $w$ states of the node, where $w$ represents the window size.
- **State space 2**: The state space includes the current state of the node and the previous state of the neighbourhood.

The approach is similar to the one presented in the initial contributions of QL for scheduling.
Rather than manually crafting the state space using -1, 0, and 1 values to represent increasing, stable, or decreasing trends of the local output, I employed a neural network to learn the trend of the local output.
However, the latter approach did not yield satisfactory results.

## Scenario 1: Gradient with changing source
In this scenario, the global program shared consists in computing the gradient of a source.
The source node change its value after 100 seconds (the total time is 200 seconds).
The initial source is selected randomly.

