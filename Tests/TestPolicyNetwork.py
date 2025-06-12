import sys
import os

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Policy_Network.GCNLayer import GCNLayer
from Policy_Network.Nonlinearity import NonlinearityLayer
from Policy_Network.DenseLayer import DenseLayer
from Policy_Network.PolicyNetwork import PolicyNetwork
from Training.TrainPolicy import TrainPolicy
from Tests.TestEnvironments import TestEnvironments

gcn = GCNLayer(input_dim=2, output_dim=4)
relu = NonlinearityLayer('relu')
dense = DenseLayer(input_dim=4, output_dim=2)
network = [gcn, relu, dense]
policy = PolicyNetwork(network)

trainer = TrainPolicy(TestEnvironments(), policy_network=policy, num_episodes=2000, lr=0.01)
trainer.train()
