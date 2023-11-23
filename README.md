# Driver Profile Classification 

Federated learning is a technique that decentralizes deep learning processes across
client devices, offering distinct advantages by avoiding data aggregation and storage
on a central server and reducing the computational load on the central server. The
the methodology involves training models on individual clients, transmitting local model
weights to a central server, aggregating these local weights from multiple clients to
generate a global weight, and subsequently conduct further training on clients
utilizing these global weights 

However, despite these advantages, federated learning is not without its drawbacks,
primarily related to security vulnerabilities. Specifically, there are concerns that local
model weights can be compromised and an adversary may reconstruct a client's
original data using generative models, posing a substantial security risk, and
compromising the integrity of the federated learning system.

To address these security concerns, differential privacy is introduced at the model
training stage, utilizing an optimization technique known as differentially private
stochastic gradient descent (DPSGD). DPSGD introduces a regularization mechanism
to constrain the influence of individual data examples. When DPSGD is employed, the
regularization effect introduces noise that, while enhancing privacy preservation, can
potentially decrease accuracy or extend the training duration needed for convergence.
However, this trade-off allows for the selection of an optimal noise level that maximizes
privacy while minimizing accuracy reduction.

Privacy accounting is employed as a metric to monitor and quantify the level of privacy
achieved during the training process. This metric serves as a tool for classifying the
extent of privacy obtained during training, considering parameters such as epoch,
noise, and clipping bound of the DPSGD optimizer.

This thesis explores the training of models on three client devices for the task of driver
profile identification, incorporating varying levels of noise to determine the noise values
that offer maximum privacy without compromising accuracy. A central focus of this
research is to address the challenge of selecting the optimal noise level without
diminishing accuracy.


![image](https://media.github.tik.uni-stuttgart.de/user/3542/files/8cad1fd3-7e32-44d7-b2ad-a0111c112eb5)


## Model Architecture

![image](https://github.com/karthikziffer/Federated-Learning-Driver-Profile-Identification/assets/24503303/b15a52a1-99a8-4acc-a363-56d4ce342301)




## Execution commands

- Command to run the server docker container
  
![image](https://github.com/karthikziffer/Federated-Learning-Driver-Profile-Identification/assets/24503303/a46100c9-dff0-497e-971e-6a9d5733bacb)

- Command to run the client's docker container

![image](https://github.com/karthikziffer/Federated-Learning-Driver-Profile-Identification/assets/24503303/d594dd93-50ea-494a-b280-44ef94533585)


