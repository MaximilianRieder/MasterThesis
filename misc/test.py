import torch
import torch.nn as nn

# define the predicted and ground truth probability distributions
predicted = torch.randn(1, 3, 256, 256)  # 3 classes, 256x256 image
ground_truth = torch.randn(1, 3, 256, 256)  # 3 classes, 256x256 image

p = torch.tensor([[-0.2, -0.1, -0.9]])
#p = torch.where(p == 1, torch.zeros_like(p), torch.ones_like(p))
p = p.neg()
q = torch.tensor([[0., 0., 1.]])
# define the KL divergence loss
kl_loss = nn.KLDivLoss(reduction='mean')

# compute the loss
loss = kl_loss(nn.functional.log_softmax(p, dim=1),
               nn.functional.softmax(q, dim=1))

print(loss)

#print(loss)
x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

# Switch ones and zeros
y = torch.where(x == 1, torch.zeros_like(x), torch.ones_like(x))
#print(x,y)

# Gute prediction -> soll penalized werden
pred_model = torch.tensor([[0.5, 0.3, -0.4]])
