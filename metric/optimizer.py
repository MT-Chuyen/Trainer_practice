import torch

class otm():
    om= torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    return om