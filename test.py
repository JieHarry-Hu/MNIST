from uniformer import uniformer_small
import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = uniformer_small(pretrained=True)
channel_in = model.head.in_features
class_num=10
model.head = nn.Sequential(
                        nn.Linear(channel_in, 256),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(256, class_num),
                            nn.LogSoftmax(dim=1)
                        )
print(model.head)