from detecto import core, utils, visualize
import torch
dataset = core.Dataset('/home/tandanml/ML/ExtractInfor/ExText/CrawDataCmndC/')
print(len(dataset))
model = torch.load("/home/tandanml/ML/ExtractInfor/ExText/DetectCMNDC1011") 
losses = model.fit(dataset, epochs=200, verbose=True, learning_rate=0.001)
torch.save(model,'DetectCMNDC1111')