from torch import nn
from torchvision import models

class classifyResnet50(nn.Module):
    def __init__(self, numClass):
        super(classifyResnet50, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.linear = nn.Linear(1000, numClass)

    def forward(self, image):
        return self.linear(self.backbone(image))


class classifyDenseNet121(nn.Module):
    def __init__(self, numClass):
        super(classifyDenseNet121, self).__init__()
        self.backbone = models.densenet121(pretrained=True)
        self.linear = nn.Linear(1000, numClass)

    def forward(self, image):
        return self.linear(self.backbone(image))


class classifyInceptionv3(nn.Module):
    def __init__(self, numClass):
        super(classifyInceptionv3, self).__init__()
        self.backbone = models.inception_v3(pretrained=True, aux_logits=False)
        # self.backbone = models.inception_v3(pretrained=True)
        self.linear = nn.Linear(1000, numClass)

    def forward(self, image):
        # print(self.backbone(image))
        output = self.backbone(image)
        # if self.training:
        #     return self.linear(output.logits+output.aux_logits)
        return self.linear(output)