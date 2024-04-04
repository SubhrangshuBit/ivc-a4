import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
manualSeed = 42
torch.manual_seed(manualSeed)
class FCN8s(nn.Module):
    """ 
    Implementation of Fully Convolutional Networks with stride 8.
    """
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # Encoder - this part reduces the spatial dimensions of the input image and increases the depth. it is also known as downsampling.
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16; instead of a fully connected layer, use a 1x1 convolution
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Decoder - this part upsamples the image to the original spatial dimensions through a series of deconvolutions
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False) # Upsample the output of the classifier 
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False) 
        self.upscore_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False) # reason why it is called FCN-8s

        # Skip connections - during upsampling, some of the spatial finer details are lost. To recover these details, skip connections are used. These allow the network to use the finer details from the lower layers.
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1) # takes output from the 4th pooling layer (depth 512)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1) # takes output from the 3rd pooling layer (depth 256)

    def forward(self, x):
        """ 
        The forward pass works as follows: 
        1. Pass the input image through the encoder part of the network.
        2. Pass the output of the encoder through the classifier part of the network.
        3. Upsample the output of the classifier to the original spatial dimensions, while also adding the finer details from the lower layers.
        """
        # x shape = (batch_size, 3, height, width)
        # Encoder 
        # print(f"Input shape: {x.shape}")
        block1 = self.features_block1(x) # shape = (batch_size, 64, height/2, width/2)
        # print(f"Block1 shape: {block1.shape}")
        block2 = self.features_block2(block1) # shape = (batch_size, 128, height/4, width/4)
        # print(f"Block2 shape: {block2.shape}")
        block3 = self.features_block3(block2) # shape = (batch_size, 256, height/8, width/8)
        # print(f"Block3 shape: {block3.shape}")
        block4 = self.features_block4(block3) # shape = (batch_size, 512, height/16, width/16)
        # print(f"Block4 shape: {block4.shape}")
        block5 = self.features_block5(block4) # shape = (batch_size, 512, height/32, width/32)
        # print(f"Block5 shape: {block5.shape}")

        # Classifier
        score = self.classifier(block5) # shape = (batch_size, num_classes, height/32, width/32)
        # print(f"Classifier Score shape: {score.shape}")

        # Decoder
        
        score_pool4 = self.score_pool4(block4) # skip connection 
        # print(f"Score_pool4 shape: {score_pool4.shape}")
        score_pool3 = self.score_pool3(block3) # skip connection
        # print(f"Score_pool3 shape: {score_pool3.shape}")

        upscore2 = F.interpolate(score, score_pool4.size()[2:], mode='bilinear', align_corners=True) # upsample the output of the classifier
        # print(f"Upscore2 shape: {upscore2.shape}")
        upscore_pool4 = self.upscore_pool4(upscore2 + score_pool4) # shape = (batch_size, num_classes, height/8, width/8)
        # print(f"Upscore_pool4 shape: {upscore_pool4.shape}")

        upscore_pool4 = F.interpolate(upscore_pool4, score_pool3.size()[2:], mode='bilinear', align_corners=True)
        # print(f"Reshaped upscore_pool4 shape: {upscore_pool4.shape}")
        # print(f"Fused: {(upscore_pool4 + score_pool3).shape}")
        upscore_final = self.upscore_final(upscore_pool4 + score_pool3) # shape = (batch_size, num_classes, height, width)
        # print(f"Upscore_final shape: {upscore_final.shape}")
        output = F.interpolate(upscore_final, x.size()[2:], mode='bilinear', align_corners=True)
        # print(f"Output shape: {output.shape}")
        return output # shape = (batch_size, num_classes, height, width)