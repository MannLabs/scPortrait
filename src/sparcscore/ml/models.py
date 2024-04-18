import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict

class VGGBase(nn.Module): 
    """
    Base implementation of VGG Model Architecture. Can be implemented with varying number of 
    convolutional neural layers and fully connected layers.

    Args:
        nn (Module): Pytorch Module

    Returns:
        nn.Module: Pytorch Module
    """
    cfgs = {
        'A': [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        'B': [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M", 512, "M"],
        'D': [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        'E': [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    }

    cfgs_MLP = {
        'A': [2048, "M", 1024, "M",],
        'B': [1024, "M", 512, "M", 256, "M",],
    }

    def make_layers(self, cfg, in_channels, batch_norm = True):
        """
        Create sequential models layers according to the chosen configuration provided in 
        cfg with optional batch normalization for the CNN.

        Args: 
            cfg (list): A list of integers and “M” representing the specific
            VGG architecture of the CNN
            in_channels (int): Number of input channels
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.

        Returns:
            nn.Sequential: A sequential model representing the VGG architecture.
        """
        layers = []
        i = 0
        for v in cfg:
            if v == "M":
                layers += [(f"maxpool{i}", nn.MaxPool2d(kernel_size=2, stride=2))]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [(f"conv{i}", conv2d), (f"batchnorm{i}", nn.BatchNorm2d(v)), (f"relu{i}", nn.ReLU(inplace=True))]
                else:
                    layers += [(f"conv{i}", conv2d), (f"relu{i}", nn.ReLU(inplace=True))]
                in_channels = v
                i +=1
        return nn.Sequential(OrderedDict(layers))
    
    def make_layers_MLP(self, cfg_MLP, cfg, single_output = False):
        """
        Create sequential models layers according to the chosen configuration provided in 
        cfg for the MLP.

        Args:
            cfg_MLP (list): A list of integers and “M” representing the specific
            VGG architecture of the MLP
            cfg (list): A list of integers and “M” representing the specific
            VGG architecture of the CNN

        Returns: 
            nn.Sequential: A sequential model representing the MLP architecture.
        """
        # get output feature size of CNN with chosen configuration
        in_features = int(cfg[-2]) * 2 * 2
        
        layers = []
        i = 0
        for out_features in cfg_MLP:
            if out_features == "M":
                layers += [(f"MLP_relu{i}", nn.ReLU(True)), (f"MLP_dropout{i}", nn.Dropout())]
                i+=1         
            else:
                linear = (f"MLP_linear{i}", nn.Linear(in_features, out_features))
                layers += [linear]
                in_features = out_features
        
        if single_output:
            linear = (f"MLP_linear_final", nn.Linear(in_features, 1))
            layers += [linear]
        else:
            linear = (f"MLP_linear{i}_classes", nn.Linear(in_features, self.num_classes))
            layers += [linear]  
        
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.norm(x)
        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)
        x = self.softmax(x)
        return x
    
    def encoder(self, x):
        x = self.norm(x)
        x = self.features(x)
        return torch.flatten(x, 1)

class VGG1(VGGBase):
    """
    Instance of VGGBase with the model architecture 1.
    """
    def __init__(self,
                cfg = "B",
                cfg_MLP = "A",
                dimensions = 196,
                in_channels = 1,
                num_classes = 2,
                ):
        
        super(VGG1, self).__init__()

        #save num_classes for use in making MLP head
        self.num_classes = num_classes

        self.norm = nn.BatchNorm2d(in_channels)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.features = self.make_layers(self.cfgs[cfg], in_channels)
        self.classifier = self.make_layers_MLP(self.cfgs_MLP[cfg_MLP], self.cfgs[cfg])
        
    def vgg(cfg, in_channels,  **kwargs):
        model = VGG1(self.make_layers(self.cfgs[cfg], in_channels), **kwargs)
        return model

class VGG2(VGGBase):
    """
    Instance of VGGBase with the model architecture 2.
    """
    def __init__(self,
                cfg = "B",
                cfg_MLP = "B",
                dimensions = 196,
                in_channels = 1,
                num_classes = 2,
                ):
        
        super(VGG2, self).__init__()
        
        #save num_classes for use in making MLP head
        self.num_classes = num_classes

        self.norm = nn.BatchNorm2d(in_channels)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.features = self.make_layers(self.cfgs[cfg], in_channels)
        self.classifier = self.make_layers_MLP(self.cfgs_MLP[cfg_MLP], self.cfgs[cfg])
        
    def vgg(cfg, in_channels,  **kwargs):
        model = VGG2(self.make_layers(self.cfgs[cfg], in_channels), **kwargs)
        return model

class VGG2_regression(VGGBase):
    """
    Instance of VGGBase with regression output.
    """
    def __init__(self,
                cfg = "B",
                cfg_MLP = "B",
                dimensions = 196,
                in_channels = 1,
                num_classes = 2,
                ):
        
        super(VGG2_regression, self).__init__()

        self.norm = nn.BatchNorm2d(in_channels) 
        self.features = self.make_layers(self.cfgs[cfg], in_channels)
        self.classifier = self.make_layers_MLP(self.cfgs_MLP[cfg_MLP], self.cfgs[cfg], single_output = True)
        
    def vgg(cfg, in_channels,  **kwargs):
        model = VGG2_regression(self.make_layers(self.cfgs[cfg], in_channels), **kwargs)
        return model
    
    def forward(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

### CAE Model Architecture

class CAEBase(nn.Module):
    """
    Base implementation of CAE Model Architecture.
    """
    en_cfgs = {
        'A': [16, "M", 32, "M", 64, "M", 128, "M", 256, "M", 512, "M"],
        'A_deep': [32, "M", 64, "M", 128, "M", 256, "M", 512, 512, "M", 512, 512, "M"],
        'B': [16, "M", 32, "M", 64, "M", 128, "M", 256, "M", 256, "M"],
        'C': [16, "M", 32, "M", 64, "M", 128, "M", 256, "M", 512, "M",512, "M"],
        'D': [16, "M", 32, "M", 64, "M", 128, "M", 256, "M", 256, "M",256, "M"],
        'E': [16, "M", 32, "M", 64, "M", 128, "M", 256, "M", 256, "M",128, "M"],
        }
    
    de_cfgs = {
        "A": [512, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16, "U", 16],
        "A_deep": [512,512, "U", 512,512, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16],
        "B": [256, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16, "U", 16],      
        "C": [512, "U", 512, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16, "U", 16],
        "D": [256, "U", 256, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16, "U", 16],
        "E": [128, "U", 256, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16, "U", 16],
    }
    
    latent_dim = {
        "A": 2048,
        "A_deep": 2048,
        "B": 1024,
        "C": 512,
        "D": 256,
        "E": 128,
    }
    
    initial_decoder_depth = {
        "A": 512,
        "A_deep": 512,
        "B": 256,
        "C": 512,
        "D": 256,
        "E": 128,
    }
    
    def __init__(self,
                cfg = "B",
                in_channels = 5,
                out_channels = 5,
                ):
        
        super(CAEBase, self).__init__()
        
        self.norm = nn.BatchNorm2d(in_channels)
                        
        self.encoder = self.make_encoder(self.en_cfgs[cfg], in_channels) 
        self.decoder = self.make_decoder(self.de_cfgs[cfg], self.initial_decoder_depth[cfg], out_channels)
        
        
    def latent(self,x):
        x = self.norm(x)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x
        
    def forward(self, x):
        x = self.norm(x)
        x = self.encoder(x)

        shape2d = x.shape
        x = torch.flatten(x, 1)
        
        x = torch.reshape(x, shape2d)

        x = self.decoder(x)
        x = torch.sigmoid(x)

        return x
        
        
    def make_encoder(self, cfg, in_channels, batch_norm = True):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
            
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def make_decoder(self, cfg, in_channels, out_channels):
        layers = []
        
        for v in cfg:
            if v == "U":
                layers += [nn.Upsample(scale_factor = 2)]
            else:
                tconv2d = nn.ConvTranspose2d(in_channels,v,3,padding=1, stride=1)
                layers += [tconv2d, nn.ReLU(inplace=True)]
                in_channels = v
                
        layers += [nn.ConvTranspose2d(in_channels,out_channels,3,padding=1, stride=1)]
        
        return nn.Sequential(*layers)


### VAE Model Architecture

class VAEBase(nn.Module):
    """
    Base implementation of VAE Model Architecture. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_dim,
                 hidden_dims = None, **kwargs):
        
        super(VAEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512, 512, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= out_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        Args:
            input (torch.Tensor): Input tensor to the encoder with dimensions [N x C x H x W].

        Returns:
            list: A list containing mean and log variance components of the latent Gaussian distribution.
        """
        result = self.encoder(input)

        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z):
        """
        Maps the given latent codes onto the image space.

        Args:
            z (torch.Tensor): Latent code tensor with dimensions [B x D].

        Returns:
            torch.Tensor: Reconstructed tensor with dimensions [B x C x H x W].
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian with dimensions [B x D].
            logvar (torch.Tensor): Log variance of the latent Gaussian with dimensions [B x D].
            
        Returns:
            torch.Tensor: Reparameterized tensor with dimensions [B x D].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        """
        Performs the forward pass of the VAE model.

        The input passes through the encoder to obtain the mean and log variance.
        Then, the reparameterization trick is applied to generate the latent vector z.
        Finally, z is passed through the decoder to generate the reconstructed input.

        Args:
            input (torch.Tensor): The input tensor to be passed through the VAE model.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list: A list containing the reconstructed input, mean and log variance.
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]
    
    def loss_function(self, target, output,
                      *args, 
                      **kwargs):
        r"""
        Computes the VAE loss function.
        
        .. math::
            KL(N(\mu, \sigma), N(0, 1)) =  \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        Args:
            target (torch.Tensor): The target tensor.
            output (tuple): A tuple containing the reconstructed tensor, mean, and log variance.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.            

        Returns:
            dict: A dictionary containing the total loss, reconstruction loss, and KL divergence loss. 
        """
        recons = output[0]
        mu = output[1]
        log_var = output[2]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, target)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}


    """
   Instance of VGGBase with model architecture 2.
    """
     
    def __init__(self,
                cfg = "B",
                cfg_MLP = "B",
                dimensions = 196,
                in_channels = 5,
                num_classes = 2,
                ):
        
        super(VGG2, self).__init__()
        
        self.norm = nn.BatchNorm2d(in_channels)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.features = self.make_layers(self.cfgs[cfg], in_channels)
        self.classifier = self.make_layers(self.cfgs_MLP[cfg_MLP], self.cfgs[cfg])
        
    def vgg(cfg, in_channels,  **kwargs):
        model = VGG2(self.make_layers(self.cfgs[cfg], in_channels), **kwargs)
        return model
    

#### DEPRECATED FUNCTIONS FOR BACKWARD COMPATABILITY

class _VGG1(nn.Module):
    
    cfgs = {
        'A': [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        'B': [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M", 512, "M"],
        'D': [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        'E': [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    }
    
    
    def __init__(self,
                cfg = "B",
                dimensions = 196,
                in_channels = 5,
                num_classes = 2,
                 
                ):
        
        super(_VGG1, self).__init__()
        
        self.norm = nn.BatchNorm2d(in_channels)
        
        #self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.features = self.make_layers(self.cfgs[cfg], in_channels)
        
        self.classifier_1 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 2048),
        )
        
        self.classifier_2 = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
        )
            
        self.classifier_3 = nn.Sequential( 
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        
    def forward(self, x):
        x = self.norm(x)
        x = self.features(x)
        #x = self.avgpool(x)

        x = torch.flatten(x, 1)
        
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)

        x = self.softmax(x)
        return x
    
    def encoder(self, x):
        x = self.norm(x)
        x = self.features(x)
        return torch.flatten(x, 1)
    
    def encoder_c1(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        return x
    
    def encoder_c2(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        return x
    
    def encoder_c3(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        return x

    def make_layers(self, cfg, in_channels, batch_norm = True):       
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
            
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
        
    def vgg(cfg, in_channels,  **kwargs):
        model = _VGG1(make_layers(cfgs[cfg], in_channels), **kwargs)
        return model
    
class _VGG2(nn.Module):
    
    cfgs = {
        'A': [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        'B': [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M", 512, "M"],
        'D': [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        'E': [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    }
    
    def __init__(self,
                cfg = "B",
                dimensions = 196,
                in_channels = 5,
                num_classes = 2,
                 
                ):
        
        super(_VGG2, self).__init__()
        
        self.norm = nn.BatchNorm2d(in_channels)
        #self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.features = self.make_layers(self.cfgs[cfg], in_channels)
        
        self.classifier_1 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
        )
        
        self.classifier_2 = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
        )
            
        self.classifier_3 = nn.Sequential( 
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
        )

        self.classifier_4 = nn.Sequential( 
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.features(x)
        #x = self.avgpool(x)

        x = torch.flatten(x, 1)
        
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        x = self.classifier_4(x)

        x = self.softmax(x)
        return x
    
    def encoder(self, x):
        x = self.norm(x)
        x = self.features(x)
        return torch.flatten(x, 1)
    
    def encoder_c1(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        return x
    
    def encoder_c2(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        return x
    
    def encoder_c3(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        return x
    
    def encoder_c4(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        x = self.classifier_4(x)
        return x
  
    def make_layers(self, cfg, in_channels, batch_norm = True):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
            
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
        
    def vgg(self, cfg, in_channels,  **kwargs):
        model = _VGG1(make_layers(cfgs[cfg], in_channels), **kwargs)
        return model