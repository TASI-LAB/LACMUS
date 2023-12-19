import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantizer import vq, vq_st
import numpy as np

# function
def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


# Function to initialize the weights of our network
def weights_init(m):
    className = m.__class__.__name__
    if className.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", className)


# Structure of the embedding layer
class VQEmbedding(nn.Module):
    def __init__(self, K, D, center=None):
        super().__init__()
        # creating the embedding
        self.embedding = nn.Embedding(K, D)
        # weights belong to a uniform distribution
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)
        self.center = center

    # z_e_x --> latent code for the input image
    def forward(self, z_e_x):
        # converting BCHW --> BHWC
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        # Retrieving the indices corresponding to the input
        #print(f"z_e_x_: {z_e_x_.shape}; self.embedding.weight: {self.embedding.weight.shape}")
        
        if self.center is not None:
            ## Change to K-Means clustering 512 centriods coordinates
            # weights_kmeans = np.load("train_cluster_centers.npy").astype('float32')
            # weights_kmeans = torch.tensor(weights_kmeans, device="cuda:0")
            print("using clustered centers")
            latents, distance = vq(z_e_x_, self.center)
        else:
            latents, distance = vq(z_e_x_, self.embedding.weight)
        
        self.distance = distance
        return latents

    # z_e_x --> latent code for the input image
    def straight_through(self, z_e_x):
        # converting BCHW --> BHWC
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()

        # z_q_x --> latent code from the embedding nearest to the input code
        z_q_x_, indices, distance = vq_st(z_e_x_, self.embedding.weight.detach())
        self.distance = distance
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()
        
        # z_q_x_bar --> backprop possible
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()
        # used for generating the image (decoding)
        return z_q_x, z_q_x_bar


# Structure of the residual block
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


# Architecture of VQ-VAE
class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512, center=None):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Sequential(
            # Release soon
        )

        self.codeBook = VQEmbedding(K, dim, center)

        self.decoder = nn.Sequential(
            # Release soon
           
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codeBook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codeBook.embedding(latents).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        self.z_e_x = z_e_x
        latents = self.codeBook(z_e_x)
        self.latents = latents
        self.distance = self.codeBook.distance
        z_q_x_st, z_q_x = self.codeBook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x
    
    def generate_matrix(self, matrix):
        matrix = matrix.to(torch.int)
        self.latents = matrix
        z_q_x = self.codeBook.embedding(matrix).permute(0, 3, 1, 2) 
        #.permute(0, 3, 1, 2) 
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_q_x
    
    def generate_with_mask(self, x, concept_id):
        # generate samples with the given concept
        z_e_x = self.encoder(x)
        self.z_e_x = z_e_x
        latents = self.codeBook(z_e_x)
        sample_stack = None
        for i in range(latents.shape[1]):
            for j in range(latents.shape[2]):
                curlatents = latents.clone()
                curlatents[:,i,j] = concept_id
                
                z_q_x = self.codeBook.embedding(curlatents).permute(0, 3, 1, 2)
                x_tilde = self.decoder(z_q_x)
                
                if sample_stack is not None :
                    sample_stack = torch.concatenate(( sample_stack,x_tilde))
                else:
                    sample_stack = x_tilde
                
        z_q_x = self.codeBook.embedding(latents).permute(0, 3, 1, 2)
        x_tilde = self.decoder(z_q_x)
        sample_stack = torch.concatenate(( sample_stack,x_tilde))
        
        return sample_stack