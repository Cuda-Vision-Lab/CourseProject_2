from CONFIG import config
import torch.nn as nn

class TransformerAutoEncoder(nn.Module):
    """
    Transformer Autoencoder module for both holistic and object centric scene representations
    """
    def __init__(self, encoder, decoder):

        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder       

        return
        
    def forward(self, images, masks=None, bboxes=None):
        
        encoded_features = self.encoder(images, masks, bboxes)
        
        recons, loss = self.decoder(
                                    encoded_features, 
                                    target=images
                                    )
        return recons, loss
    
class TransformerPredictor(nn.Module):
    """
    Transformer Predictor module for both holistic and object centric scene representations
    """
    def __init__(self, encoder, decoder, predictor, mode):
        
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor 
        self.mode = mode
        return
    
    def forward(self, images, masks=None, bboxes=None):
        
        # Note: input and target ranges are not used in the predictor. Only for the visualizations of the images in the notebook.
        
        encoded_features = self.encoder(images, masks, bboxes)
        if self.mode == 'inference':
            preds, loss, input_range, target_range = self.predictor(encoded_features, mode=self.mode) 
            return preds, loss, input_range, target_range
        else:
            preds, loss = self.predictor(encoded_features, mode=self.mode)
            return preds, loss
    
    
    
class OCVP(nn.Module):
    """
    OCVP module for both holistic and object centric scene representations
    """
    def __init__(self, encoder, decoder, predictor):
        
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        
        
    def forward(self, images, masks=None, bboxes=None):
        
        encoded_features = self.encoder(images, masks, bboxes)
        
        prediced_features, _ = self.predictor(encoded_features)

        recons, _ = self.decoder(
                                    prediced_features, 
                                    target=None
                                    )
        return recons