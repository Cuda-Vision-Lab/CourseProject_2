from CONFIG import config
import torch.nn as nn

class TransformerAutoEncoder(nn.Module):
    
    def __init__(self, encoder, decoder):
        
        # self.cfg = config
        # self.mode = mode
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder        
        
    def forward(self, images, masks=None, bboxes=None):
        
        encoded_features, all_masks, all_ids_restore = self.encoder(images, masks, bboxes)
        
        # target images are patchified in the decoder
        
        # Decode (only for image modality)
        
        recons, loss = self.decoder(
                                    encoded_features, 
                                    all_masks, 
                                    all_ids_restore, 
                                    target=images
                                    )
        return recons, loss
    
class TransformerPredictor(nn.Module):
    
    def __init__(self, encoder, predictor):
        
        # self.cfg = config
        # self.mode = mode
        super().__init__()
        
        self.encoder = encoder()
        self.predictor = predictor        
        ## mask ratio = 0.0
        
    def forward(self, images, masks=None, bboxes=None):
        
        encoded_features, all_masks, all_ids_restore = self.encoder(images, masks, bboxes)
    
        preds, loss = self.predictor(encoded_features) # predictor should return loss in the training mode
    
        return preds, loss
    
    
    
class OCVP(nn.Module):
    
    def __init__(self, encoder, decoder, predictor):
        
        super().__init__()
        
        self.encoder = encoder(mask_ratio = 0.0)
        self.decoder = decoder
        self.predictor = predictor
        
        
    def forward(self, images, masks=None, bboxes=None):
        
        encoded_features, all_masks, all_ids_restore = self.encoder(images, masks, bboxes)
        
        prediced_features = self.predictor(encoded_features)
        
        # Decode (only for image modality for now)
        recons, loss = self.decoder(
                                    prediced_features, 
                                    # all_masks['image'], 
                                    # all_ids_restore['image'], 
                                    # target=images
                                    )
        return recons, loss