from base.baseTrainer import baseTrainer
from CONFIG import config
import argparse
import logging
from model.ocvp import TransformerAutoEncoder, TransformerPredictor, OCVP
# from model.encoder import MultiModalVitEncoder
from model.decoder import VitDecoder
from model.predictor import Predictor
from model.holistic_encoder import HolisticEncoder
from model.oc_encoder import ObjectCentricEncoder
from utils.utils import load_model
  
def get_encoder(scene_rep, mode, mask_ratio ):
    if scene_rep == 'holistic':
        return HolisticEncoder(mode= mode ,mask_ratio = mask_ratio)
    
    elif scene_rep == 'oc':
        return ObjectCentricEncoder(mode= mode, mask_ratio = mask_ratio)
    else:
        raise ValueError(f"Invalid scene representation type: {scene_rep}")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Trainer for Autoencoder, Predictor, and Inference modes. Options:\n"
                    "  -a,  --ae         Enable autoencoder training mode\n"
                    "  -p,  --predictor  Enable predictor training mode\n"
                    "  -i,  --inference  Enable end-to-end inference mode\n"
                    "  -ac, --ackpt      Checkpoint path to pretrained autoencoder\n"
                    "  -pc, --pckpt      Checkpoint path to pretrained predictor"
                    "  -s,  --scene_rep  Select scene representation type. Options: holistic, oc"
    )
    parser.add_argument('-a',   '--ae', action='store_false', help='Enable autoencoder training mode')
    parser.add_argument('-p',   '--predictor', action='store_false', help='Enable predictor training mode')
    parser.add_argument('-i',   '--inference', action='store_false', help='Enable end-to-end inference mode')
    parser.add_argument('-ac',  '--ackpt', help='Checkpoint path to pretrained autoencoder')
    parser.add_argument('-pc',  '--pckpt', help='Checkpoint path to pretrained predictor')
    parser.add_argument('-s',   '--scene_rep', help='Scene representation type. Options: holistic, oc')
    
    parser.print_help()
    args = parser.parse_args()
    
    logging.info(f"Training configuration:")
    logging.info(f"  - Epochs: {config['training']['num_epochs']}")
    logging.info(f"  - Batch size: {config['data']['batch_size']}")
    logging.info(f"  - Learning rate: {config['training']['lr']}")
    logging.info(f"  - Patch size: {config['data']['patch_size']}")
    logging.info(f"  - Mask ratio: {config['vit_cfg']['mask_ratio']}")
    logging.info(f"  - Use masks: {config['vit_cfg']['use_masks']}")
    logging.info(f"  - Use bboxes: {config['vit_cfg']['use_bboxes']}")
    
    trainer = baseTrainer(config)
    
    if args.ae:
        # Create autoencoder model
        encoder = get_encoder(scene_rep = args.scene_rep, mode= 'training',mask_ratio = 0.75)
        decoder = VitDecoder(mode= 'training')
        
        model = TransformerAutoEncoder(encoder, decoder)
        
        #Train and save encoder and decoder checkpoints
        # trainer.setup_model(model=model)
        # trainer.train_model()
              
    elif args.predictor:
        
        if not args.ackpt:
            raise FileNotFoundError("Please specify the checkpoint to the pretrained AutoEncoder model")
            
        encoder = get_encoder(scene_rep = args.scene_rep, mode= 'predictor',mask_ratio = 0.0)
        
        # Load AE weights
        encoder,_,_,_= load_model(model= encoder, savepath= args.ackpt) # TODO: also pass optimizer here
        
        predictor = Predictor()
        
        model = TransformerPredictor(encoder, predictor)
        model.encoder.requires_grad_(False)
        
        # Load AE weights
        # ae_checkpoint = torch.load("ae_checkpoint.pth", map_location=device)
        # model_full.encoder.load_state_dict(ae_checkpoint['encoder_state_dict'])
        # model_full.decoder.load_state_dict(ae_checkpoint['decoder_state_dict'])

        # # Freeze encoder and decoder
        # model_full.encoder.requires_grad_(False)
        # model_full.decoder.requires_grad_(False)

        # # Training setup for predictor
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, model_full.parameters()),  # Only predictor params
        #     lr=0.001
        # )
        
        #Train and save predictor checkpoints
        # trainer.setup_model(model=model)
        # trainer.train_model()
        

    elif args.inference:
        
        if (not args.ackpt) or (not args.pckpt):
            raise FileNotFoundError("Please specify the checkpoint to both pretrained AutoEncoder and Predictor models")
        

        encoder = get_encoder(scene_rep = args.scene_rep, mode= 'inference',mask_ratio = 0.0)          
        decoder = VitDecoder(mode='inference')
        predictor = Predictor()
         
        encoder,_,_,_= load_model(model= encoder, savepath= args.ackpt) # TODO: also pass optimizer here
        decoder,_,_,_= load_model(model= decoder, savepath= args.ackpt) # TODO: also pass optimizer here
        predictor,_,_,_= load_model(model= predictor, savepath= args.pckpt) # TODO: also pass optimizer here
        
        model = OCVP(encoder, decoder, predictor)
        
        # do inference and save results somewhere. 
        # some inference.py that takes the above models and do the inference
        
    else:
        raise ValueError("Please specify a valid mode.")
        
    