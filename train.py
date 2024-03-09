import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import model
import load_data

from dataset import EnTamV2Dataset, BucketingBatchSampler

from model_multilayer_word2vec import EncoderDecoder
#from model_multilayer import EncoderDecoder
#from model import EncoderDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_dataloader, model, n_epochs, PAD_idx, learning_rate=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_idx)
    #criterion = nn.CrossEntropyLoss(ignore_index=PAD_idx)
    
    if not os.path.isdir("trained_models"):
        os.mkdir("trained_models")

    min_loss = float('inf')
    for epoch in range(1, n_epochs + 1):
        loss = 0
        for iter, batch in enumerate(train_dataloader):
            # Batch tensors: [B, SeqLen]
            input_tensor  = batch[0].to(device)
            input_mask    = batch[2].to(device)
            target_tensor = batch[1].type(torch.LongTensor).to(device)

            loss += train_step(input_tensor, input_mask, target_tensor,
                               model, optimizer, criterion)
        print('Epoch {} Loss {}'.format(epoch, loss / iter))
        
        epoch_loss = loss/float(len(train_dataloader)*len(batch[0]))
        # serialization
        if epoch % 10 == 0 or epoch_loss < min_loss:
            torch.save(model.state_dict(), "trained_models/MatthewHausknecht_epoch%d_loss%.5f.pt" % (epoch, epoch_loss))
            min_loss = epoch_loss

        # add gradient clipping
        for param in model._parameters:
            model._parameters[param] = torch.clip(model._parameters[param], min=-5, max=5) 

def train_step(input_tensor, input_mask, target_tensor, model,
               optimizer, criterion):
    optimizer.zero_grad()

    decoder_outputs, decoder_hidden = model(input_tensor, input_mask, target_tensor.size(1), None) #target_tensor)
    
    # Collapse [B, Seq] dimensions for NLL Loss
    loss = criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
        target_tensor.view(-1) # [B, Seq] -> [B*Seq]
    )

    loss.backward()
    optimizer.step()
    return loss.item()

def greedy_decode(model, dataloader):
    with torch.no_grad():
        for batch in dataloader:
            input_tensor  = batch[0].to(device)
            input_mask    = batch[2].to(device)
            target_tensor = batch[1].to(device)

            decoder_outputs, decoder_hidden = model(input_tensor, input_mask, max_len=target_tensor.shape[1])
            topv, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()
        
            for idx in range(input_tensor.size(0)):
                input_sent = train_dataset.vocab_indices_to_sentence(input_tensor[idx], "en")
                output_sent = train_dataset.vocab_indices_to_sentence(decoded_ids[idx], "ta")
                target_sent = train_dataset.vocab_indices_to_sentence(target_tensor[idx], "ta")
                print('Input:  {}'.format(input_sent))
                print('Target: {}'.format(target_sent))
                print('Output: {}'.format(output_sent))
                print()


if __name__ == '__main__':
    
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", help="Verbose flag for dataset stats", action="store_true")
    ap.add_argument("--nosymbols", "-ns", help="Symbols flag for eliminating symbols from dataset", action="store_true")
    ap.add_argument("--no_start_stop", "-nss", help="Remove START and STOP tokens", action="store_true")
    ap.add_argument("--morphemes", "-m", help="Morphemes flag for morphological analysis", action="store_true")
    ap.add_argument("--batch_size", "-b", help="Batch size (int)", type=int, default=64)
    ap.add_argument("--num_layers", "-n", help="Number of RNN layers (int)", type=int, default=3)
    ap.add_argument("--n_epochs", "-ne", help="Number of training epochs (int)", type=int, default=500)
    args = ap.parse_args()
   
    train_dataset = EnTamV2Dataset("train", symbols=not args.nosymbols, verbose=args.verbose, morphemes=args.morphemes, start_stop_tokens=not args.no_start_stop)
    eng_vocab, tam_vocab = train_dataset.return_vocabularies()
    
    PAD_idx = train_dataset.ignore_index
    SOS_token = train_dataset.bos_idx
    EOS_token = train_dataset.eos_idx
    
    val_dataset = EnTamV2Dataset("dev", symbols=not args.nosymbols, verbose=args.verbose, morphemes=args.morphemes, 
                                  vocabularies=(eng_vocab, tam_vocab), start_stop_tokens=not args.no_start_stop)
    #test_dataset = EnTamV2Dataset("test", symbols=not args.nosymbols, verbose=args.verbose, morphemes=args.morphemes, 
    #                              vocabularies=(eng_vocab, tam_vocab), start_stop_tokens=not args.no_start_stop)
    
    from torch.utils.data import DataLoader

    #train_bucketing_batch_sampler = BucketingBatchSampler(train_dataset.bucketing_indices, batch_size=args.batch_size)
    #train_dataloader = DataLoader(train_dataset, batch_sampler=train_bucketing_batch_sampler)
    val_bucketing_batch_sampler = BucketingBatchSampler(val_dataset.bucketing_indices, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_bucketing_batch_sampler)
    
    #val_bucketing_batch_sampler = BucketingBatchSampler(val_dataset.bucketing_indices, batch_size=args.batch_size)
    #val_dataloader = DataLoader(val_dataset, batch_sampler=train_bucketing_batch_sampler)

    hidden_size = 256
    input_size = len(eng_vocab)
    output_size = len(tam_vocab)
    
    train_dataset.eng_embedding = torch.tensor(train_dataset.eng_embedding).to(device)
    train_dataset.tam_embedding = torch.tensor(train_dataset.tam_embedding).to(device)
    model = EncoderDecoder(hidden_size, input_size, output_size, num_layers=args.num_layers,
                            weights=(train_dataset.eng_embedding, train_dataset.tam_embedding)).to(device)
    
    import glob
    import os

    model_chkpts = glob.glob("trained_models/*")
    if len(model_chkpts) > 0:
        model_chkpts = sorted(model_chkpts, reverse=True, key=lambda x: float(x.split('loss')[1].split('.pt')[0]))
        model.load_state_dict(torch.load(model_chkpts[-1]))
        
        print ("Loaded model from file: %s" % model_chkpts[-1])

    train(val_dataloader, model, n_epochs=args.n_epochs, PAD_idx=PAD_idx)
    greedy_decode(model, val_dataloader)
