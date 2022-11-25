""" This file uses the saved checkpoint file (bart_files/epoch=2-step=114198.ckpt) to generate the responses """

# imports
from transformers import BartTokenizer, BartForConditionalGeneration
import os
import pytorch_lightning as pl
import torch
import argparse

class LitModel(pl.LightningModule):
  # Instantiate the model
  def __init__(self, learning_rate, tokenizer, model, hparams):
    super().__init__()
    self.tokenizer = tokenizer
    self.model = model
    self.learning_rate = learning_rate
    self.hp = hparams
  
  def freeze_embeds(self):
    """ Freeze the positional embedding parameters of the model; adapted from finetune.py """
    freeze_params(self.model.model.shared)
    for d in [self.model.model.encoder, self.model.model.decoder]:
      freeze_params(d.embed_positions)
      freeze_params(d.embed_tokens)

  # Do a forward pass through the model
  def forward(self, input_ids, **kwargs):
    return self.model(input_ids, **kwargs)
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
    return optimizer

  def training_step(self, batch, batch_idx): 
    # Load the data into variables
    src_ids, src_mask = batch[0], batch[1]
    tgt_ids = batch[2]
    # Shift the decoder tokens right (but NOT the tgt_ids)
    decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

    # Run the model and get the logits
    outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
    lm_logits = outputs[0]
    # Create the loss function
    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    # Calculate the loss on the un-shifted tokens
    loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

    return {'loss':loss}

  def validation_step(self, batch, batch_idx):

    src_ids, src_mask = batch[0], batch[1]
    tgt_ids = batch[2]

    decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
    
    # Run the model and get the logits
    outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
    lm_logits = outputs[0]

    ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

    return {'loss': val_loss}
  
  # Method that generates text using the BartForConditionalGeneration's generate() method
  def generate_text(self, text, eval_beams, early_stopping = True, max_len = 40):
    """ Function to generate text """
    generated_ids = self.model.generate(
        text["input_ids"],
        attention_mask=text["attention_mask"],
        use_cache=True,
        decoder_start_token_id = self.tokenizer.pad_token_id,
        num_beams= eval_beams,
        max_length = max_len,
        early_stopping = early_stopping
    )
    return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]


def freeze_params(model):
    """ Function that takes a model as input (or part of a model) and freezes the layers for faster training """
    for layer in model.parameters():
        layer.requires_grade = False


def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>). """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def load_model():
    """ 
        Imports the fine-tuned model that we have already previously trained. 
        Returns the loaded model and the tokenizer for the input.
    """
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)
    hparams = argparse.Namespace()

    model = LitModel.load_from_checkpoint(os.path.join(os.getcwd(), "dialogue_generation/bart_files/epoch=2-step=11418.ckpt"), tokenizer=tokenizer, model=bart_model, hparams=hparams)

    # Put the model on eval mode
    model.to(torch.device('cpu'))
    model.eval()

    return model, tokenizer


def generate_reply(sentence, model_, tokenizer_):
    prompt_line_tokens = tokenizer_(text = sentence, max_length = 32, return_tensors = "pt", truncation = True)
    line = model_.generate_text(prompt_line_tokens, eval_beams = 4)

    return line[0].strip()