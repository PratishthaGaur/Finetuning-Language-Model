import torch
import torch.nn as nn 
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from utilities import Utilities
import argparse
from transformer import Encoder, Decoder,FeedForward,RefinedDecoder

seed = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(encoder,classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(encoder(X)[0].mean(dim=1))
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy

def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100,vocab_size=5755):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    total_loss = 0
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        logits,attn_maps = decoderLMmodel(X)
        loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
    
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def LanguageModeling(decoder,train_LM_loader,tokenizer,test_LM_loaders):
    
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    param=sum(p.numel() for p in decoder.parameters())
    print("Number of Parameters: ",param)
    
 
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        logits,attn_maps = decoder(xb)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), yb.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0:
            train_ppl = compute_perplexity(decoder, train_LM_loader, eval_iters,vocab_size=tokenizer.vocab_size)
            print(f"Step {i+1}, Train Perplexity: {train_ppl:.2f}")

            for president in test_LM_loaders: 
                president_ppl = compute_perplexity(decoder, test_LM_loaders[president], eval_iters,vocab_size=tokenizer.vocab_size)
                print(f"Step {i+1}, {president} Perplexity: {president_ppl:.2f}")
    
def ClassifierTraining(encoder, train_CLS_loader,tokenizer,test_CLS_loader):
    
    param=sum(p.numel() for p in encoder.parameters())
    print("Number of Parameters: ",param)
    classifier = FeedForward(n_embd).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs_CLS):
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = classifier(encoder(xb)[0].mean(dim=1))
            loss = loss_fn(outputs, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy = compute_classifier_accuracy(encoder, classifier, train_CLS_loader)
        print(f"Epoch {epoch+1}, Train Accuracy: {accuracy:.2f}%")


        accuracy = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
        print(f"Epoch {epoch+1}, Test Accuracy: {accuracy:.2f}%")
        print(f"Epoch {epoch+1}, Loss: {loss:.2f}%")



def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size) 
    
    parser = argparse.ArgumentParser(description='Run specific functions')
    parser.add_argument('-part2', action='store_true', help='Run LM modeling task')
    parser.add_argument('-part1', action='store_true', help='Run Classification task')
    parser.add_argument('-part3', action='store_true', help='Run Exploration task')
    args = parser.parse_args()
    if args.part1:
        #Loading Data
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=False)
        encoder = Encoder(tokenizer.vocab_size, n_embd, n_head, n_layer,block_size).to(device)
        #Finetuning
        ClassifierTraining(encoder, train_CLS_loader,tokenizer,test_CLS_loader)
        #Sanity Check
        u=Utilities(tokenizer, encoder)
        u.sanity_check("All should contribute, but the burden should not be excessive for any one group of programs or people.", block_size)


    elif args.part2:
        with open("speechesdataset/train_LM.txt", 'r', encoding='utf-8') as f:
                lmtrainText = f.read()
        LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        LM_loader = DataLoader(LM_dataset, batch_size=batch_size, shuffle=True)

        with open('speechesdataset/test_LM_obama.txt', 'r', encoding='utf-8') as f:
                lmtrainText = f.read()
        obama_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        obama_loader = DataLoader(obama_dataset, batch_size=batch_size, shuffle=True)

        with open('speechesdataset/test_LM_hbush.txt', 'r', encoding='utf-8') as f:
                lmtrainText = f.read()
        hbush_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        hbush_loader = DataLoader(hbush_dataset, batch_size=batch_size, shuffle=True)

        with open('speechesdataset/test_LM_wbush.txt', 'r', encoding='utf-8') as f:
                lmtrainText = f.read()
        wbush_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        wbush_loader = DataLoader(wbush_dataset, batch_size=batch_size, shuffle=True)
        
        test_LM_loaders = {'obama': obama_loader,'hbush': hbush_loader, 'wbush': wbush_loader}
        decoder = Decoder(tokenizer.vocab_size, n_embd, n_head, n_layer,block_size).to(device)
        #Finetuning
        LanguageModeling(decoder,LM_loader,tokenizer,test_LM_loaders)
        #Sanity Check
        # u=Utilities(tokenizer, decoder)
        # u.sanity_check("All should contribute, but the burden should not be excessive for any one group of programs or people.", block_size)

    
    elif args.part3:
        with open("speechesdataset/train_LM.txt", 'r', encoding='utf-8') as f:
                lmtrainText = f.read()
        LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        LM_loader = DataLoader(LM_dataset, batch_size=batch_size, shuffle=True)

        with open('speechesdataset/test_LM_obama.txt', 'r', encoding='utf-8') as f:
                lmtrainText = f.read()
        obama_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        obama_loader = DataLoader(obama_dataset, batch_size=batch_size, shuffle=True)

        with open('speechesdataset/test_LM_hbush.txt', 'r', encoding='utf-8') as f:
                lmtrainText = f.read()
        hbush_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        hbush_loader = DataLoader(hbush_dataset, batch_size=batch_size, shuffle=True)

        with open('speechesdataset/test_LM_wbush.txt', 'r', encoding='utf-8') as f:
                lmtrainText = f.read()
        wbush_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        wbush_loader = DataLoader(wbush_dataset, batch_size=batch_size, shuffle=True)
        
        test_LM_loaders = {'obama': obama_loader,'hbush': hbush_loader, 'wbush': wbush_loader}
        decoder = RefinedDecoder(tokenizer.vocab_size, n_embd, n_head, n_layer,block_size).to(device)
        #Finetuning
        LanguageModeling(decoder,LM_loader,tokenizer,test_LM_loaders)
        u=Utilities(tokenizer, decoder) 
        u.sanity_check("All should contribute, but the burden should not be excessive for any one group of programs or people.", block_size)

       
    else:
        print("No function specified. Use -part1, -part2, or -part3.")
    
if __name__ == "__main__":
    main()
