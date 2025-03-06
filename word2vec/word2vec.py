import torch
import random
import matplotlib.pyplot as plt

def loadDataBasic():
    utterances = []
    with open('data/basic.csv', 'r') as dataset:
        for sample in dataset:
            utterances.append(f'<start> {sample.strip()} <stop>')
    return utterances

def build_vocabulary_matrix(utterances:list, dimensions:int):
    vocab_idx = {}
    for utterance in utterances:
        for word in utterance.split(' '):
            if word not in vocab_idx:
                vocab_idx[word] = len(vocab_idx)
    
    return torch.randn((len(vocab_idx), dimensions), requires_grad=True, dtype=float), vocab_idx

def plot_vocabulary(vocab_matrix, vocab_idx_inverted):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([x[0].detach().numpy() for x in vocab_matrix], [x[1].detach().numpy() for x in vocab_matrix], [x[2].detach().numpy() for x in vocab_matrix], c='b', marker='o')

    ax.set_title('Word Embeddings')
    for i in range(len(vocab_matrix)):
        ax.text(vocab_matrix[i][0].detach().numpy(), vocab_matrix[i][1].detach().numpy(), vocab_matrix[i][2].detach().numpy(), f'{vocab_idx_inverted[i]}', fontsize=10, color='red')

    plt.show()

def calc_probability(Vo, Vc, Wo, Wc, vocab_idx):
    '''
    Predict the probability that a given word appears in the context of some center word.

    Given a matrix of vocabulary words, Vo, and a matrix of center vocabulary words, Vc,
    calculate the probability that for a given word pair (Wo, Wc) Wo occurs inside the context window of a center word Wc.
    
    P(Wo | Wc) = exp(Wo @ Wc) / sum( exp(Wj @ Wc) )
    
    '''
    # calc word similarity
    similarity = Vo[vocab_idx[Wo]] @ Vc[vocab_idx[Wc]]    
    # scale the similarity score by softmax
    scale_factor = torch.sum(torch.exp(Vo @ Vc[vocab_idx[Wc]]))
    return torch.exp(similarity) / scale_factor

class EmbeddingsModel(torch.nn.Module):

    def __init__(self, utterances:list, dimensions:int):
        super(EmbeddingsModel, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=0)
        self.utterances = utterances
        self.dimensions = dimensions
        Vo, vocab_idx = build_vocabulary_matrix(utterances, dimensions=3)
        Vc, vocab_idx = build_vocabulary_matrix(utterances, dimensions=3)
        
        self.Vo = torch.nn.Parameter(Vo)
        self.Vc = torch.nn.Parameter(Vc)
        self.vocab_idx = vocab_idx

        self.vocab_idx_inverted = {v:k for (k,v) in self.vocab_idx.items()}

        # print initial weights
        print(self.vocab_idx)
        print(self.Vo)
        print(self.Vc)

    def forward(self, center:str):
        return self.logsoftmax(self.Vo @ self.Vc[self.vocab_idx[center]])

def train(train_epochs):
    utterances = loadDataBasic()

    model = EmbeddingsModel(utterances=utterances, dimensions=3)
    nll_loss = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(train_epochs):
        epoch_loss = 0
        for sample in utterances:
            tokenized_sample = sample.split()
            for i in range(1, len(tokenized_sample)-1):
                # evaluate and backprop center word with outside word to the left
                optimizer.zero_grad()
                output = model(tokenized_sample[i])
                loss = nll_loss(output, torch.tensor(model.vocab_idx[tokenized_sample[i-1]]))
                epoch_loss += loss
                loss.backward()
                optimizer.step()
                # evaluate and backprop center word with outside word to the right
                optimizer.zero_grad()
                output = model(tokenized_sample[i])
                loss = nll_loss(output, torch.tensor(model.vocab_idx[tokenized_sample[i+1]]))
                epoch_loss += loss
                loss.backward()
                optimizer.step()
        print(output, model.vocab_idx[tokenized_sample[i]])
        print("Sample loss", loss)
        print(f'Epoch {epoch+1} loss: {epoch_loss}')
        # shuffle the dataset between epochs
        random.shuffle(utterances)
        epoch += 1

    # print final weights
    print(model.vocab_idx)
    print(model.Vc)
    print(model.Vo)

    # plot the weights
    plot_vocabulary(vocab_matrix=model.Vc, vocab_idx_inverted=model.vocab_idx_inverted)
    plot_vocabulary(vocab_matrix=model.Vo, vocab_idx_inverted=model.vocab_idx_inverted)

def test_calc_probability():
    utterances = ["<start> he is sad", "<start> she is sad"]

    # initialize 3 dimensional word vocabularies
    Vo, vocab_idx = build_vocabulary_matrix(utterances, dimensions=3)
    # Vo =
    # [[-0.1115,  0.1204, -0.3696],
    #  [-0.2404, -1.1969,  0.2093],
    #  [-0.9724, -0.7550,  0.3239],
    #  [-0.1085,  0.2103, -0.3908],
    #  [ 0.2350,  0.6653,  0.3528]]
    print(Vo)
    Vc, vocab_idx = build_vocabulary_matrix(utterances, dimensions=3)
    # Vc = 
    # [[ 0.9728, -0.0386, -0.8861],
    #  [-0.4709, -0.4269, -0.0283],
    #  [ 1.4220, -0.3886, -0.8903],
    #  [-0.9601, -0.4087,  1.0764],
    #  [-0.4015, -0.7291, -0.1218]]
    print(Vc)

    # {'<start>': 0, 'he': 1, 'is': 2, 'sad': 3, 'she': 4}
    print(vocab_idx)
    

    # steps to calc probability that "he" and "she" appear next to each other
    # 1) lookup word vectors in vocab matrices
    # he (idx 1) Vo -> [-0.2404, -1.1969,  0.2093] = Wo
    # she (idx 4) Vc -> [-0.4015, -0.7291, -0.1218] = Wc
    # 2) take the dot product of Wo and Wc
    # Wo @ Wc = (-0.2404 * -0.4015) + (-1.1969 * -0.7291) + (0.2093 * -0.1218) = 0.94368765
    # 3) take the dot product of the center word with all outside words
    # Vo0 (<start>) -> [-0.1115,  0.1204, -0.3696]
    #   Vo0 @ wc = [-0.1115,  0.1204, -0.3696] @ [-0.4015, -0.7291, -0.1218] = (-0.1115 * -0.4015) + (0.1204 * -0.7291) + (-0.3696 * -0.1218) = 0.00200089
    # Vo1 (he) -> [-0.1115,  0.1204, -0.3696]
    #   Vo1 @ wc = [-0.2404, -1.1969,  0.2093] @ [-0.4015, -0.7291, -0.1218] = (-0.2404 * -0.4015) + (-1.1969 * -0.7291) + (0.2093 * -0.1218) = 0.94368765
    # Vo2 (is) -> [-0.9724, -0.7550,  0.3239]
    #   Vo2 @ wc =

    print(calc_probability(Vo, Vc, "he", "she", vocab_idx))

if __name__ == '__main__':
    print('test_calc_probability')
    torch.manual_seed(123)
    test_calc_probability()

    print('test_EmbeddingsModel')
    torch.manual_seed(123)
    model = EmbeddingsModel(utterances=["<start> he is sad", "<start> she is sad"], dimensions=3)
    prediction = model("she")
    print(prediction)

    torch.manual_seed(123)
    print('test_trainEmbeddings')
    train(train_epochs=500)