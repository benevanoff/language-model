import torch

def calc_probability(Vo, Vc, Wo, Wc, vocab_idx):
    '''
    Predict the probability that a given word appears in the context of some center word.

    Given a matrix of vocabulary words, Vo, and a matrix of center vocabulary words, Vc,
    calculate the probability that for a given word pair (Wo, Wc) Wo occurs inside the context window of a center word Wc.
    
    P(Wo | Wc) = exp(Wo @ Wc) / sum( exp(Wj @ Wc) )
    
    '''
    # calc word similarity
    similarity = Vo[vocab_idx[Wo]] @ Vc[vocab_idx[Wc]]
    print(similarity)
        
    scale_factor = torch.sum(torch.exp(Vo @ Vc[vocab_idx[Wc]]))
    print(scale_factor)

    return torch.exp(similarity) / scale_factor

def build_vocabulary_matrix(utterances:list, dimensions:int):
    vocab_idx = {}
    for utterance in utterances:
        for word in utterance.split(' '):
            if word not in vocab_idx:
                vocab_idx[word] = len(vocab_idx)
    
    return torch.randn(len(vocab_idx), dimensions), vocab_idx

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

class EmbeddingsModel(torch.nn.Module):

    def __init__(self, utterances:list, dimensions:int):
        super(EmbeddingsModel, self).__init__()
        self.utterances = utterances
        self.dimensions = dimensions
        self.Vo, self.vocab_idx = build_vocabulary_matrix(utterances, dimensions=3)
        self.Vc, self.vocab_idx = build_vocabulary_matrix(utterances, dimensions=3)
        print(self.Vo)
        print(self.Vc)

    def forward(self, outside:str, center:str):
        return calc_probability(Vo=self.Vo, Vc=self.Vc, Wo=outside, Wc=center, vocab_idx=self.vocab_idx)

if __name__ == '__main__':
    torch.manual_seed(123)

    print('test_calc_probability')
    test_calc_probability()

    print('test_EmbeddingsModel')
    model = EmbeddingsModel(utterances=["<start> he is sad", "<start> she is sad"], dimensions=3)
    prediction = model("he", "she")
    print(prediction)