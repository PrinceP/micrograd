from micrograd.multi_layer_perceptron import MLP

def test_exp():    
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4,4,1])
    n(x)
    print(len(n.parameters()))
    assert len(n.parameters()) == 41