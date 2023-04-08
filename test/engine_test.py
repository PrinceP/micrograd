from micrograd.engine import Value
import torch

def test_add():
    a = Value(2.0)
    a = a + 1
    assert a.data == 3.0
    a = 1 + a
    assert a.data == 4.0
    
def test_subtract():
    a = Value(2.0)
    a = a - 1
    assert a.data == 1.0
    a = 1 - a
    assert a.data == 0.0    
    
def test_mul():
    a = Value(2.0)
    a = a * 1
    assert a.data == 2.0
    a = 1 * a
    assert a.data == 2.0
    
def test_divide():
    a = Value(2.0)
    a = a / 1
    assert a.data == 2.0
    a = 1 / a
    assert a.data == 0.5
    
def test_power():
    a = Value(2.0)
    a = a ** 2
    assert a.data == 4.0

def test_exp():
    a = Value(2.0)
    assert a.exp().data == 7.38905609893065

def test_forward_backward_pass():
    # taken from https://github.com/karpathy/micrograd#example-usage
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).tanh()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    print(f'{g.data}') # the outcome of this forward pass
    
    tol = 1e-6
    assert abs(g.data - 4.625192144640556) < tol
    g.backward()
    print(f'{a.grad}') # the numerical value of dg/da
    print(f'{b.grad}') # the numerical value of dg/db

    tol = 1e-6
    assert abs(a.grad - 27.060135135849574) < tol
    assert abs(b.grad - 117.33448345392071) < tol
    

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).tanh()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).tanh()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
    