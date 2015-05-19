def g(a):
    return 10

def f(x):
    if x<0:
        return g(x)
    else:
        return 0

def main(y):
    a = f(y)
    if a==0:
        return 0
    else:
        return 1

def expected_result():
    return [0,1]
