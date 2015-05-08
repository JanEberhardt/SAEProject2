def f(x):
    if x<0:
        return 0
    else:
        return 1

def g(x):
    if x<10:
        return 0
    else:
        return 1

def h(x):
    return x+x

def main(y):
    return f(y) + g(h(y)+h(y))

