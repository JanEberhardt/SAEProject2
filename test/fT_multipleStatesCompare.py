def f(a):
    if a<0:
        return 200
    else:
        return 100
def g(b):
    if b<0:
        return 200
    else:
        return 100

def main(x, y):
    if f(x) < g(y):
        return 1
    else:
        return 0

def expected_result():
    return [1, 0]
