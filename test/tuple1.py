def f(a):
    if a<0:
        return 1
    else:
        return 0

def main(x, y):
    a = (f(x), f(y))
    return 1

def expected_result():
    return [1]
