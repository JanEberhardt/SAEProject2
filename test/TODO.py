def f(a,b):
    if a==b:
        return 100
    else:
        return 200

def g(x, y):
    if x != y:
        return 10
    else:
        return 20

def main(x, y):
    a = f(x, y)
    b = g(x, y)
    if x <= y:
        return (1+f(a,b))+g(a,b)
    elif x >= y:
        return (2+f(a,b))+g(a,b)
    elif x < y:
        return (3+f(a,b))+g(a,b)
    elif x > y:
        return (4+f(a,b))+g(a,b)
    else:
        return 0

def expected_result():
    return [211, 212]
