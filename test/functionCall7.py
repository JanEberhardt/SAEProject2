def f(x):
    a = x+2
    b = 3
    if (a<b) and (b>x):
        return 1
    else:
        return 0

def main(y):
    return f(y)

def expected_result():
    return [1,0]
