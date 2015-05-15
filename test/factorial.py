def fact(x):
    if x<1:
        return 1
    else:
        return x*fact(x-1)

def main(y):
    return fact(3)

def expected_result():
    return [6]
