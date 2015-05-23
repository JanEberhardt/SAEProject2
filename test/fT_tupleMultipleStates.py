def f(a):
    if a<0:
        return 1
    elif a<2:
        return 2
    else:
        return 3

def main(x,y,z):
    a = (f(x), f(y), f(z))
    b = (1,2,3)
    if a == b:
        return 200 
    else:
        return 0

def expected_result():
    return [0,200]
