def f(y):
    if y < -20:
        return -20 
    else:
        return y

def myfunction(a,b, c):
    if a < 0:
        if b < 0:
            return 0
        else:
            return 1
    else:
        return 2

def main(x,y):
    z = 2
    return myfunction(f(x), f(y), f(z))
    
def expected_result():
    return [0, 1, 2]
