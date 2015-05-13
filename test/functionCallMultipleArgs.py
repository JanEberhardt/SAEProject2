#TODO: Something's wrong here!
def f(y):
    if y < -20:
        return -20 
    else:
        return y

def myfunction(a,b):
    if a < 0:
        if b < 0:
            return 0
        else:
            return 1
    else:
        return 2

def main(x,y):
    return myfunction(f(x), f(y))
    
