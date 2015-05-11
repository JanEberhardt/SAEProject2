def f(y):
    if y < 0:
        return y+100 
    else:
        return y+y

def myfunction(a,b,unused):
    if a < b:
        return 0
    else:
        return 1

def main(x,y):
    return myfunction(f(x), f(y), 10)
    
