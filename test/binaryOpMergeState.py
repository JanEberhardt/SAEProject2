def f(a):
    b = 2
    if b != 2:
        return 100
    if a>0:
        return 1
    else:
        return 0
    
def main(x,y):
    return f(x)+f(y)
