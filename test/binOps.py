def main(z):
    x=2
    y=4
    if z==1:
        return x+y
    elif z==2:
        return x-y
    elif z==3:
        r= x*y
    elif z==4:
        r= y/x
    elif z==5:
        r= x%y
    elif z==6:
        r= x**y
    elif z==7:
        r = 16 << 1
    elif z==8:
        r = 128 >> 1
    else:
        r= 0

    if r == 0:
        r = (x+y) * (x-y) * (x*x) - (x*x)
    return r

def expected_result():
    return [6, -2, 8, 2, 2, 16, -52, 32, 64]
