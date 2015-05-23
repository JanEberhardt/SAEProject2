def my_func(aa,bb):
    if (aa>2):
        return 200
    elif (aa==1):
        return 6
    else:
        return 1
def foo(x):
    if (x>10):
        return 1
    else:
        return 2

def main ( x ):
    bbbb = foo(foo(my_func(x,foo(x))))
    y = my_func(x,20)
    yy = my_func(x,20)
    if (x>15):
        if (y>300):
            return 1
        else:
            return 2
    else:
        return 2211122
    return 2

def expected_result():
    return [2211122, 2]
