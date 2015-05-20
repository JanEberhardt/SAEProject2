def foo ( x ):
    if x > 0:
        return 1
    else :
        return 0

def main ( x ):
    y = foo ( x )
    return 0

def expected_result():
    return [0]
