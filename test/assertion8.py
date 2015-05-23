def odd ( x ):
    if ( x % 2 == 0):
        return True
    else :
        return False

def main ( x ):
    assert ( odd ( x *2))
    return 0

def expected_result():
    return [0]
