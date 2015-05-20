def might_fail ( x ):
    assert ( x >= 0) # Assert #1
    return True

def main ( x ):
    assert ( might_fail ( x *2)) # Assert #2
    return 0

def expected_result():
    return [0]
