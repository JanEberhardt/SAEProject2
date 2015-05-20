def main (x):
    assert ( False ) # Assert #1
    if x > 0:
        assert ( x > 1) # Assert #2
    return 0

def expected_result():
    return [0]
