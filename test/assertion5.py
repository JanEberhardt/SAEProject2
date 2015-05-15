def main(x, y):
    assert(test(x,y) == test(x, 0))
    return 1

def test(x, y):
    return x+y

def expected_result():
    return [1]
