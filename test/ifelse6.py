def main(x):
    if x < 0:
        return 0
    y = twice(x)
    return y

def twice(x):
    return 2*x + 1

def expected_result():
    return [0,1]
