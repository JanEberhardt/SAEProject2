def main(n):
    a = 7
    b = 12
    a, b = b, a+b
    return b

def expected_result():
    return [19]
