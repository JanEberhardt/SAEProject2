def main(x, y):
    z = x * y
    if z > 20:
        z = 20

    if x > 1:
        return 1

    return x*z

def expected_result():
    return [1,-600, 1, 0]
