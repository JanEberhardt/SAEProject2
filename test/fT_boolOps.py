def main(x, y, z):
    if (x < y and y < z) or (x < z):
        return 1
    else:
        return 0

def expected_result():
    return [0,1]
