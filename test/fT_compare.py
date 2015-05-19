def main(x, y):
    a = 0
    if x == y:
        a=a+1
    if x != y:
        a=a+10
    if x < y:
        a=a+100
    if x<=y:
        a=a+1000
    if x > y:
        a=a+10000
    if x >= y:
        a=a+100000

    return a

def expected_result():
    return [101001, 110010, 1110]
