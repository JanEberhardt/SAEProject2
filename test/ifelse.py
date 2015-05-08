def main(x):
    if x>1:
        if x>2:
            if x>3:
                if x>4:
                    return 4
                else:
                    y= 3
            else:
                y=2
        else:
            y=1
    else:
        y=0
    return y

def expected_result():
    return [0, 1, 2, 3, 4]
