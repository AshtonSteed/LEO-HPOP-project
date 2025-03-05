import butchertableau as bt

def main():
    x = bt.butcher(8, 0)
    a, b, c = x.radau()

    print(a, b, c)