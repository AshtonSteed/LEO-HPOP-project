import butchertableau as bt

def main():
    x = bt.butcher(8, 8)
    a, b, c = x.radau()

    print(x)
    print(a)
    print(b)
    print(c)

if __name__ == '__main__':
    main()