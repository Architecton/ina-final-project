import os


def main():
    path = os.getcwd()[:len(os.getcwd()) - 3] + "data\\deep_learning\\djava"
    out = os.getcwd()[:len(os.getcwd()) - 3] + "data\\deep_learning\\djava_transformed"
    f = open(path, "r")
    g = open(out, "w")
    for line in f:
        l = line.split()
        s = "".join(l[0] + "," + l[1] + ",0")
        g.write(s + "\n")
        print(s)
    f.close()
    g.close()


if __name__ == '__main__':
    main()