with open(r'C:\Users\white\OneDrive\School\CSCI-447 Machine Learning\Project2\CSCI447_P2\6D 1HL Learning Curve.txt',
          'r') as infile, open(
    r'C:\Users\white\OneDrive\School\CSCI-447 Machine Learning\Project2\CSCI447_P2\Results\6D 1HL Learning Curve.txt',
    'w') as outfile:
    data = infile.read()
    data = data.replace("[", "")
    data = data.replace("]", "")
    outfile.write(data)
