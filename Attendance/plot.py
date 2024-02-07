from matplotlib import pyplot as plt


data = [3, 7, 4, 11, 5, 6]
lab = ["math", "eng", "sci", "you", "hau", "igb"]
plt.bar(lab, data)
plt.ylim(0, 11)
plt.show()
