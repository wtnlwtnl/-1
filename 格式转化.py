import numpy as np

src = 'semeion.data.txt'          # 原始数据：256像素 + 10列one-hot
dst = 'semeion_digits.arff'       # 目标ARFF文件

raw = np.loadtxt(src)
X = raw[:, :256].astype(int)          # 像素（0/1）
y = np.argmax(raw[:, 256:], axis=1)   # 把one-hot转回0..9

with open(dst, 'w') as f:
    f.write("% Auto-generated ARFF from Semeion one-hot format\n")
    f.write("@relation semeion_digits\n\n")
    for i in range(1, 257):
        f.write(f"@attribute pixel{i} numeric\n")
    f.write("@attribute class {0,1,2,3,4,5,6,7,8,9}\n\n")
    f.write("@data\n")
    for i in range(X.shape[0]):
        row = ",".join(map(str, X[i].tolist() + [int(y[i])]))
        f.write(row + "\n")

print("Saved:", dst)
