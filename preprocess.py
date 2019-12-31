import pandas as pd
import numpy as np
from scipy import stats

#đầu tiên là đọc file input vào
data= pd.read_csv("student_train.csv")
data.head()

#dùng zscore để loại bỏ những giá trị nằm ngoài vùng

#tính z score của từng giá trị trong data theo từng cột
z = np.abs(stats.zscore(data))

#loại bỏ các outlier theo từng cột ( nghĩa là giá trị lớn hơn 3)
data = data[(z < 3).all(axis=1)]

#tính ra trung vị ở phía dưới và trung vị của phía trên của từng attribute
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

#lấy ra những giá trị nằm trong khoảng giữa và chênh lệnh ko quá nhiều so với trung vị để giải quyết các outlier
data1 = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

data.to_csv("student_trainzscore.csv",index=False)
data1.to_csv("student_quantileafterzscore.csv",index=False)


