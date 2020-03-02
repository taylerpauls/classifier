import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy import stats

df = pd.read_csv('/Users/I26259/wavenet-features/baji_feats_files_test_zp10.txt',names=['filename'])
df['filename'] = df['filename'].str.strip('[]')
df['filename'] = df['filename'].str.strip('[]').astype(str)
df['filename'] =  df['filename'].apply(lambda x: x.replace('[','').replace(']',''))

# if the filename is like :
# person_utterance_1
# ... named by block 1-- how many blocks from that file , need to reverse and remove the numbering from the blocks to get mode

#df_reverse = df.loc[:,'filename'].apply(lambda x: x[::-1])
#df_reverse = df_reverse.str[3:]
#df_reverse.str.strip()

df_predicts = pd.read_csv('/Users/I26259/wavenet-features/baji_feats_preds_zp10.txt',names=['preds'])
df_true = pd.read_csv('/Users/I26259/wavenet-features/baji_feats_labels_zp10.txt',names=['labels'])
preds = df_predicts.as_matrix()
labels = df_true.as_matrix()
con_m = confusion_matrix(labels,preds)
print("confusion matrix for raw (not mode) ")
print(con_m)

diags = np.diag(con_m)
sums_c = np.sum(con_m,axis=1)
print('accuracies by class -- not mode',diags/sums_c)
#a,b = df_true.labels.value_counts()
equals = np.equal(preds,labels).sum()
a = equals/len(labels)
print("accuracy for raw: ",a)
from sklearn.metrics import recall_score
print(recall_score(labels,preds,average='macro'))
new_p = []
new_l = []
#files_mat = df_reverse.as_matrix()
files_mat = df.as_matrix()
un = np.unique(files_mat)            # unique values in `array`
for i in range(0,len(un)):
    files_mat == un[i]                  # retrieve a boolean mask where elements are equal to 1
    id_1 =(files_mat == un[i]).nonzero()[0]
    pred = preds[id_1]
    lab = labels[id_1]
    m_p = stats.mode(pred)
    m_l = stats.mode(lab)
    mp = int(m_p[0])
    ml = int(m_l[0])
    new_p.append(mp)
    new_l.append(ml)


true = np.asarray(new_l)
p = np.asarray(new_p)
con_m = confusion_matrix(true,p)
#from sklearn.metrics import recall_score
print(recall_score(true,p,average='macro'))


print("mode confusion matrix")
print(con_m)
diags = np.diag(con_m)
sums_c = np.sum(con_m,axis=1)
print('accuracies by class by mode',diags/sums_c)

equals = np.equal(true,p).sum()
a = equals/len(p)
print(" total accuracy for mode: ",a)
