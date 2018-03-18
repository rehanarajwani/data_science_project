# -*- coding: utf-8 -*-
000000000000000000000000000000000000000000000000000000000000000000000000000000   
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
Created on Mon Mar 10 18:47:28 2018
Copyright © 2018 Kevin Feiyu Li
          @author: Kevin Feiyu Li
The author is under no obligation to use an open-source license that permit commercial use. The absence of an open-source license demonstrate that default copyright and ownership laws apply. The author retain all rights to this source code and that nobody else may reproduce, distribute, or create derivative works for commercial use from the author's work without permission from the author. 
"""
000000000000000000000000000000000000000000000000000000000000000000000000000000   
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

import csv
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#-----------------------------------------------------------------------------
def csv_read_column(file_obj,column_num):
    lovereader = csv.reader(file_obj)
    '''country=[]
    for row in spamreader:
         country.append(row[1])  #read the first element of each row'''
    data = [row[column_num] for row in lovereader] #same as above code
    return data
#-----------------------------------------------------------------------------
def csv_whole_read(file_obj):
    lovereader = csv.reader(file_obj)
    t=[]
    for row in lovereader:
        a=(",".join(row))
        t.append(a)
    return t
#-----------------------------------------------------------------------------
def csv_dict_reader(file_obj,col_nm1, col_nm2,col_nm3,col_nm4,col_nm5):
    reader = csv.DictReader(file_obj, delimiter=',')
    a=[]
    b=[]
    c=[]
    d=[]
    e=[]
    for line in reader:
        a.append(line[col_nm1])
        b.append(line[col_nm2])
        c.append(line[col_nm3])
        d.append(line[col_nm4])
        e.append(line[col_nm5])
    return a,b,c,d,e
#-----------------------------------------------------------------------------
def split_line(text):
    # split the text
    words = text.split(",")
    # for each word in the line:
    to_ls = []
    for word in words:
        to_ls.append(word)
    return to_ls
#-----------------------------------------------------------------------------
 #find not empty rows
def not_empty_idx(ls):
    not_empty_index=[]
    for i in range(len(ls)):
       if not ls[i]=='':
         not_empty_index.append(i)
    return not_empty_index
#-----------------------------------------------------------------------------
 #fill empty rows with 'Other'
def fill_empty(ls):
    fill_empty=[]
    for i in range(len(ls)):
       if ls[i]=='':
          ls[i]='Other'
    return ls
#-----------------------------------------------------------------------------
 #filter txt with useful rows
def filter_rows_has_txt(idx_ls,var_wait_for_filter):
    filter_=[]
    for i in range(len(idx_ls)):
          filter_.append(var_wait_for_filter[idx_ls[i]])
    return filter_
#-----------------------------------------------------------------------------
 #currency conversion   Convert all to USD    overwrite original data
def currecy_conver(input_sal, input_cur ,cur_country, ex_rt):
    fil_cur_us=[]
    fil_sal_us=[]
    for i in range(len(input_cur)):
      for j in range(len(cur_country)):
        if input_cur[i]==cur_country[j]:
           input_cur[i] ='USD'
           input_sal[i]=input_sal[i]*ex_rt[j]
    return input_sal,input_cur
#-----------------------------------------------------------------------------
  #verization function
def vec_d(text_input):
     vectorizer = TfidfVectorizer()
     X = vectorizer.fit_transform(text_input)
     arr_x = X.toarray()
     feature = vectorizer.get_feature_names() 
     return arr_x,feature
'''##############################################################################'''
'''##############################################################################'''
'''##############################################################################'''

if __name__ == "__main__":
  
#-----------------------------------------------------------------------------
    #read info by specify column name
    csv_path = 'multipleChoiceResponses.csv'
    name1="Country"
    name2="LearningDataScience"    #"Age"
    with open(csv_path, 'r',encoding='latin-1') as file_obj:
        a,b= csv_dict_reader(file_obj,name1,name2)
  
    #read info by specify column number
    csv_path = 'multipleChoiceResponses.csv'
    column_num =1
    with open(csv_path, 'r',encoding='latin-1') as file_obj:
         column_data = csv_read_column(file_obj,column_num)

    #read the whole dataset into list
    csv_path = 'multipleChoiceResponses.csv'
    with open(csv_path, 'r',encoding='latin-1') as file_obj:
         whole_data = csv_whole_read(file_obj)
    #then use states.split()

#-----------------------------------------------------------------------------    
    #count the number of questions we have in multiple choice response
    sub=','
    sg=whole_data[0]
    total_num_question=sg.count(sub)+1
#-----------------------------------------------------------------------------    
    #read the compensation and skills and maybe other features out
        #read info by specify column name
    csv_path = 'multipleChoiceResponses.csv'
    name1= "CompensationAmount"
    name2= "CompensationCurrency"    
    name3= "WorkToolsSelect"   #the tool they are using at work
    name4= "WorkCodeSharing" 
    name5= "WorkDatasetSize"
    with open(csv_path, 'r',encoding='latin-1') as file_obj:
      compen,compen_curr,code_tool,work_sharing_tool,work_datasize= csv_dict_reader(file_obj,name1,name2,name3,name4,name5)    
    
#-----------------------------------------------------------------------------    
    #delete the empty answer
    #count number of response to see which questions were answered the most; and then do analysis from there
    #whole data row 1 is the name of columns and row 2 is useless
    whole_data1 = whole_data[2:]
    compen_has_txt_idx=not_empty_idx(compen)
    #extract the empty rows   filtered_compen = [x for x in compen if not x=='']
    fil_compen=filter_rows_has_txt(compen_has_txt_idx,compen)
    fil_compen_curr=filter_rows_has_txt(compen_has_txt_idx,compen_curr)
    fil_code_tool=filter_rows_has_txt(compen_has_txt_idx,code_tool)
    fil_work_sharing_tool=filter_rows_has_txt(compen_has_txt_idx,work_sharing_tool)
    fil_work_datasize=filter_rows_has_txt(compen_has_txt_idx,work_datasize)
    #extract the text idx of fil_compen_curr
    fil_compen_curr_has_txt_idx=not_empty_idx(fil_compen_curr)
    
    #extract the final useful info
    fil_compen_curr1=filter_rows_has_txt(fil_compen_curr_has_txt_idx,fil_compen_curr)
    fil_compen1=filter_rows_has_txt(fil_compen_curr_has_txt_idx,fil_compen)
    fil_code_tool1=filter_rows_has_txt(fil_compen_curr_has_txt_idx,fil_code_tool)
    fil_work_sharing_tool1=filter_rows_has_txt(fil_compen_curr_has_txt_idx,fil_work_sharing_tool)
    fil_work_datasize1=filter_rows_has_txt(fil_compen_curr_has_txt_idx,fil_work_datasize)

#-----------------------------------------------------------------------------    
  #Useful varilable for later use
    #fill few empty rows with "Other"
    fil_tool=fill_empty(fil_code_tool1)
    fil_work_shr_tool=fill_empty(fil_work_sharing_tool1)
    fil_work_dtsz=fill_empty(fil_work_datasize1)
    #rename for easy use
    fil_sal=fil_compen1
    #standarize the fil_sal data
    for x in range(len(fil_sal)):
        if fil_sal[x]=='-':
           fil_sal[x]='0'
        if fil_sal[x]=='':
           fil_sal[x]='0'
    fil_cur= fil_compen_curr1
    fil_sal[216]='100000000000'
#-----------------------------------------------------------------------------  
  #currency conversion
    csv_path = 'conversionRates.csv'
    name1= "originCountry"
    name2= "exchangeRate"    
    name3= "originCountry"   #the tool they are using at work
    name4= "originCountry" 
    name5= "originCountry"
    with open(csv_path, 'r',encoding='latin-1') as file_obj:
      cur_country,ex_rt,no_use,no_use1,no_use2= csv_dict_reader(file_obj,name1,name2,name3,name4,name5)    
   
    #convert str to float points
    for i in range(len(ex_rt)):
            ex_rt[i]=float(ex_rt[i])
    for j in range(len(fil_sal)):
            fil_sal[j]=float(re.sub("[^\d\.]", '', fil_sal[j]))
 
   #convert all to USD
   fil_sal_us,fil_cur_us=currecy_conver(fil_sal, fil_cur ,cur_country, ex_rt)
   
#-----------------------------------------------------------------------------  
  #label the fil_sal_us  
  lb_sal=[2]*len(fil_sal_us) #creat a random list
  for i in range(len(fil_sal_us)):
       if fil_sal_us[i]<50000:
          lb_sal[i]=1
       if 50000<fil_sal_us[i]<75000:
          lb_sal[i]=2
       if 75000<fil_sal_us[i]<100000:
          lb_sal[i]=3
       if fil_sal_us[i]>100000:
          lb_sal[i]=4
  plt.hist(lb_sal)

#-----------------------------------------------------------------------------  
  #vectorization and extract the feature names
     arr_tool, fea_tool = vec_d(fil_tool)  
     arr_shrtool, fea_shrtool = vec_d(fil_work_shr_tool) 
     arr_dtsz, fea_dtsz = vec_d(fil_work_dtsz) 
 
     arr_whole = np.concatenate((arr_tool, arr_shrtool), axis=1)
     arr_whole = np.concatenate((arr_whole, arr_dtsz), axis=1)
     
     arr_lb_sal = np.asarray(lb_sal)
    

    
     from sklearn.model_selection import train_test_split
     from sklearn.linear_model import LogisticRegression
     X_train, X_test, y_train, y_test = train_test_split(arr_whole, arr_lb_sal, test_size=0.30, random_state=42)
     lr = LogisticRegression(penalty='l2',dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None)
     lr.fit(X_train,y_train) 
     print('The Train Accuracy {0:.3f}.'.format((lr.predict(X_train)== y_train).mean()))
     print('The Test Accuracy {0:.3f}.'.format((lr.predict(X_test)== y_test).mean()))
     
     from sklearn.neighbors import KNeighborsClassifier
     i=3 #give the best result  For whole data. The Train Accuracy 0.596. The Test Accuracy 0.422.
         
     knn = KNeighborsClassifier(n_neighbors=i)
     knn.fit(X_train,y_train) 
     print('i=', i)
     print('The Train Accuracy {0:.3f}.'.format((knn.predict(X_train)== y_train).mean()))
     print('The Test Accuracy {0:.3f}.'.format((knn.predict(X_test)== y_test).mean()))
     
     
     https://machinelearningmastery.com/feature-selection-machine-learning-python/
     read section: look for Feature Importance
     https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/
     read section: View Feature Importance
     
     
     
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

clf = RandomForestClassifier(n_estimators=20,max_depth=9, min_samples_split=2,bootstrap=True)
clf = clf.fit(X_train, y_train)
print('The Train Accuracy {0:.3f}.'.format((clf.predict(X_train)== y_train).mean()))
print('The Test Accuracy {0:.3f}.'.format((clf.predict(X_test)== y_test).mean()))


clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores = cross_val_score(clf, arr_whole, arr_lb_sal) 
scores.mean()                             


clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, arr_whole, arr_lb_sal)
scores.mean()                             


clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, arr_whole, arr_lb_sal)
scores.mean() 


clf = ExtraTreesClassifier(n_estimators=15, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, arr_whole, arr_lb_sal)
scores.mean() 
   

000000000000000000000000000000000000000000000000000000000000000000000000000000   
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
   useful dataset: fil_sal_us,fil_tool,fil_work_shr_tool,fil_work_dtsz
    1. convert the currecy  --done
    2. convert salary range to 1 2 3 4 5 --done
    3. vectorization --done
    3. convert text of fil_tool to value (by converter)  --done
    3. predict   --done  only around 59%
    3. put more features and do importance rating
    3. testing prediction is too low // doesnt make sense
    4. plot datasize against the other things
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
000000000000000000000000000000000000000000000000000000000000000000000000000000

    
    
    '''t=['0', '1', '', '0', '0', '8.4583', '0']       
    for i in range(len(t)):
         if  t[i]=='':
             t[i]=0
         else:
             t[i]=1  '''      
  
#-----------------------------------------------------------------------------    
    #run all data into a dict 
    ls_question = split_line(whole_data[0])
#-----------------------------------------------------------------------------       
 list_of_answers=[]  #ls_queation is the dynamic key names
 for line in whole_data1:
     dic={}
    #print(line)
     x=line.split(",",228)  #maximum allawance of split to the 229th location
     for i in range(228):
         dic[ls_question[i]]=x[i]
     list_of_answers.append(dic)
#-----------------------------------------------------------------------------    
#-----------------------------------------------------------------------------    
#use this demo code to strip out the list_of_answers
    usr = []  #not gonna use this in training; but can do analysis; word count
for line in list_of_dic:
    #print((line['class']))
    #replace " by space
    usr.append(line['user'].replace('"', ' ').strip())
    
lb_list=[]
for line in list_of_dic:
    #print((line['class']))
    #replace " by space
    lb_list.append(int(line['label'].replace('"', ' ').strip()))
#-----------------------------------------------------------------------------    

         
pd.read_csv('multipleChoiceResponses.csv', sep='|', names=None , encoding='latin-1')



         
         
