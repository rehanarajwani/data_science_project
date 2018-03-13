# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:47:28 2018

@author: kevin
"""

import csv
import pandas as pd

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
def csv_dict_reader(file_obj,column_name1, column_name2):
    reader = csv.DictReader(file_obj, delimiter=',')
    a=[]
    b=[]
    for line in reader:
        a.append(line[column_name1])
        b.append(line[column_name2])
    return a,b

   
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
    #run all data into a dict 
    
     lis=['l', 'u', 't']  #dynamic key names
 for line in classified1:
    dic={}
    #print(line)
    x=line.split(",",5)  #maximum allawance of split to the fifth location
    a=x[0]
    b=x[4]
    c=x[5]
    dic[lis[0]]=a
    dic[lis[1]]=b
    dic[lis[2]]=c
    list_of_dic1.append(dic)
    

    dic[label]=a
    dic[user]=b
    dic[text]=c
    list_of_dic.append(dic)
    
    
        for k in lis:
             dic[k]=
    
#-----------------------------------------------------------------------------    
    #count number of response to see which questions were answered the most; and then do analysis from there 
    
    t=['0', '1', '', '0', '0', '8.4583', '0']       
    for i in range(len(t)):
         if  t[i]=='':
             t[i]=0
         else:
             t[i]=1        
  
         
pd.read_csv('multipleChoiceResponses.csv', sep='|', names=None , encoding='latin-1')



         
         
