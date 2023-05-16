# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from flask_mail import Mail, Message
from flask import send_file
import mysql.connector
import hashlib
import datetime
from datetime import datetime
from datetime import date
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser
#from plotly import graph_objects as go

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from werkzeug.utils import secure_filename

import urllib.request
import urllib.parse
import socket    
import csv

import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from collections import OrderedDict

import re    # for regular expressions 
#import nltk  # for text manipulation 
import string # for text manipulation 
import warnings
#from nltk.stem.porter import *
#from nltk.corpus import stopwords
#nltk.download()
#from nltk.stem import PorterStemmer
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords
###

import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from gensim.parsing.porter import PorterStemmer
#from gensim.summarization.textcleaner import tokenize_by_word

from spacy.lang.es import Spanish
nlp = Spanish()

import email.policy
from bs4 import BeautifulSoup
#import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn
###

from imblearn.over_sampling import SMOTEN
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from wordcloud import WordCloud

plt.rc("axes.spines", right=False, top=False)
plt.rc("font", family="serif")
###
'''from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
tqdm.pandas(desc="progress-bar")
#from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

from sklearn.linear_model import LogisticRegression
#from gensim.models.doc2vec import TaggedDocument
import re'''
###


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="email_spam"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
##email
mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": "rnd1024.64@gmail.com",
    "MAIL_PASSWORD": "kazxlklvfrvgncse"
}

app.config.update(mail_settings)
mail = Mail(app)
#######

def sendmail(usermail,mess1):

    subj1="Spam-Spoiler"
    with app.app_context():
        msg = Message(subject=subj1, sender=app.config.get("MAIL_USERNAME"),recipients=[usermail], body=mess1)
        mail.send(msg)

@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')

    
@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    msg=""

    act=request.args.get("act")
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            ff=open("user.txt","w")
            ff.write(uname)
            ff.close()
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'

    
    return render_template('login_user.html',msg=msg,act=act)


@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""
    act=request.args.get("act")
    #usermail=""
    #mess1="mytest"
    #sendmail(usermail,mess1)
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('train_data'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg,act=act)

@app.route('/register', methods=['GET', 'POST'])
def register():
   
    msg=""
    act=request.args.get("act")
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        #email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        #password=request.form['password']
       
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM register where uname=%s",(uname,))
        cnt = mycursor.fetchone()[0]

        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                    
            sql = "INSERT INTO register(id,name,mobile,uname,pass) VALUES (%s, %s, %s, %s, %s)"
            val = (maxid,name,mobile,uname,pass1)
            mycursor.execute(sql, val)
            mydb.commit()            
            #print(mycursor.rowcount, "Registered Success")
            msg="success"
            #if mycursor.rowcount==1:
            return redirect(url_for('register',act='1'))
        else:
            msg='Already Exist!'
    return render_template('register.html',msg=msg,act=act)

@app.route('/setting', methods=['GET', 'POST'])
def setting():
   
    msg=""
    uname=""
    act=request.args.get("act")
    
    if 'username' in session:
        uname = session['username']
    ff=open("user.txt","r")
    uname=ff.read()
    ff.close()

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    det = mycursor.fetchone()
    em=det[3]
    pw=det[6]
        
    if request.method=='POST':
        
        email=request.form['email']
        password=request.form['pass']

        mycursor.execute("update register set email=%s,password=%s where uname=%s",(email,password,uname))
        mydb.commit()
       
        return redirect(url_for('setting',act='1'))
        
    return render_template('setting.html',msg=msg,em=em,pw=pw,act=act,det=det)


def emailsink(usermail,pwd,uname):
    import email
    import imaplib
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    (retcode, capabilities) = mail.login(usermail,pwd)
    mail.list()
    mail.select('inbox')

    subj1="Spam-Spoiler"
    n=0
    (retcode, messages) = mail.search(None, '(UNSEEN)')
    if retcode == 'OK':

       for num in messages[0].split() :
          print ('Processing ')
          n=n+1
          typ, data = mail.fetch(num,'(RFC822)')
          for response_part in data:
             if isinstance(response_part, tuple):
                 original = email.message_from_bytes(response_part[1])

                # print (original['From'])
                # print (original['Subject'])
                 raw_email = data[0][1]
                 raw_email_string = raw_email.decode('utf-8')
                 email_message = email.message_from_string(raw_email_string)
                 for part in email_message.walk():
                            if (part.get_content_type() == "text/plain"): # ignore attachments/html
                                  body = part.get_payload(decode=True)
                                  save_string = str(r"data.txt" )
                                  myfile = open(save_string, 'a')
                                  myfile.write(original['From']+'\n')
                                  myfile.write(original['Subject']+'\n')            
                                  myfile.write(body.decode('utf-8'))
                                  subj=original['Subject']
                                  sender=original['From']
                                  mess=body.decode('utf-8')
                                  

                                  
                                  myfile.write('**********\n')
                                  myfile.close()

                                  if subj1==subj:
                                      print("subject")

                                  else:

                                      ###
                                      x=0
                                      spam_st=""
                                      f1=open("spammail.txt","r")
                                      dat=f1.read()
                                      f1.close()
                                      dat1=dat.split("|")
                                      for rd in dat1:
                                          rd1=rd.split('##')
                                          spam_st=rd1[1]
                                          t1=mess
                                          t2=rd1[0] #rd.strip()
                                          if t2 in t1:
                                              x+=1
                                              print("yes")
                                              break
                                          else:
                                              print("no")
                                      mail_det=""

                                      if x>0:
                                          
                                          print(spam_st)
                                          if spam_st=="1":
                                              mail_det="Fraudulent"
                                          elif spam_st=="2":
                                              mail_det="Harrasment"
                                          elif spam_st=="3":
                                              mail_det="Suspicious"
                                              
                                          mycursor = mydb.cursor()
                                          mycursor.execute("SELECT max(id)+1 FROM read_data")
                                          maxid = mycursor.fetchone()[0]
                                          if maxid is None:
                                              maxid=1
                                                
                                          sql = "INSERT INTO read_data(id,subject,sender,uname,message,spam_st) VALUES (%s, %s, %s, %s, %s, %s)"
                                          val = (maxid,subj,sender,uname,mess,mail_det)
                                          mycursor.execute(sql, val)
                                          mydb.commit()

                                          ##
                                          #Reply mail
                                          
                                          mess1=mail_det+" mail has deleted *** "+mess+" *** "
                                          sendmail(usermail,mess1)

                                          #Delete mail
                                          mail.store(num,'+FLAGS',r'(\Deleted)')
                                          
                                      ###
                            else:
                                  continue

                 typ, data = mail.store(num,'+FLAGS','\\Seen')

    #print (n)
    return n
                 
    
@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
   
    msg=""
    uname=""
    
    
    if 'username' in session:
        uname = session['username']
    ff=open("user.txt","r")
    uname=ff.read()
    ff.close()

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    det = mycursor.fetchone()
    em=det[3]
    email=det[3]
    pwd=det[6]

        
    return render_template('userhome.html',msg=msg,det=det)             

@app.route('/spam_detect', methods=['GET', 'POST'])
def spam_detect():
   
    msg=""
    uname=""
    
    
    if 'username' in session:
        uname = session['username']
    ff=open("user.txt","r")
    uname=ff.read()
    ff.close()

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    det = mycursor.fetchone()
    em=det[3]
    email=det[3]
    pwd=det[6]
    print(email)
    print(pwd)

    now = datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    rtime=now.strftime("%H-M")
    dtt=rdate+" "+rtime
    #########

    res=emailsink(email,pwd,uname)
    unread=res
    
    ########
    mycursor.execute("SELECT * FROM read_data where uname=%s order by id desc",(uname,))
    data = mycursor.fetchall()
    
        
    return render_template('spam_detect.html',msg=msg,em=em,data=data,unread=unread)


@app.route('/train_data', methods=['GET', 'POST'])
def train_data():
    msg=""
    act = request.args.get('act')

    pd.set_option("display.max_colwidth", 200) 
    warnings.filterwarnings("ignore") #ignore warnings

    #dataset/SEFACED_Email_Forensic_Dataset1.csv
    data = pd.read_csv(
        "static/dataset/train.csv",
        header=0,
        encoding="latin-1",
        usecols=[0, 1],
        names=["label", "text"],
    )
    #dat1 = pd.read_csv("static/dataset/SEFACED_Email_Forensic_Dataset1.csv", header=0)
    #dat=dat1.head()
    data1=[]
    i=0
    for ds in data.values:
        #if i<=200:
        data1.append(ds)
        #i+=1
    '''plt.rc("axes.spines", right=False, top=False)
    plt.rc("font", family="serif")
            
    data = pd.read_csv(
        "static/dataset/data1.csv",
        header=0,
        encoding="latin-1",
        usecols=[0, 1],
        names=["label", "text"],
    )
    dat=data.head()
    data1=[]
    for ds in dat.values:
        data1.append(ds)'''

    '''for label, cmap in zip(["ham", "spam"], ["winter", "autumn"]):
        text = data.query("label == @label")["text"].str.cat(sep=" ")
        plt.figure(figsize=(10, 6))
        #wc = WordCloud(width=1000, height=600, background_color="#f8f8f8", colormap=cmap)
        #wc.generate_from_text(text)
        #plt.imshow(wc)
        #plt.axis("off")
        #plt.title(f"Words Commonly Used in ${label}$ Messages", size=20)
        #plt.show()'''

    '''data["length (words)"] = data["text"].str.split().apply(len)
    print(data.groupby("label").agg([min, max, "mean"]))
    ax = data.boxplot(by="label", figsize=(6, 4.5))
    _ = ax.set_title("")
    #plt.show()
    #plt.close()
    ######

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], random_state=8, stratify=data["label"]
    )

    _ = y_train.value_counts().plot.bar(
        color=["aqua", "orangered"], edgecolor="#555", alpha=0.5
    )
    #plt.show()
    #plt.close()'''
    #############
    


   

    
    return render_template('train_data.html',msg=msg,data1=data1)


@app.route('/process1', methods=['GET', 'POST'])
def process1():
    pd.set_option("display.max_colwidth", 200) 
    warnings.filterwarnings("ignore") #ignore warnings

    #dff = pd.read_csv("static/dataset/SEFACED_Email_Forensic_Dataset1.csv",encoding='latin-1')
    dff = pd.read_csv("static/dataset/train.csv",encoding='latin-1')

    class_weight = 1 / dff["Text"].value_counts()
    class_weight = dict(class_weight / class_weight.sum())
    

    ####
    df = pd.read_csv("static/dataset/data1.csv",encoding='latin-1')
    df.head()

    #stop_words = stopwords.words('english')
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",',','.','I','\'','-','/']
    ##Tokenize
    data1=[]
    i=0
    for ds in df.values:
        dt=[]
        if i<5:
            
            dt.append(ds[1])
            text=ds[1]

            doc = nlp(text)
            text_tokens = [token.text for token in doc]
            #text_tokens=tokenize_by_word(text)
            #text_tokens =word_tokenize(text)
            tokens_without_sw = [word for word in text_tokens if not word in stop_words]
            dt.append(tokens_without_sw)

            
            data1.append(dt)
        i+=1

    #Stemming
    ps = PorterStemmer()
     
    # choose some words to be stemmed
    #words = ["program", "programs", "programmer", "programming", "programmers"]
     
    
    data2=[]
    i=0
    for ds2 in df.values:
        dt2=[]
        if i<5:
            
            
            text2=ds2[1]
            dt2.append(ds2[1])

            doc = nlp(text)
            text_tokens = [token.text for token in doc]
            #text_tokens =word_tokenize(text2)
            tokens_without_sw = [word for word in text_tokens if not word in stop_words]
            
            swrd=[]
            for w in tokens_without_sw:
                sw=ps.stem(w)
                swrd.append(sw)
            dt2.append(swrd)

            
            data2.append(dt2)
        i+=1
    ##Stop words

    stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    #data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
    data3=[]
    i=0
    for ds3 in df.values:
        dt3=[]
        if i<5:
            
            
            content=ds3[1]
            
            
            dt3.append(ds3[1])
            content = content.lower()
            swrd=[]
            # Remove stop words
            for stopword in stopwords:
                content = content.replace(stopword + " ", "")
                content = content.replace(" " + stopword, "")
                swrd.append(content)
            data3.append(swrd)
        i+=1
    

        
    #DATASET_COLUMNS = ["Text","Class_Label"]
    #data.columns = DATASET_COLUMNS
    #data.head()
   

    #stop_words = stopwords.words('english')
    #df['clean_Text'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    #dat=df['clean_Text']
    #print(dat)

    

    
    return render_template('process1.html',data1=data1,data2=data2,data3=data3,class_weight=class_weight)

@app.route('/process2', methods=['GET', 'POST'])
def process2():

    ##########
    plt.rc("axes.spines", right=False, top=False)
    plt.rc("font", family="serif")
    data = pd.read_csv(
        "static/dataset/data1.csv",
        header=0,
        encoding="latin-1",
        usecols=[0, 1],
        names=["label", "text"],
    )
    dat=data.head()

    

    data["length (words)"] = data["text"].str.split().apply(len)
    dataval=data.groupby("label").agg([min, max, "mean"])
    print(data.groupby("label").agg([min, max, "mean"]))
    ax = data.boxplot(by="label", figsize=(6, 4.5))
    _ = ax.set_title("")
    #plt.show()
    plt.savefig("static/dataset/graph2.png")
    plt.close()
    ######

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], random_state=8, stratify=data["label"]
    )

    _ = y_train.value_counts().plot.bar(
        color=["aqua", "orangered"], edgecolor="#555", alpha=0.5
    )
    #plt.show()
    plt.savefig("static/dataset/graph3.png")
    plt.close()
    ############################################################################

    data = pd.read_csv(
        "static/dataset/data2.csv",
        header=0,
        encoding="latin-1",
        usecols=[0, 1],
        names=["label", "text"],
    )
    dat=data.head()

    

    data["length (words)"] = data["text"].str.split().apply(len)
    dataval2=data.groupby("label").agg([min, max, "mean"])
    print(data.groupby("label").agg([min, max, "mean"]))
    ax = data.boxplot(by="label", figsize=(6, 4.5))
    _ = ax.set_title("")
    #plt.show()
    plt.savefig("static/dataset/graph4.png")
    plt.close()
    ######

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], random_state=8, stratify=data["label"]
    )

    _ = y_train.value_counts().plot.bar(
        color=["aqua", "orangered"], edgecolor="#555", alpha=0.5
    )
    #plt.show()
    plt.savefig("static/dataset/graph5.png")
    plt.close()
    #############################################################################
    data = pd.read_csv(
        "static/dataset/data3.csv",
        header=0,
        encoding="latin-1",
        usecols=[0, 1],
        names=["label", "text"],
    )
    dat=data.head()

    

    data["length (words)"] = data["text"].str.split().apply(len)
    dataval3=data.groupby("label").agg([min, max, "mean"])
    print(data.groupby("label").agg([min, max, "mean"]))
    ax = data.boxplot(by="label", figsize=(6, 4.5))
    _ = ax.set_title("")
    #plt.show()
    plt.savefig("static/dataset/graph6.png")
    plt.close()
    ######

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], random_state=8, stratify=data["label"]
    )

    _ = y_train.value_counts().plot.bar(
        color=["aqua", "orangered"], edgecolor="#555", alpha=0.5
    )
    #plt.show()
    plt.savefig("static/dataset/graph7.png")
    plt.close()
    #################################################################
    
    return render_template('process2.html',dataval=dataval,dataval2=dataval2,dataval3=dataval3)

@app.route('/process3', methods=['GET', 'POST'])
def process3():

    df = pd.read_csv('static/dataset/spam11.csv')

    print(df.shape)
    dat=df.head()
    data1=[]
    for ds1 in dat.values:
        data1.append(ds1)
    
    dat2=df.describe()
    data2=[]
    drr=['count','mean','std','min','25%','50%','75%','max']
    i=0
    for ds2 in dat2.values:
        dt=[]
        dt.append(drr[i])
        dt.append(ds2)
        i+=1
        data2.append(dt)
    
    #df.info()
    dat3=df.corr()
    data3=[]
    for ds3 in dat3.values:
        data3.append(ds3)


    #visualize correlation of variable using pearson correlation
    plt.figure(figsize = (8,6))
    sns.heatmap(df.corr(), vmax = 0.9, cmap = 'YlGnBu')
    plt.title('Pearson Correlation', fontsize = 15, pad = 12, color = 'r')
    plt.savefig("static/dataset/ff_g1.png")
    #plt.show()

    #transform spam column to categorical data
    df.spam[df['spam'] == 0] = 'ham'
    df.spam[df['spam'] == 1] = 'spam'
    dat4=df.head()
    data4=[]
    for ds4 in dat4.values:
        data4.append(ds4)
    
    #analyze of spam status based on capital run length average
    dat5=pd.pivot_table(df, index = 'spam', values = 'capital_run_length_average', 
                   aggfunc = {'capital_run_length_average' : np.mean}).sort_values('capital_run_length_average', ascending = False)

    print(dat5)

    #analyze of spam status based on count of capital run length longest
    pd.pivot_table(df, index = 'spam', values = 'capital_run_length_longest',
                  aggfunc = {'capital_run_length_longest' : np.sum}).sort_values('capital_run_length_longest', ascending = False)


    #anayze of spam status based on count of capital run length total
    pd.pivot_table(df, index = 'spam', values = 'capital_run_length_total',
                  aggfunc = {'capital_run_length_total' : np.sum}).sort_values('capital_run_length_total', ascending = False)


    #anayze of spam status based on capital run length average, capital run length longest and capital run length total
    pd.pivot_table(df, index = 'spam', values = ['capital_run_length_average', 'capital_run_length_longest', 
                                                 'capital_run_length_total'], 
                   aggfunc = {'capital_run_length_average' : np.mean, 'capital_run_length_longest' : np.sum, 
                              'capital_run_length_total' : np.sum}).sort_values(['capital_run_length_average', 
                                                                                 'capital_run_length_longest', 
                                                                                 'capital_run_length_total'], ascending = False)


    #visualize the factor of spam message based on capital run length average, capital run length longest and capital run length total
    plt.figure(figsize = (14,6))
    chart = df.boxplot()
    chart.set_xticklabels(chart.get_xticklabels(), rotation = 90)
    plt.title('The Factor of Spam Message', fontsize = 15, pad = 12, color = 'b')
    plt.xlabel('Factor')
    plt.ylabel('Count')
    plt.savefig("static/dataset/ff_g2.png")

    return render_template('process3.html',data1=data1,data2=data2,data3=data3,data4=data4,dat5=dat5)


def process4():

    ##########
    plt.rc("axes.spines", right=False, top=False)
    plt.rc("font", family="serif")
    data = pd.read_csv(
        "static/dataset/data1.csv",
        header=0,
        encoding="latin-1",
        usecols=[0, 1],
        names=["label", "text"],
    )
    dat=data.head()
 
    data["length (words)"] = data["text"].str.split().apply(len)
    dataval=data.groupby("label").agg([min, max, "mean"])
    print(data.groupby("label").agg([min, max, "mean"]))
    ax = data.boxplot(by="label", figsize=(6, 4.5))
    _ = ax.set_title("")
 

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], random_state=8, stratify=data["label"]
    )

 
    ############################################################################

    data = pd.read_csv(
        "static/dataset/data2.csv",
        header=0,
        encoding="latin-1",
        usecols=[0, 1],
        names=["label", "text"],
    )
    dat=data.head()

    

    data["length (words)"] = data["text"].str.split().apply(len)
    dataval2=data.groupby("label").agg([min, max, "mean"])
    print(data.groupby("label").agg([min, max, "mean"]))
    ax = data.boxplot(by="label", figsize=(6, 4.5))
    _ = ax.set_title("")
 

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], random_state=8, stratify=data["label"]
    )
 
    #############################################################################
    data = pd.read_csv(
        "static/dataset/data3.csv",
        header=0,
        encoding="latin-1",
        usecols=[0, 1],
        names=["label", "text"],
    )
    dat=data.head()

    data["length (words)"] = data["text"].str.split().apply(len)
    dataval3=data.groupby("label").agg([min, max, "mean"])
    print(data.groupby("label").agg([min, max, "mean"]))
    ax = data.boxplot(by="label", figsize=(6, 4.5))
    _ = ax.set_title("")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], random_state=8, stratify=data["label"]
    )
    
    df = pd.read_csv('static/dataset/SEFACED_Email_Forensic_Dataset1.csv',delimiter=',',encoding='latin-1')
    df = df[['Class_Label','Text']]
    df = df[pd.notnull(df['Text'])]
    df.rename(columns = {'Message':'Text'}, inplace = True)
    print(df.head())
    data1=[]        
    dsf=df.shape
    print(dsf)
    print(dsf[0])
    df.index = range(dsf[0])
    df['Text'].apply(lambda x: len(x.split(' '))).sum()

    cnt_pro = df['Class_Label'].value_counts()

    return render_template('process4.html',dataval=dataval,dataval2=dataval2,dataval3=dataval3)



##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)


