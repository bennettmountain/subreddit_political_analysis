import matplotlib
matplotlib.use('Agg')
import simplejson as json
import bz2
import os
import ast
import matplotlib.pyplot as plt
import textblob
from textblob import TextBlob
import lzma
import nltk
import numpy as np
import pandas as pd
from textblob import TextBlob
from textblob import Blobber
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import operator
import string
import gzip
import math
import matplotlib.colors as colors
from sklearn.cluster import SpectralCoclustering
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
#import sklearn.linear_model as lm
import scipy
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ngrams',type=int,default=1)
parser.add_argument('--limit')
parser.add_argument('--num_words',type=int,default=20)
parser.add_argument('--num_eig',type=int,default=0)
parser.add_argument('--penalty',type=str,default='l2')
parser.add_argument('--C',type=float,default=1.0)
parser.add_argument('--no_biggest_words',action='store_true')
parser.add_argument('--norm',type=int,default=2)
parser.add_argument('--no_cocluster',action='store_true')
parser.add_argument('--features',choices=['counts','tf','tfidf'],default='tfidf')
args = parser.parse_args()

subreddit_list = ['politics','uspolitics','AmericanPolitics','progressive','democrats','Liberal','Republican',
                    'Conservative','Libertarian']

subreddit_members = {'politics':5.5E6,'uspolitics':1.65E4,'AmericanPolitics':1.12E4,'progressive':6.17E4,'democrats':1.11E5,
                        'Liberal':7.78E4,'Republican':9.21E4,'Conservative':268E5,'Libertarian':345E5}
                        
# get more conservative and liberal words???
liberal_words = ["progressive","Biden","universal basic income","AOC", "Ocasio-Cortez", "liberal", "democrat", "Obama", "Clinton", "Sanders", 
"green new deal", "leftist", "Yang", "Warren", "Kamala", "medicare for all", "obamacare", "pro choice"]

conservative_words = ["Cheney","Shapiro","Koch","Paul Ryan","Rand Paul","Bush","Palin","Mattis","McCain","Romney", "Trump",
"Cruz", "republican", "Kushner", "conservative", "GOP", "Pence"]

# output is of the form {"subreddit_1_name":[('post_1_title',post_1_score),(''post_2_title',post_2_score),...], "subreddit_2_name"...}
scores = {}
scores_dates = {} # of the form{'date_1':[('subreddit_1',score),('subreddit_2',score)],'date_2':...}
aggregated_titles = {}
bigram_count = {}
text_for_matrix = []
labels = []
 
def populate_dicts(f):
    content = f
    for line in content:
        try:
            post = json.loads(line)
            sub = post.get("subreddit")
            if sub in subreddit_list:
                if post.get("score") > 10: # arbitrary threshold
                    log_normalized_score = (math.log(post.get("score")) * 1.0) / subreddit_members.get(sub)
                    if sub in data:
                        data[sub].append([post.get("title"), log_normalized_score])
                    if sub not in data:
                        data[sub] = [[post.get("title"), log_normalized_score]]
                    if sub in master_data:
                        master_data[sub].append([post.get("title"), log_normalized_score])
                    if sub not in master_data:
                        master_data[sub] = [[post.get("title"), log_normalized_score]]
        except:
            pass

def open_files():
    '''
    Goes through the directory containing all the data files.
    '''
    #path = os.path.expanduser('/data/files.pushshift.io/reddit/submissions')
    os.chdir('/data/files.pushshift.io/reddit/submissions')
    #files = [f for f in os.listdir(path)] #issue with RS_2011-01.bz2 having some non unicode-32 characters.
    #files = ['RS_2017-11.bz2','RS_2017-10.bz2','RS_2017-08.bz2','RS_2017-07.bz2','RS_2017-06.bz2','RS_2017-05.bz2','RS_2017-04.bz2']
    #files = ['RS_2011-01.bz2', 'RS_2012-01.bz2','RS_2013-01.bz2','RS_2014-01.bz2','RS_2015-01.gz','RS_2016-01.gz','RS_2017-01.bz2','RS_2018-01.xz','RS_2019-01.gz']
    #files = ['RS_2012-01.bz2','RS_2013-01.bz2','RS_2014-01.bz2','RS_2015-01.gz','RS_2016-01.gz','RS_2017-01.bz2','RS_2018-01.xz','RS_2019-01.gz']
    files = ['RS_2011-01.bz2']
    for i in files:
        year = i[3:7]
        if year == '2011':
            with open("/home/bmountain/dm_project/output_2011.json", "r+") as json_date_file:
                data = json.load(json_date_file)
                with open("/home/bmountain/dm_project/output_master.json", "r+") as json_master:
                    master_data = json.load(json_master)
                    with bz2.open(i, "r") as content:
                        print(datetime.datetime.now(), 'opening ' + i)
                        for line in content:
                            try:
                                post = json.loads(line)
                                sub = post.get("subreddit")
                                if sub in subreddit_list:
                                    if post.get("score") > 10: # arbitrary threshold
                                        log_normalized_score = math.log(post.get("score")) * 1.0
                                        if sub in data:
                                            data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in data:
                                            data[sub] = [[post.get("title"), log_normalized_score]]
                                        if sub in master_data:
                                            master_data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in master_data:
                                            master_data[sub] = [[post.get("title"), log_normalized_score]]
                            except:
                                pass
                with open("/home/bmountain/dm_project/output_master.json", "w") as master_write:
                    json.dump(master_data,master_write)
            with open("/home/bmountain/dm_project/output_2011.json","w") as j_file:
                json.dump(data,j_file)
        if year == '2012':
            with open("/home/bmountain/dm_project/output_2012.json", "r+") as json_date_file:
                data = json.load(json_date_file)
                with open("/home/bmountain/dm_project/output_master.json", "r+") as json_master:
                    master_data = json.load(json_master)
                    with bz2.open(i, "r") as content:
                        print(datetime.datetime.now(), 'opening ' + i)
                        for line in content:
                            try:
                                post = json.loads(line)
                                sub = post.get("subreddit")
                                if sub in subreddit_list:
                                    if post.get("score") > 10: # arbitrary threshold
                                        log_normalized_score = math.log(post.get("score")) * 1.0
                                        if sub in data:
                                            data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in data:
                                            data[sub] = [[post.get("title"), log_normalized_score]]
                                        if sub in master_data:
                                            master_data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in master_data:
                                            master_data[sub] = [[post.get("title"), log_normalized_score]]
                            except:
                                pass
                with open("/home/bmountain/dm_project/output_master.json", "w") as master_write:
                    json.dump(master_data,master_write)
            with open("/home/bmountain/dm_project/output_2012.json","w") as j_file:
                json.dump(data,j_file)
        if year == '2013':
            with open("/home/bmountain/dm_project/output_2013.json", "r+") as json_date_file:
                data = json.load(json_date_file)
                with open("/home/bmountain/dm_project/output_master.json", "r+") as json_master:
                    master_data = json.load(json_master)
                    with bz2.open(i, "r") as content:
                        print(datetime.datetime.now(), 'opening ' + i)
                        for line in content:
                            try:
                                post = json.loads(line)
                                sub = post.get("subreddit")
                                if sub in subreddit_list:
                                    if post.get("score") > 10: # arbitrary threshold
                                        log_normalized_score = math.log(post.get("score")) * 1.0
                                        if sub in data:
                                            data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in data:
                                            data[sub] = [[post.get("title"), log_normalized_score]]
                                        if sub in master_data:
                                            master_data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in master_data:
                                            master_data[sub] = [[post.get("title"), log_normalized_score]]
                            except:
                                pass
                with open("/home/bmountain/dm_project/output_master.json", "w") as master_write:
                    json.dump(master_data,master_write)
            with open("/home/bmountain/dm_project/output_2013.json","w") as j_file:
                json.dump(data,j_file)
        if year == '2014':
            with open("/home/bmountain/dm_project/output_2014.json", "r+") as json_date_file:
                data = json.load(json_date_file)
                with open("/home/bmountain/dm_project/output_master.json", "r+") as json_master:
                    master_data = json.load(json_master)
                    with bz2.open(i, "r") as content:
                        print(datetime.datetime.now(), 'opening ' + i)
                        for line in content:
                            try:
                                post = json.loads(line)
                                sub = post.get("subreddit")
                                if sub in subreddit_list:
                                    if post.get("score") > 10: # arbitrary threshold
                                        log_normalized_score = math.log(post.get("score")) * 1.0
                                        if sub in data:
                                            data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in data:
                                            data[sub] = [[post.get("title"), log_normalized_score]]
                                        if sub in master_data:
                                            master_data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in master_data:
                                            master_data[sub] = [[post.get("title"), log_normalized_score]]
                            except:
                                pass
                with open("/home/bmountain/dm_project/output_master.json", "w") as master_write:
                    json.dump(master_data,master_write)
            with open("/home/bmountain/dm_project/output_2014.json","w") as j_file:
                json.dump(data,j_file)
        if year == '2015':
            with open("/home/bmountain/dm_project/output_2015.json", "r+") as json_date_file:
                data = json.load(json_date_file)
                with open("/home/bmountain/dm_project/output_master.json", "r+") as json_master:
                    master_data = json.load(json_master)
                    with gzip.open(i) as content:
                        print(datetime.datetime.now(), 'opening ' + i)
                        for line in content:
                            try:
                                post = json.loads(line)
                                sub = post.get("subreddit")
                                if sub in subreddit_list:
                                    if post.get("score") > 10: # arbitrary threshold
                                        log_normalized_score = math.log(post.get("score")) * 1.0
                                        if sub in data:
                                            data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in data:
                                            data[sub] = [[post.get("title"), log_normalized_score]]
                                        if sub in master_data:
                                            master_data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in master_data:
                                            master_data[sub] = [[post.get("title"), log_normalized_score]]
                            except:
                                pass
                with open("/home/bmountain/dm_project/output_master.json", "w") as master_write:
                    json.dump(master_data,master_write)
            with open("/home/bmountain/dm_project/output_2015.json","w") as j_file:
                json.dump(data,j_file)
        if year == '2016':
            with open("/home/bmountain/dm_project/output_2016.json", "r+") as json_date_file:
                data = json.load(json_date_file)
                with open("/home/bmountain/dm_project/output_master.json", "r+") as json_master:
                    master_data = json.load(json_master)
                    with gzip.open(i) as content:
                        print(datetime.datetime.now(), 'opening ' + i)
                        for line in content:
                            try:
                                post = json.loads(line)
                                sub = post.get("subreddit")
                                if sub in subreddit_list:
                                    if post.get("score") > 10: # arbitrary threshold
                                        log_normalized_score = math.log(post.get("score")) * 1.0
                                        if sub in data:
                                            data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in data:
                                            data[sub] = [[post.get("title"), log_normalized_score]]
                                        if sub in master_data:
                                            master_data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in master_data:
                                            master_data[sub] = [[post.get("title"), log_normalized_score]]
                            except:
                                pass
                with open("/home/bmountain/dm_project/output_master.json", "w") as master_write:
                    json.dump(master_data,master_write)
            with open("/home/bmountain/dm_project/output_2016.json","w") as j_file:
                json.dump(data,j_file)
        if year == '2017':
            with open("/home/bmountain/dm_project/output_2017.json", "r+") as json_date_file:
                data = json.load(json_date_file)
                with open("/home/bmountain/dm_project/output_master.json", "r+") as json_master:
                    master_data = json.load(json_master)
                    with bz2.open(i, "r") as content:
                        print(datetime.datetime.now(), 'opening ' + i)
                        for line in content:
                            try:
                                post = json.loads(line)
                                sub = post.get("subreddit")
                                if sub in subreddit_list:
                                    if post.get("score") > 10: # arbitrary threshold
                                        log_normalized_score = math.log(post.get("score")) * 1.0
                                        if sub in data:
                                            data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in data:
                                            data[sub] = [[post.get("title"), log_normalized_score]]
                                        if sub in master_data:
                                            master_data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in master_data:
                                            master_data[sub] = [[post.get("title"), log_normalized_score]]
                            except:
                                pass
                with open("/home/bmountain/dm_project/output_master.json", "w") as master_write:
                    json.dump(master_data,master_write)
            with open("/home/bmountain/dm_project/output_2017.json","w") as j_file:
                json.dump(data,j_file)
        if year == '2018':
            with open("/home/bmountain/dm_project/output_2018.json", "r+") as json_date_file:
                data = json.load(json_date_file)
                with open("/home/bmountain/dm_project/output_master.json", "r+") as json_master:
                    master_data = json.load(json_master)
                    with lzma.open(i, mode='rt') as content:
                        print(datetime.datetime.now(), 'opening ' + i)
                        for line in content:
                            try:
                                post = json.loads(line)
                                sub = post.get("subreddit")
                                if sub in subreddit_list:
                                    if post.get("score") > 10: # arbitrary threshold
                                        log_normalized_score = math.log(post.get("score")) * 1.0
                                        if sub in data:
                                            data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in data:
                                            data[sub] = [[post.get("title"), log_normalized_score]]
                                        if sub in master_data:
                                            master_data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in master_data:
                                            master_data[sub] = [[post.get("title"), log_normalized_score]]
                            except:
                                pass
                with open("/home/bmountain/dm_project/output_master.json", "w") as master_write:
                    json.dump(master_data,master_write)
            with open("/home/bmountain/dm_project/output_2018.json","w") as j_file:
                json.dump(data,j_file)
        if year == '2019':
            with open("/home/bmountain/dm_project/output_2019.json", "r+") as json_date_file:
                data = json.load(json_date_file)
                with open("/home/bmountain/dm_project/output_master.json", "r+") as json_master:
                    master_data = json.load(json_master)
                    with gzip.open(i) as content:
                        print(datetime.datetime.now(), 'opening ' + i)
                        for line in content:
                            try:
                                post = json.loads(line)
                                sub = post.get("subreddit")
                                if sub in subreddit_list:
                                    if post.get("score") > 10: # arbitrary threshold
                                        log_normalized_score = (math.log(post.get("score")) * 1.0)
                                        if sub in data:
                                            data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in data:
                                            data[sub] = [[post.get("title"), log_normalized_score]]
                                        if sub in master_data:
                                            master_data[sub].append([post.get("title"), log_normalized_score])
                                        if sub not in master_data:
                                            master_data[sub] = [[post.get("title"), log_normalized_score]]
                            except:
                                pass
                with open("/home/bmountain/dm_project/output_master.json", "w") as master_write:
                    json.dump(master_data,master_write)
            with open("/home/bmountain/dm_project/output_2019.json","w") as j_file:
                json.dump(data,j_file)
       
def aggregate_titles(subreddit):
    '''
    Aggregate all the post titles for each subreddit
    '''
    with open("/home/bmountain/dm_project/output_master.json", "r+") as json_file:
        data = json.load(json_file)
        aggregated_titles[subreddit] = " ".join(j[0] for j in data[subreddit])
        
def create_metric():
    '''
    Creates a bar graph with each subreddit and their aggregated political bias score
    TODO: go through scores_dates and populate it that way
    {'2011':[('uspolitics',0), ']}
    '''
    for subreddit in subreddit_list:
        scores[subreddit] = 0
    for subreddit in subreddit_list:
        for date in scores_dates:
            sub_list = scores_dates[date]
            for sub in sub_list:
                if sub[0] == subreddit:
                    scores[subreddit] += sub[1]

def create_scores_for_each_date():
    '''
    TODO:
    implement with the separate json files
    '''
    json_date_files = ["/home/bmountain/dm_project/output_2011.json", "/home/bmountain/dm_project/output_2012.json", "/home/bmountain/dm_project/output_2013.json",
                "/home/bmountain/dm_project/output_2014.json", "/home/bmountain/dm_project/output_2015.json", "/home/bmountain/dm_project/output_2016.json",
                "/home/bmountain/dm_project/output_2017.json", "/home/bmountain/dm_project/output_2018.json", "/home/bmountain/dm_project/output_2019.json"]
    for i in json_date_files:
        date = i[34:38] + "-01"
        with open(i, "r+") as json_file:
            subreddit_dict = json.load(json_file)
            scores_dates[date] = []
            for subreddit_in_original_list in subreddit_list:
                if subreddit_in_original_list not in subreddit_dict:
                    scores_dates[date].append((subreddit_in_original_list, 0))
            for sub in subreddit_dict:
                posts = subreddit_dict[sub]
                score = 0
                num_posts = 0
                for p in posts:
                    num_posts+=1
                    num_cons_words = 0
                    num_lib_words = 0
                    title_category_factor = 1
                    if type(p[0]) == str:
                        title_string = p[0]
                        title_score = p[1]
                    else:
                        title_string = p[0][0]
                        title_score = p[0][1]
                    for word in title_string: 
                        if word in conservative_words:
                            cons_words+=1
                        elif word in liberal_words:
                            lib_words+=1
                    if num_cons_words >= num_lib_words:
                        title_category_factor = -1
                    title = TextBlob(title_string)
                    if title.sentiment.subjectivity > 0.0:
                        sntmnt = (title.sentiment.polarity * 50.0 * title.sentiment.subjectivity)
                    else:
                        sntmnt = title.sentiment.polarity
                    score += ((title_score * 1.0) * sntmnt) * title_category_factor
                score /= (num_posts * 1.0)
                scores_dates[date].append((sub,score))
def create_spaghetti_plot():
    '''
    Creates a spaghetti plot. X axis is dates, y axis is scores, each line is a subreddit
    TODO:
    implement for new methods
    '''
    # style
    plt.style.use('seaborn-darkgrid')
    
    # create a color palette
    palette = plt.get_cmap('tab10')
    
    # multiple line plot
    num=0
    # want to loop through each subreddit first then get the dates
    with open("/home/bmountain/dm_project/output_master.json", "r+") as json_file:
        data = json.load(json_file)
        for sub in data:
            num+=1
            sub_scores = [] # should only need to scores_dates
            date_list = []
            for date in scores_dates:
                date_list.append(date)
                subs = scores_dates[date] # list of tuples
                for tup in subs:
                    if tup[0] == sub: # found the corresponding subreddit
                        sub_scores.append(tup[1])
                
            plt.xticks(range(len(date_list)), date_list, rotation=70)
            plt.plot(range(len(date_list)), sub_scores, marker='', color=palette(num), linewidth=1, label=sub)
    
    # Add legend
    plt.legend(loc=2, ncol=2)
    
    # Add titles
    plt.title("Suberddit Bias Over Time")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.savefig('/home/bmountain/dm_project/spaghetti_plot.png', bbox_inches = "tight")
    plt.clf()
    plt.cla()
    plt.close()


def create_bigrams(subreddit):
    '''
    Creates a dict with the frequency of the bigrams for each subreddit
    '''
    bigram_count_mini = {} # holds bigram frequencies for each subreddit
    text = aggregated_titles[subreddit]
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.lower()
    words = text.split()
    words = [i for i in words if i not in set(STOPWORDS)]
    bi_grams = list(zip(words, words[1:]))
    for gram in bi_grams:
        if gram not in bigram_count_mini:
            bigram_count_mini[gram] = 1
        else:
            bigram_count_mini[gram]+=1
    bigram_count[subreddit] = bigram_count_mini


def plot_matrix(model,all_feature_names_arg,mat,filename,force_no_cocluster=False):
    print(datetime.datetime.now(),'plot_matrix')
    print('  mat.shape=',mat.shape)
    plt.figure(figsize=(10,4))

    # set the x-axis to only include the biggest words
    if not args.no_biggest_words:
        l2_norms=np.linalg.norm(mat,axis=0,ord=args.norm)
        indices = l2_norms.argsort()[-args.num_words:]
        mat = mat[:,indices]
        all_feature_names = all_feature_names_arg
        words = [ all_feature_names[i] for i in indices ]
        plt.xticks(range(0,len(words)),words,rotation=-90)

    # cocluster the axes
    if not args.no_cocluster and not force_no_cocluster:
        clustering = SpectralCoclustering(n_clusters=6,random_state=1).fit(mat)
        col_indices = np.argsort(clustering.column_labels_)
        mat = mat[:,col_indices]
        try:
            words = [ words[i] for i in col_indices ]
            plt.xticks(range(0,len(words)),words,rotation=-90)
        except:
            pass

    # plot the figure
    plt.imshow(
            mat,
            aspect='auto',
            cmap='RdBu',
            norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=-1e6, vmax=1e6)
            )

    plt.yticks([0,1,2,3,4,5,6,7,8],model.classes_)
    plt.ylim(-0.5,8.5)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)

def load_text_labels_for_matrix():
    with open("/home/bmountain/dm_project/output_master.json", "r+") as json_file:
        data = json.load(json_file)
        subreddit_keys = data
        for sub in subreddit_keys:
            posts = subreddit_keys[sub]
            for post in posts:
                text_for_matrix.append(post[0])
                labels.append(sub)


def plot_bigrams():
    for subreddit in bigram_count:
        bigram_dict = bigram_count[subreddit]
        bigram_string = '/home/bmountain/dm_project/'+ subreddit + '_top_10_bigrams_.png'
        top_10_bigrams_ = dict(sorted(bigram_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
        plt.bar(range(len(top_10_bigrams_)), list(top_10_bigrams_.values()), align='center')
        plt.xticks(range(len(top_10_bigrams_)), list(top_10_bigrams_.keys()), rotation = 70)
        plt.ylabel('Count')
        plt.xlabel('Bigrams')
        plt.title('r/' + subreddit + ' Top 10 Bigrams')
        plt.savefig(bigram_string, bbox_inches = "tight")
        plt.clf()
        plt.cla()
        plt.close()

def plot_metric():
    plt.bar(range(len(scores)), list(scores.values()), align='center')
    plt.xticks(range(len(scores)), list(scores.keys()), rotation = 70)
    plt.xlabel('Subreddits')
    plt.ylabel('Bias')
    plt.title('Subreddit Bias Scores')
    plt.savefig('/home/bmountain/dm_project/subreddit_scores.png', bbox_inches = "tight")
    plt.clf()
    plt.cla()
    plt.close()

def plot_wordclouds(subreddit):
    '''
    Creates a wordcloud for each subreddit
    '''
    agg_text = aggregated_titles[subreddit]
    stopwords= set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords,max_font_size=50, max_words=100, background_color="white").generate(agg_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    wordcloud_string = '/home/bmountain/dm_project/'+ subreddit + '_wordcloud.png'
    wordcloud.to_file(wordcloud_string)
    plt.clf()
    plt.cla()
    plt.close()

def main():
    #open_files()
    print('done opening all files')
    print(datetime.datetime.now(),' starting aggregating titles and creating metric for each subreddit')
    for subreddit in subreddit_list: # switched output for subreddit_list
        aggregate_titles(subreddit)
    print(datetime.datetime.now(),' done aggregating titles and creating metric for each subreddit')
    for subreddit in aggregated_titles:
        create_bigrams(subreddit)
        plot_wordclouds(subreddit)
    print(datetime.datetime.now(),' done creating bigrams and plotting wordclouds for each subreddit')
    plot_bigrams()
    print(datetime.datetime.now(),' create_scores_for_each_date()')
    create_scores_for_each_date()
    create_metric()
    print(scores)
    plot_metric()
    create_spaghetti_plot()
    load_text_labels_for_matrix()
    print(datetime.datetime.now(), ' plotting matrix')
    

    print(datetime.datetime.now(),'CountVectorizer()')
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(ngram_range=(1,args.ngrams),stop_words='english')
    features = count_vect.fit_transform(text_for_matrix)
    features = features.astype(np.float64)
    all_feature_names = count_vect.get_feature_names()
    
    #not sure if i need these,
    # also unsure about completely getting rid of anything to do with args
    if args.features=='tf':
        print(datetime.datetime.now(),'TF')
        from sklearn.feature_extraction.text import TfidfTransformer
        tf_transformer = TfidfTransformer(use_idf=False).fit(features)
        features = tf_transformer.transform(features)
        print('  features.shape=',features.shape)

    if args.features=='tfidf':
        print(datetime.datetime.now(),'TF-IDF')
        from sklearn.feature_extraction.text import TfidfTransformer
        tf_transformer = TfidfTransformer(use_idf=True).fit(features)
        features = tf_transformer.transform(features)
        print('  features.shape=',features.shape)

    print(datetime.datetime.now(),'PCA')
    model = LogisticRegression(
            penalty=args.penalty,
            C=args.C,
            solver='liblinear',
            class_weight='balanced',
            multi_class='auto'
            )
    if args.num_eig>0:
        gram = np.dot(np.transpose(features),features)
        print('  gram.shape=',gram.shape)
        w, v = scipy.sparse.linalg.eigsh(gram,k=args.num_eig)
        print('  w.shape=',w.shape)
        print('  v.shape=',v.shape)

        plt.figure(figsize=(20,10))
        plt.bar(range(0,args.num_eig),w)
        plt.savefig('/home/bmountain/dm_project/eigenvalues.png')

        plot_matrix(model,all_feature_names,np.transpose(v),filename='/home/bmountain/dm_project/eigenvectors.png') ,#force_no_cocluster=True)
    
    print(datetime.datetime.now(),'logreg')
    
    
    model.fit(features, labels)
    print('  model.coef_.shape=',model.coef_.shape)

    
    plot_matrix(model,all_feature_names,model.coef_,'/home/bmountain/dm_project/matrix.png')

    
main()