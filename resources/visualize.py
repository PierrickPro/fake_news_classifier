import re
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

'''
plots bar chart of fake and real news 
'''
def labels_bar_plot(news):
    sns.barplot(news['label'].value_counts().index, news['label'].value_counts())

'''
plots confusion matrix to visualise prediciton results compared to y_test
'''
def plot_cf_matrix(y_test, pred):
    score = metrics.accuracy_score(y_test, pred)
    cf_matrix = metrics.confusion_matrix(y_test, pred)
    
    s = sns.heatmap(cf_matrix, cmap='Blues', annot=True, fmt='d', 
                    xticklabels=['fake', 'real'], yticklabels=['fake', 'real'])
    
    s.set_xlabel('Predicted Label', fontsize=10)
    s.set_ylabel('True Label', fontsize=10)
    s.set_title('Accuracy: %0.3f%%' % (score));

'''
- remove html code
- remove punctuation
- make everything lower case
'''
def clean(text):
    text = re.sub(r'[^\w\s]','', text)
    text = re.sub('<[^>]*>', '', text)
    text = text.lower()
    return text

'''
plot pie charts to visualise prediciton results compared to y_test
'''
def plot_pie_chart(y_test, pred):
    plt.figure()
    labels = 'Real','Fake'
    colors = ['#ff9999','#66b3ff']
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].set_title("True Label")
    plt.sca(ax[0])
    plt.pie([len(y_test[y_test==0]), len(y_test[y_test==1])],colors = colors,
    labels=labels, autopct='%1.1f%%', startangle=90)

    plt.sca(ax[1])
    ax[1].set_title("Predicted Label")
    plt.pie([len(pred[pred==0]), len(pred[pred==1])], colors = colors, 
    labels=labels, autopct='%1.1f%%', startangle=90)
