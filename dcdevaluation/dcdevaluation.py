# Imports

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

# Metrics
from scipy.stats             import ks_2samp
from sklearn.metrics         import confusion_matrix, auc, roc_curve, recall_score, accuracy_score, precision_score
from sklearn.metrics         import roc_auc_score, precision_recall_curve, average_precision_score, f1_score

# Definição da classe evaluators

class Evaluators:
    
    def __init__(self, predicted, true):
        
        self.predicted = predicted
        self.true      = true
        self.df        = pd.DataFrame({'True' : true     ,
                                       'Pred' : predicted})    
        
# ------------------- KS -------------------         
        
    def get_ks(self):
    
        """ Function to calculate the KS score.

        --------------  Attributes  --------------
        predicted = Values predicted by the model
        true      = The true values of the target feature

        --------------  Output  --------------
        Model's KS score
        """    
       
        self.ks = round(ks_2samp(self.df.loc[self.df['True'] == 0, 'Pred'],
                                 self.df.loc[self.df['True'] == 1, 'Pred'])[0]*100, 2)
        
# ------------------- AUC ------------------- 
        
    def get_auc(self):

        """ Function to calculate the ROC AUC score.

        --------------  Attributes  --------------
        predicted = Values predicted by the model
        true      = The true values of the target feature

        --------------  Output  --------------
        Model's AUC score
        """ 

        self.auc = round(roc_auc_score(np.asarray(self.df['True']),
                                       np.asarray(self.df['Pred']))*100, 2)
        
# ------------------- F1 -------------------       

    def get_f1(self):
    
        """ Function to calculate F1 score.

        --------------  Attributes  --------------
        predicted = Values predicted by the model
        true      = The true values of the target feature

        --------------  Output  --------------
        Model's F1 score
        """
        
        self.f1 = round(f1_score(np.asarray(self.df['True']),
                                 np.asarray(self.df['Pred'].round()),
                                 average = 'binary')*100,2)
        
# ------------------- Recall -------------------       

    def get_recall(self):

        """ Function to calculate the recall score.

        --------------  Attributes  --------------
        predicted = Values predicted by the model
        true      = The true values of the target feature

        --------------  Output  --------------
        Model's F1 score
        """
        self.recall = round(recall_score(np.asarray(self.df['True']),
                                         np.asarray(self.df['Pred'].round()))*100,2)
        
# ------------------- Precision -------------------  

    def get_precision(self):
    
        """ Function to calculate the precision score.

        --------------  Attributes  --------------
        predicted = Values predicted by the model
        true      = The true values of the target feature

        --------------  Output  --------------
        Model's F1 score
        """

        self.precision = round(precision_score(np.asarray(self.df['True']),
                                               np.asarray(self.df['Pred'].round()))*100,2)
        
# ------------------- Precision -------------------

    def get_accuracy(self):
    
        """ Function to calculate the accuracy score.

        --------------  Attributes  --------------
        predicted = Values predicted by the model
        true      = The true values of the target feature

        --------------  Output  --------------
        Model's F1 score
        """
        self.accuracy = round(accuracy_score(np.asarray(self.df['True']),
                                             np.asarray(self.df['Pred'].round()),
                                             normalize=True)*100,2)
        
# ------------------- All metrics ------------------- 

    def evaluate(self, verbose = True):
        
        """ Calls all evaluation metrics.

        --------------  Attributes  --------------
        predicted = Values predicted by the model
        true      = The true values of the target feature

        --------------  Output  --------------
        All mapped evaluation metrics
        """
        
        self.get_ks()
        self.get_auc()
        self.get_f1()
        self.get_recall()
        self.get_precision()
        self.get_accuracy()
        
        if verbose == True:
            return print(' KS: {}\n'       .format(self.ks ),
                         'AUC: {}\n'       .format(self.auc),
                         'F1: {}\n'        .format(self.f1 ),
                         'Precision: {}\n' .format(self.precision),
                         'Recall: {}\n'    .format(self.recall),
                         'Accuracy: {}\n'  .format(self.accuracy))
        
# ------------------- Make table ------------------- 

    def to_table(self, dataset = '0'):
        
        """ Makes a table containing all metrics.
        
        *MUST BE RUN AFTER "EVALUATE" method.
        
        --------------  Attributes  --------------
        dataset = Name of the dataset under evaluation
        * must be string
        e.g.: Train, Test, Validation

        --------------  Output  --------------
        A table contaning all mapped evaluation 
        metrics
        
        A transposed table contaning all mapped 
        evaluation metrics
        """         
        
        self.evaluate(verbose = False)
        
        self.metric_df = pd.DataFrame({'KS'        : self.ks       ,
                                       'AUC'       : self.auc      ,
                                       'F1'        : self.f1       ,
                                       'Recall'    : self.recall   ,
                                       'Precision' : self.precision,
                                       'Accuracy'  : self.accuracy }, index = [dataset])
        
        self.t_metric_df = self.metric_df.T
        
        return self.metric_df
        
# ------------------- Spliting into bins ------------------- 

    def get_bins(self, bins = 10):
        
        """ Splits observations in evenly sized bins
        
        --------------  Attributes  --------------
        bins = Number of groups that will be created.
        Default is 10.
        
        --------------  Output  --------------
        A DataFrame groupped by differente splits
        """  
        
        self.df['Split']       = pd.qcut(self.df['Pred'], q = bins, labels = False)
        
        self.bins_df           = pd.DataFrame()
        self.bins_df           = self.df.groupby('Split').agg({'True' : 'sum'}).reset_index()
        self.bins_df['Count']  = self.df.groupby('Split').agg({'True' : 'count'}).reset_index()['True']
        self.bins_df['%Decil'] = round((self.bins_df['True'] / self.bins_df['Count']) * 100, 2)
        self.bins_df['%Total'] = round((self.bins_df['True'] / self.df.shape[0]) * 100, 2)
        
        self.bins_df.drop(['Count', 'True'],
                           axis = 1        , 
                           inplace = True  )
        
# ------------------- Create Bad Rate Graph ------------------- 

    def split_rate_graph(self, bins = 10):
        
            
        """ Creates a graph showing the bad rate along
        different splits
        
        --------------  Attributes  --------------
        bins = Number of groups that will be created.
        Default is 10.
        
        --------------  Output  --------------
        A Graph showing different splits
        """  
        
        
        self.get_bins(bins) 
        label    = []
        values   = list(self.bins_df['%Decil'])
        
        i = 0
        
        while i < bins:
            
            x = 'S' + str(i)
            
            label.append(x)
            
            i = i + 1
        
        x      = np.arange(len(label))
        width  = 0.6
        
        
        fig, ax = plt.subplots(figsize = (12,8))
        
        rects   = ax.bar(x - width/2, values, width, label = '% True', color = 'purple')
                
        ax.set_ylim([0,120])
        ax.set_ylabel('%')
        ax.set_xlabel('Split')
        ax.set_title('Good/Bad Rate Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(label)
        ax.legend()
    
        def autolabel(rects):
            
            """Attach a text label above each bar in *rects*,
            displaying its height."""
            
            for rect in rects:
                height = rect.get_height()
                plt.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


        autolabel(rects)

        plt.show()
        
# ------------------- Find Cut Spot ------------------- 

    def find_cut(self, minimum = 0, maximum = 20):
        
        """ Cuts the scored dataframe in 20 different points
        to help the scientist find the optimal cutting point
        
        --------------  Attributes  --------------
        minimum = Starting cutting point.
        Default is 0.
        maximum = Ending cutting point
        Default is 20.
        
        --------------  Output  --------------
        A table showing the Precision, Recall, F1 and count
        for different cutting points
        """  
        
        cut_df = pd.DataFrame(columns = {'Cut', 'Precision', 'Recall', 'F1',  'Count'})
        
        for i in range(minimum ,maximum):
            
            cut = i / 20
            
            tp = self.df.loc[(self.df['True'] == 1) & (self.df['Pred'] >= cut)]
            fn = self.df.loc[(self.df['True'] == 1) & (self.df['Pred'] <  cut)]
            fp = self.df.loc[(self.df['True'] == 0) & (self.df['Pred'] >= cut)]
            
            cut_df.at[i, 'Cut']       = round(cut, 2)
            
            if (fp.shape[0] + tp.shape[0]) == 0:
                cut_df.at[i, 'Precision'] == 'undefined'
                
            else:
                cut_df.at[i, 'Precision'] = round((tp.shape[0] / (tp.shape[0] + fp.shape[0]))* 100, 2)
                
            cut_df.at[i, 'Recall']    = round((tp.shape[0] / (tp.shape[0] + fn.shape[0]))* 100, 2)
            cut_df.at[i, 'F1']        = round((tp.shape[0] / (tp.shape[0] + (0.5 * (fp.shape[0] + fn.shape[0]))))* 100, 2)
            cut_df.at[i, 'Count']     = self.df[self.df['Pred'] >= cut].shape[0]
            
        cut_df = cut_df[['Cut', 'Count','Precision','Recall','F1']].set_index('Cut')
        self.cut_df = cut_df
            
        return cut_df
    
# ------------------- ROC AUC Graph ------------------- 

    def ROC_curve(self, label = 'Dataset'):
        
        
        """ Creates a graph showing the ROC curve.
           
        --------------  Output  --------------
        Graph with the ROC curve vs coin.
        """  

        fig = plt.subplots(figsize = (12,8))
        
        title_font = {'fontname' : 'DejaVu Sans',
                      'size'     : '17',
                      'weight'   : 'bold'}
        axis_font  = {'fontname' : 'DejaVu Sans',
                      'size'     : '12'}

        lr_fpr_0, lr_tpr_0, _ = roc_curve(np.asarray(self.df['True']), np.asarray(self.df['Pred']))

        plt.plot([0.0, 1.0], [0.0, 1.0], 'r--', linewidth = 0.5, label = 'Coin', color = 'black')
        plt.plot(lr_fpr_0, lr_tpr_0, linewidth = 0.5, label = label, color = 'blue')

        plt.title('ROC curve', title_font)
        plt.xlabel('False Positive Rate', axis_font)
        plt.ylabel('True Positive Rate', axis_font)
        plt.legend()

