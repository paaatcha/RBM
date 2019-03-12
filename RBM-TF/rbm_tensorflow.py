# -*- coding: utf-8 -*-
'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com

This class implements the restricted Boltzmann machine (RBM) in TensorFlow according to
Hinton's guideline: 

[1] Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." 
Momentum, v. 9, n. 1, p. 926, 2010.

You can chose either Gaussian-Bernoulli RBM or Bernoulli-Bernoulli RBM. All the code is 
very commented to ease the understanding.

If you find some bug, please e-mail me =)

'''

import numpy as np 
import tensorflow as tf
import sys

# Function to compute the sigmoid
def sigmoid (v):
    return 1/(1+np.exp(-v))

# Function to compute the mean square error    
def mse (x,y):
    return np.power((x-y),2).mean(axis=1).mean()

class RBM_TF:
    # Atributes
    dataIn = None    # A numpy m x n array, which m = samples and n = attributes
    numHid = None    # The neuron's number on hidden layer. Default = 20
    numVis = None    # The neuron's number on visible layer.     
    visBias = None   # Visible bias
    hidBias = None   # Hidden bias
    weights = None   # Weights' connection                     
    nSamples = None  # The samples' number on dataset
    dataIn = None    # The input data
    rbmType = None   # You can choose between either Gaussian-Bernoulli (GBRBM) RBM or 
                     # Bernoulli-Bernoulli RBM (BBRBM). For more information about , check [1].
                     # Default: GBRBM
    
   # The constructor method
    def __init__ (self, dataIn, numHid=20, weights=None, visBias=None, hidBias=None, rbmType='GBRBM', sess=None):
        self.dataIn = dataIn#tf.convert_to_tensor(dataIn,dtype=tf.float32)
        self.numHid = numHid       
        
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        
        # The attributes number is equal the visible neurons number
        self.nSamples, self.numVis = dataIn.shape       
        
        # If the weights and bias are set, we don't initialize them and use the values passed
        if weights is not None:
            self.weights = tf.convert_to_tensor(weights,dtype=tf.float32, name='weightsInit')
            self.visBias = tf.convert_to_tensor(visBias, dtype=tf.float32, name='visBiasInit')
            self.hidBias = tf.convert_to_tensor(hidBias,dtype=tf.float32, name='hidBiasInit')
        else:
            # initializing the bias and weights
            self.visBias = tf.Variable(tf.zeros(shape=[1,self.numVis], dtype=tf.float32), name='visBias')
            self.hidBias = tf.Variable(tf.zeros(shape=[1,self.numHid], dtype=tf.float32), name='hidBias')
            self.weights = tf.Variable (tf.random_normal(shape=[self.numVis,self.numHid], mean=0.0, stddev=0.1, dtype=tf.float32), name='weights')
            
        # Just checking the rbms names:
        if rbmType == 'GBRBM' or rbmType == 'BBRBM':        
            self.rbmType = rbmType
        else:
            print 'ERROR: this <%s> type does not exist in this code.' % rbmType
            raise Exception('RBM type error')              
        
        self.sess.run(tf.global_variables_initializer())   
       
            
    # This destructor is used just for close de tf.session()
    def __del__(self):
        self.sess.close()
        tf.reset_default_graph()
    
    def train (self, maxIter=200, lr=0.001, wc=0.0002, iMom=0.5, fMom=0.9, cdIter=1, batchSize=100, verbose=True, freqPrint=10, tol=10e-5):
        '''
        This method is called when we'd like to train the RBM. If you alredy have the weights,
        you just need to pass them as a parameter on the constructor method
        Input:
            maxIter: The iteration's max number. Default = 500    
            lr: learning rate. Default = 0.001 
            wc: Weight cost. Default = 0.0002
            iMom: Initial momentum. Default = 0.5
            fMom: Final momentum. Default = 0.9
            cdIter: The CD iterations' number. Default = 1
            batchSize: The mini-batche's size. Default= 100. If it's equal zero, then we use the whole dataset without batches       
            Verbose: If you'd like to print the training error, set True. Default = False
            freqPrint: This var control the print frequence. Every freqPrint iteration, the error will be printed. Default: 10
            tol: The convergence tolerance error for the weights. Default: 10e-5	
        '''
        print 'Starting the training...'
        # Checkin the batch size. If it's 0, we use the whole dataset        
        if batchSize == 0:
            batchSize = self.nSamples              
        else:
            batchSize = batchSize              
            
        self.batchSize = batchSize
        # Initializing the deltas matrix. They will aux us to update the weights and bias
        deltaWeights = tf.Variable(tf.zeros_like(self.weights, dtype=tf.float32), name='deltaWeights')       
        deltaVisBias = tf.Variable(tf.zeros_like(self.visBias, dtype=tf.float32), name='deltaVisBias')
        deltaHidBias = tf.Variable(tf.zeros_like (self.hidBias, dtype=tf.float32), name='deltaHidBias')           
        prevWeights = tf.Variable(tf.zeros_like(self.weights, dtype=tf.float32), name='prevWeights')
        attPrevWeights = prevWeights.assign(self.weights)
        
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(attPrevWeights)
        
        
        # These tensors will store layers
        visTensor = tf.placeholder(dtype=tf.float32, name='visTensor')     
        vis0Tensor = tf.placeholder(dtype=tf.float32, name='vis0Tensor')
        visNTensor = tf.placeholder(dtype=tf.float32, name='visNTensor')     
        hidTensor = tf.placeholder(dtype=tf.float32, name='hidTensor')     
        hid0Tensor = tf.placeholder(dtype=tf.float32, name='hidTensor')
        hidNTensor = tf.placeholder(dtype=tf.float32, name='hidNTensor')     
        momTensor = tf.placeholder(dtype=tf.float32, name='momentum')
        
        # Computing the probabilities for the hidden layer
        probHid = tf.sigmoid(tf.matmul(visTensor,self.weights) + self.hidBias)             
        # Sampling the hidden states
        statHid = tf.cast(tf.greater(hidTensor, tf.random_uniform([1,self.numHid], minval=0, maxval=1, dtype=tf.float32)),dtype=tf.float32)
        
        if self.rbmType == 'GBRBM':
            # reconstruction the visible layer            
            # WHITOUT NORMAL
            vis = tf.matmul(statHid, self.weights, False, True) + self.visBias               
            # WITH NORMAL
            #visN = np.random.randn(visN.shape[0], visN.shape[1]) + visN
        else:            
            # reconstruction the visible layer
            vis = tf.sigmoid(tf.matmul(statHid, self.weights, False, True) + self.visBias)                    
       
           
        # Computing the value to update rules
        dw = tf.matmul(vis0Tensor,hid0Tensor,True) - tf.matmul(visNTensor,hidNTensor,True)                
        dv = tf.reduce_sum(vis0Tensor, axis=0) - tf.reduce_sum(visNTensor, axis=0)
        #dv = dv[np.newaxis,:] # Just ajusting the numpy format
        dh = tf.reduce_sum(hid0Tensor, axis=0) - tf.reduce_sum(hidNTensor, axis=0)
        #dh = dh[np.newaxis,:] # Just ajusting the numpy format                           
        
               
        
        attDeltaWeights = deltaWeights.assign((momTensor*deltaWeights) + (lr*dw/batchSize) - (wc*self.weights))
        attDeltaVisBias = deltaVisBias.assign((momTensor*deltaVisBias) + (lr*dv/batchSize))
        attDeltaHidBias = deltaHidBias.assign((momTensor*deltaHidBias) + (lr*dh/batchSize))
        
        # Updating the weights and bias
        attWeights = self.weights.assign_add(deltaWeights)
        attVisBias = self.visBias.assign_add(deltaVisBias)
        attHidBias = self.hidBias.assign_add(deltaHidBias)
        
        for it in range(maxIter):
            error = 0
            # Setting the momentum
            if it < 5:
                mom = iMom
            else:
                mom = fMom
                
            # Shuffling the data. If you've already done it, you can comment the next line
            # and uncomment the next one
            sData = np.random.permutation(self.dataIn)
            #sData = self.dataIn                       
            
            # Starting the training for the mini-batches          
            for batch in range (0, self.nSamples, batchSize):                
                if batch + batchSize > self.nSamples:
                    break
                
                # Setting the input data on visible layers to start the Gibbs sampling
                # Starting the contrastive divergence
                vis0Np = sData[batch : batch+batchSize]        

                # Computing the first hidden neurons state
                probHid0Np = self.sess.run(probHid, feed_dict={visTensor: vis0Np})                                
                lastStatHidNp = self.sess.run(statHid, feed_dict={hidTensor: probHid0Np}) 
                
                # Starting the Gibbs sampling
                for n in range(cdIter):               
                    lastVisNp = self.sess.run(vis, feed_dict={statHid: lastStatHidNp})
                    lastProbHidNp = self.sess.run(probHid, feed_dict={visTensor: lastVisNp})
                    lastStatHidNp = self.sess.run(statHid, feed_dict={hidTensor: lastProbHidNp})
                    
                
                # updating the weights and bias
                self.sess.run([attDeltaWeights,attDeltaVisBias,attDeltaHidBias], feed_dict={vis0Tensor: vis0Np, hid0Tensor: probHid0Np, visNTensor: lastVisNp, hidNTensor: lastProbHidNp,  momTensor: mom})                             
                
                self.sess.run([attWeights, attVisBias, attHidBias])
                                  
         
            
            # Reconstruction error:
            error = mse(vis0Np, lastVisNp)
            # The weights difference
            diff = mse(self.sess.run(self.weights),self.sess.run(prevWeights))
            # Updating the previous weight
            self.sess.run(attPrevWeights) 
           
            if verbose == True and it % freqPrint == 0:
                print 'Reconstruction error: ', error, 'Iter: ', it, '  Diff: ', diff                
                
            if it > 300 and diff <= tol:
                break
        
                
    # This method return a reconstruction of a given data. To do this, the RBM needs to be
    # trained. It's just a step of Gibbs sampling
    def getReconstruction (self,data):        
        vis = tf.convert_to_tensor(data, dtype=tf.float32, name='visRecs')
        probHid = tf.sigmoid(tf.matmul(vis,self.weights)+self.hidBias)
        statHid = tf.cast(tf.greater(probHid, tf.random_uniform([1,self.numHid], minval=0, maxval=1, dtype=tf.float32)),dtype=tf.float32)
        
        if self.rbmType == 'GBRBM':            
            recons = tf.matmul(statHid,self.weights, False, True)+self.visBias
        else:
            recons = tf.sigmoid(tf.matmul(statHid, self.weights, False, True) + self.visBias) 
        return self.sess.run(recons)
     
    # This method returns the extracted features of a given data. To do this, the RBM needs to be
    # trained. It's just a step of Gibbs sampling
    def getFeatures (self, data):        
        vis = tf.convert_to_tensor(data, dtype=tf.float32, name='visRecs')
        probHid = tf.sigmoid(tf.matmul(vis,self.weights)+self.hidBias)
        statHid = tf.cast(tf.greater(probHid, tf.random_uniform([1,self.numHid], minval=0, maxval=1, dtype=tf.float32)),dtype=tf.float32)
        return self.sess.run(probHid), self.sess.run(statHid)   
        
    # This method save the weights and bias in a .csv file    
    def saveWeightsAndBias (self, nameTrain='T1'):
        np.savetxt('weights'+nameTrain+'.csv', self.sess.run(self.weights))
        np.savetxt('visBias'+nameTrain+'.csv', self.sess.run(self.visBias))
        np.savetxt('hidBias'+nameTrain+'.csv', self.sess.run(self.hidBias))
        
    def getWeights (self):
        return self.sess.run(self.weights)
    
    def getHidBias (self):
        return self.sess.run(self.hidBias)
    
    def getVisBias (self):
        return self.sess.run(self.visBias)
        

