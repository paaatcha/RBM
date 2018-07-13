# -*- coding: utf-8 -*-
'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com

This class implements the restricted Boltzmann machine (RBM) according to
Hinton's guideline: 

[1] Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." 
Momentum, v. 9, n. 1, p. 926, 2010.

You can chosse either Gaussian-Bernoulli RBM or Bernoulli-Bernoulli RBM. All the code is 
very commented to ease the understanding.

If you find some bug, please e-mail me =)

'''

import numpy as np
import sys

# Insert the util's path
#sys.path.insert (0, '/home/patcha/Dropbox/Doutorado/Codigos/Python/utils/')
sys.path.insert (0, '/home/labcin/Dropbox/Codigos/Python/utils/')

from utilsClassification import sigmoid, mse

class RBM:
    # Atributes
    dataIn = None    # A numpy m x n array, which m = samples and n = attributes. If it's None
                     # you need to set the weights and bias to execute the net. Default = None
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
    def __init__ (self, dataIn=None, numHid=20, numVis=None, weights=None, visBias=None, hidBias=None, rbmType='GBRBM'):

        
        # If the weights and bias are set, we don't initialize them and use the values passed
        if weights is not None:
            self.weights = weights
            self.visBias = visBias
            self.hidBias = hidBias
        else:        
            self.dataIn = dataIn
            self.numHid = numHid       
        
            # The attributes number is equal the visible neurons number
            if self.dataIn == None:
                self.nSamples, self.numVis = None, numVis
            else:
                self.nSamples, self.numVis = dataIn.shape
            
            # initializing the bias and weights
            self.visBias = np.zeros([1,self.numVis])
            self.hidBias = np.zeros([1,self.numHid])
            self.weights = np.random.randn (self.numVis,self.numHid)*0.1
#            self.weights = np.random.uniform(-1,1,[self.numVis,self.numHid])
            
            
        self.deltaWeights = np.zeros_like(self.weights)        
        self.deltaVisBias = np.zeros_like(self.visBias)
        self.deltaHidBias = np.zeros_like (self.hidBias)   
                
        # Just checking the rbms names:
        if rbmType == 'GBRBM' or rbmType == 'BBRBM':        
            self.rbmType = rbmType
        else:
            print 'ERROR: this <%s> type does not exist in this code.' % rbmType
            raise Exception('RBM type error')       
            
        
        
    # Just a method for print the rbm's atributtes
    def __str__(self):
        return '### RBM ### \n(numVis, numHid) = (%s, %s)\n(nSamples) = (%s)\n(Weights, visBias, hidBias) = (%s, %s, %s)\n RBM type: %s\n# # #' % (self.numVis, \
         self.numHid, self.nSamples, self.weights.shape, self.visBias.shape, self.hidBias.shape, self.rbmType)
        
        
    # This method is called when we'd like to train the RBM. If you alredy have the weights,
    # you just need to pass them as a parameter on the constructor method
    # Parameters:
    # maxIter:   # The iteration's max number. Default = 500
    
    # lr:        # learning rate. Default = 0.001 
    # wc:        # Weight cost. Default = 0.0002
    # iMom:      # Initial momentum. Default = 0.5
    # fMom:      # Final momentum. Default = 0.9
    # cdIter:    # The CD iterations' number. Default = 1
    # batchSize: # The mini-batche's size. Default= 100. If it's equal zero, then 
    #              we use the whole dataset without batches       
    # Verbose:   # If you'd like to print the training error, set True. Default = False
    # freqPrint: # This var control the print frequence. Every freqPrint iteration, the
                 # error will be printed. Default: 10
    # tol        # The convergence tolerance error for the weights. Default: 10e-5
    def train (self, maxIter=200, lr=0.001, wc=0.0002, iMom=0.5, fMom=0.9, cdIter=1, batchSize=100, verbose=True, freqPrint=10, tol=10e-5):
        print 'Starting the training...'
        # Checkin the batch size. If it's 0, we use the whole dataset
        if batchSize == 0:
            batchSize = self.nSamples
        else:
            batchSize = batchSize    
    
        self.batchSize = batchSize
        # Initializing the deltas matrix. They will aux us to update the weights and bias
        deltaWeights = np.zeros_like(self.weights)        
        deltaVisBias = np.zeros_like(self.visBias)
        deltaHidBias = np.zeros_like (self.hidBias)   
        prevWeights = self.weights
        
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
            nb = 0
            for batch in range (0, self.nSamples, batchSize):                
                if batch + batchSize > self.nSamples:
                    break
                
                # Setting the input data on visible layers to start the Gibbs sampling
                # Starting the contrastive divergence
                vis0 = sData[batch : batch+batchSize]
         
                # Computing the probabilities for the hidden layer
                probHid0 = sigmoid(np.dot(vis0,self.weights)+self.hidBias)                 
                
                # Sampling the hidden states
                statHid = probHid0 > np.random.rand(1,self.numHid)         
                
                # Starting the Gibbs sampling
                for n in range(cdIter):
                    # reconstruction the visible layer                    
                    if self.rbmType == 'GBRBM':
                        # WHITOUT NORMAL
                        vis1 = np.dot(statHid, self.weights.T) + self.visBias                 
                    
                        # WITH NORMAL
                        #vis1 = np.random.randn(vis1.shape[0], vis1.shape[1]) + vis1
                    else:
                        vis1 = sigmoid(np.dot(statHid, self.weights.T) + self.visBias)                    

                    # Computing the probabilities for the hidden layer
                    probHid1 = sigmoid(np.dot(vis1,self.weights)+self.hidBias)
                    
                    # Sampling the hidden states
                    statHid = probHid1 > np.random.rand(1,self.numHid)                
                
                # Computing the value to update rules
                dw = np.dot(vis0.T,probHid0) - np.dot(vis1.T,probHid1)                
                dv = np.sum(vis0, axis=0) - np.sum(vis1, axis=0)
                dv = dv[np.newaxis,:] # Just ajusting the numpy format
                dh = np.sum(probHid0, axis=0) - np.sum(probHid1, axis=0)
                dh = dh[np.newaxis,:] # Just ajusting the numpy format     


                # TODO: REVER ESSA QUESTÃO DOS FORS
                deltaWeights = (mom*deltaWeights) + (lr*dw/batchSize) - (wc*self.weights)
                deltaVisBias = (mom*deltaVisBias) + (lr*dv/batchSize)
                deltaHidBias = (mom*deltaHidBias) + (lr*dh/batchSize)                    
            
            # Updating the weights and bias
            self.weights = self.weights + deltaWeights
            self.visBias = self.visBias + deltaVisBias
            self.hidBias = self.hidBias + deltaHidBias
            
            # Batch error:
            #error = error + np.sum( np.sum( np.power(vis0 - vis1,2) ) )
            
            error = error + mse(vis1,vis0)
            nb = nb + 1
           
            error = error/nb
            # Reconstruction error:
            diff = mse(self.weights,prevWeights)
            prevWeights = self.weights         
           
            if verbose == True and it % freqPrint == 0:
                print 'Reconstruction error: ', error, 'Iter: ', it, '  Diff: ', diff                
                
            if it > 300 and diff <= tol:
                break
        
    def train_batch (self, X, lr=0.001, wc=0.0002, mom=0.9, cdIter=1, batchSize=100, verbose=True, tol=10e-5):

        self.batchSize = batchSize
        prevWeights = self.weights       
        
        # Starting the contrastive divergence
        vis0 = X
 
        # Computing the probabilities for the hidden layer
        probHid0 = sigmoid(np.dot(vis0,self.weights)+self.hidBias)                 
        
        # Sampling the hidden states
        statHid = probHid0 > np.random.rand(1,self.numHid)         
        
        # Starting the Gibbs sampling
        for n in range(cdIter):
            # reconstruction the visible layer                    
            if self.rbmType == 'GBRBM':
                # WHITOUT NORMAL
                vis1 = np.dot(statHid, self.weights.T) + self.visBias                 
            
                # WITH NORMAL
                #vis1 = np.random.randn(vis1.shape[0], vis1.shape[1]) + vis1
            else:
                vis1 = sigmoid(np.dot(statHid, self.weights.T) + self.visBias)                    

            # Computing the probabilities for the hidden layer
            probHid1 = sigmoid(np.dot(vis1,self.weights)+self.hidBias)
            
            # Sampling the hidden states
            statHid = probHid1 > np.random.rand(1,self.numHid)                
        
        # Computing the value to update rules
        dw = np.dot(vis0.T,probHid0) - np.dot(vis1.T,probHid1)                
        dv = np.sum(vis0, axis=0) - np.sum(vis1, axis=0)
        dv = dv[np.newaxis,:] # Just ajusting the numpy format
        dh = np.sum(probHid0, axis=0) - np.sum(probHid1, axis=0)
        dh = dh[np.newaxis,:] # Just ajusting the numpy format     

        # TODO: REVER ESSA QUESTÃO DOS FORS
        self.deltaWeights = (mom*self.deltaWeights) + (lr*dw/batchSize) - (wc*self.weights)
        self.deltaVisBias = (mom*self.deltaVisBias) + (lr*dv/batchSize)
        self.deltaHidBias = (mom*self.deltaHidBias) + (lr*dh/batchSize)                    
        
        # Updating the weights and bias
        self.weights = self.weights + self.deltaWeights
        self.visBias = self.visBias + self.deltaVisBias
        self.hidBias = self.hidBias + self.deltaHidBias        
                
        # Reconstruction error:
        error = mse(vis1,vis0)       
        
        # Weights tolerance
        diff = mse(self.weights,prevWeights)        
       
        if verbose == True:
            print 'RBM reconstruction error: ', error, '  W diff: ', diff                

                
    # This method return a reconstruction of a given data. To do this, the RBM needs to be
    # trained. It's just a step of Gibbs sampling
    def getReconstruction (self,data):
        probHid0 = sigmoid(np.dot(data,self.weights)+self.hidBias)
        statHid = probHid0 > np.random.rand(1,self.numHid)
        if self.rbmType == 'GBRBM':            
            recons = np.dot(statHid,self.weights.T)+self.visBias
        else:
            recons = sigmoid(np.dot(statHid, self.weights.T) + self.visBias) 
        return recons
     
    # This method returns the extracted features of a given data. To do this, the RBM needs to be
    # trained. It's just a step of Gibbs sampling
    def getFeatures (self, data):
        probHid0 = sigmoid(np.dot(data,self.weights)+self.hidBias)
        statHid = probHid0 > np.random.rand(1,self.numHid)
        return probHid0, statHid   
        
    # This method save the weights and bias in a .csv file    
    def saveWeightsAndBias (self, nameTrain='T1'):
        np.savetxt('weights'+nameTrain+'.csv', self.weights)
        np.savetxt('visBias'+nameTrain+'.csv', self.visBias)
        np.savetxt('hidBias'+nameTrain+'.csv', self.hidBias)       
        
    def getWeights (self):
        return self.weights
    
    def getHidBias(self):
        return self.hidBias
        
    def getInputWeights (self):
        return np.concatenate ((self.weights, self.hidBias), axis = 0)

