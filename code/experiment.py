'''
Use this file to answer question 5.1 and 5.2
'''
from crf import CRF
import numpy as np
import time

CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
         'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
         'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
         'y', 'z']

# crf = []
# crf_id = 0

def num_correct(predictions,y):
    count = 0
    for sample in range(len(y)):
        if y[sample] == predictions[sample]:
            count +=1
    return count

def to_list(y):
    y = y.replace("\n", "")
    return [i for i in y]

def five_one():
    '''implement your experiments for question 5.1 here'''
    file=open('Q5_1.txt','w')
    crf = []
    crf_id = 0
    for Le in  [100,200,300,400,500,600,700,800]:
        prediction_error = np.zeros([Le])
        crf.append(CRF(L=CHARS, F=321))
        
        Y = [to_list(i) for i in open("../data/train_words.txt")][0:Le]
        X = [np.loadtxt("../data/train_img{}.txt".format(i)) for i in range(1,len(Y) + 1)][0:Le]

        # print "predictions before training"
        # for x,y in zip(X[0:Le],Y[0:Le]):
        #     print "preds", crf.predict(x), y
        #     a,b = crf.predict_logprob(x)

        # Start training

        t0 = time.time()
        print "training crf ", crf_id, ' with ', Le, ' images' 
        crf[crf_id].fit(Y=Y,X=X)
        t1 = time.time()

        # Store training statistics and learnt parameters

        W_F, W_T = crf[crf_id].get_params()
        print('Training complete. Time to learn ', Le,'images is ', t1-t0, 'seconds')
        
        file = open('learnt_model_{}.txt'.format(crf_id),'w')
        file.write("\ntraining crf "+str(crf_id)+' with '+str(Le)+' images '+'\n')
        file.write('W_F: '+str(W_F)+'\n'+'W_T: '+str(W_T) )
        file.write('\nTraining complete. Time to learn '+str(Le)+' images was '+ str(t1-t0)+' seconds')
        file.close()
        file = open('W_T_{}.npy'.format(crf_id),'w')
        np.save(file,W_T)
        file.close()
        file = open('W_F_{}.npy'.format(crf_id),'w')
        np.save(file,W_F)
        file.close()
            
        # Start testing
        W_F = np.load('W_F_{}.npy'.format(crf_id),'r')
        W_T = np.load('W_T_{}.npy'.format(crf_id),'r')
        crf[crf_id].set_params(W_F,W_T)


        Y = [to_list(i) for i in open("../data/test_words.txt")]
        X = [np.loadtxt("../data/test_img{}.txt".format(i)) for i in range(1,len(Y) + 1)]

        print "predictions after training"
        
        correct = 0.0
        total = 0.0

        for x,y in zip(X,Y):
            predictions = crf[crf_id].predict(x)
            #print "preds", predictions, y
            correct += num_correct(predictions, y)
            total += len(y)
        accuracy = float(correct/total)
        
        cond_log_likelihoods = crf[crf_id].log_likelihood(Y,X)
        
        print ('Accuracy:', accuracy, 'Correct', correct, 'Total', total)
        print ('Log Likelihood', cond_log_likelihoods)
        file.write(str(crf_id)+','+str(accuracy)+','+str(cond_log_likelihoods)+'\n')
        
        crf_id += 1
    file.close()
    
    pass


def five_two():
    '''implement your experiments for question 5.2 here'''
    
    file=open('Q5_2.txt','w')
    crf_test = CRF(L=CHARS, F=321)
    W_F = np.load('W_F_{}.npy'.format(7),'r')
    W_T = np.load('W_T_{}.npy'.format(7),'r')
    crf_test.set_params(W_F,W_T)

    Y_gen = []
    X_gen = []
    samples_per_length = 50

    for length in range(1,21):
        Y_gen.append(np.random.choice(CHARS,[samples_per_length,length]))
        X_gen.append(np.random.randint(2,size=(samples_per_length,length,321)))
        t0 = time.time()
        for x,y in zip(X_gen[length-1],Y_gen[length-1]):
            predictions = crf_test.predict(x)
        t1 = time.time()
        print ('Average time to predict ', samples_per_length, 'samples of length ', length,'is', (t1-t0)/samples_per_length)
        file.write(str(length)+','+str((t1-t0)/samples_per_length)+'\n')
    file.close()
    
    pass
five_one()
five_two()