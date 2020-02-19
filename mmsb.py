import numpy.random as npr
import numpy as np
from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt


ob = np.load('syn_train.npy')
ob = np.copy(ob[0,:,:,0])
ob = ob.astype(int)
f =2 #feature
num_item = np.shape(ob)[1]
sample_iter = 200 #sample iteration 
burn_iter = 100 #burn iteration
group_eta = 0.2 #group_prior
score_eta = 1 #score_eta
score_num = 2

item_mat = np.zeros((num_item,f))
item_mat_hist = np.zeros((num_item,f))
z_ij = np.zeros((num_item,num_item,2),dtype = int)-1
n_ab = np.zeros((f,f,2))

'''
hist
'''
item_mat_hist = np.zeros((sample_iter,num_item,f))
n_ab_hist = np.zeros((sample_iter,f,f,2))
'''
initial
'''


x,y = np.where(ob>-1)
s = npr.randint(f,size = len(x))
z_ij[x,y,0] = np.copy(s) # sender
s = npr.randint(f,size = len(x))
z_ij[x,y,1] = np.copy(s) # receiver


for i in range(len(x)):
    a = z_ij[x[i],y[i],0]
    b = z_ij[x[i],y[i],1]
    o = ob[x[i],y[i]]
    n_ab[a,b,o]= n_ab[a,b,o]+1
    item_mat[x[i],a] = item_mat[x[i],a]+1
    item_mat[y[i],b] = item_mat[y[i],b]+1
print(len(x))
print(np.sum(n_ab))



'''
sampling
'''
for t in range(sample_iter):
    for i in range(num_item):
        x = np.where(z_ij[i,:,1]>-1)[0]
        for j in range(len(x)):
            a = z_ij[i,x[j],0]
            b = z_ij[i,x[j],1]
            y = ob[i,x[j]]
            item_mat[i,a] = item_mat[i,a]-1
            n_ab[a,b,y] = n_ab[a,b,y]-1
            it_mat = item_mat[i,:]+group_eta
            p_ab = n_ab[:,b,:]+ score_eta
            p_ab = np.sum(p_ab,1)
            p_ab = np.repeat(p_ab[:,np.newaxis],2,1)
            p_ab = (n_ab[:,b,:]+ score_eta)/p_ab
            p_ab = p_ab[:,y]
            
            s = it_mat*p_ab
            s = s/np.sum(s)
            s = np.reshape(s, f)
            mult = npr.multinomial(1,s)
            index_group = np.where(mult==1)[0]
            
            z_ij[i,x[j],0] = index_group
            n_ab[index_group,b,y] = n_ab[index_group,b,y]+1
            item_mat_hist[t,i,index_group] = item_mat_hist[t,i,index_group]+1
            item_mat[i,index_group] = item_mat[i,index_group]+1
        
        x = np.where(z_ij[:,i,0]>-1)[0]

        for j in range(len(x)):
            a = z_ij[x[j],i,0]
            b = z_ij[x[j],i,1]
            y = ob[x[j],i]
            item_mat[i,b] = item_mat[i,b]-1
            n_ab[a,b,y] = n_ab[a,b,y]-1
            it_mat = np.copy(item_mat[i,:])+group_eta
            
            p_ab = n_ab[a,:,:]+ score_eta
            p_ab = np.sum(p_ab,1)
            p_ab = np.repeat(p_ab[:,np.newaxis],2,1)
            p_ab = (n_ab[a,:,:]+ score_eta)/p_ab
            p_ab = p_ab[:,y]
            
            s = it_mat*p_ab
            s = s/np.sum(s)
            s = np.reshape(s, f)
            mult = npr.multinomial(1,s)
            index_group = np.where(mult==1)[0]
            
            z_ij[x[j],i,1] = index_group
            n_ab[a,index_group,y] = n_ab[a,index_group,y]+1
            item_mat_hist[t,i,index_group] = item_mat_hist[t,i,index_group]+1
            item_mat[i,index_group] = item_mat[i,index_group]+1

    n_ab_hist[t,:,:,:] = np.copy(n_ab)
                
ob_real = np.load('syn.npy') 
ob_real = np.copy(ob_real[:,:,1])
'''
log likelihood
''' 
'''
x,y = np.where(ob>-1)
llh = np.zeros(sample_iter)  
for t in range(sample_iter):
    it = np.copy(item_mat_hist[t,:,:]+group_eta)
    s = np.sum(it,1)
    s = np.repeat(s[:,np.newaxis], f, 1)
    it = (it)/s
    
    ab = np.copy(n_ab_hist[t,:,:,:])
    s = np.sum(ab+score_eta,2)
    s = np.repeat(s[:,:,np.newaxis], score_num, 2)
    ab = (ab+score_eta)/s
    
    s = 0
    for i in range(len(x)):
        single_score = np.dot(np.dot(it[x[i],:],ab[:,:,ob[x[i],y[i]]]),np.transpose(it[y[i],:]))
        s = s+np.log(single_score)
    llh[t] = np.copy(s)
    
print(llh)
   
x = np.arange(sample_iter)
plt.plot(x,llh)
plt.show()
'''
         
'''
auc
'''        
re = np.zeros((num_item,num_item))
print(burn_iter,sample_iter)
for t in range(burn_iter,sample_iter):
    it = item_mat_hist[t,:,:]
    it = it+group_eta
    
    ab = n_ab_hist[t,:,:,:]
    ab = ab + score_eta
    s = np.sum(ab,2)
    s = np.repeat(s[:,:,np.newaxis], score_num, 2)
    ab = ab/s
    ab = ab[:,:,1]
    re = re +np.dot(np.dot(it,ab),np.transpose(it))
    
x,y = np.where(ob==-1)
our = np.copy(re[x,y])
cor = np.copy(ob_real[x,y])
print('test auc',roc_auc_score(cor, our))
x,y = np.where(ob>-1)
our = np.copy(re[x,y])
cor = np.copy(ob_real[x,y])
print('train auc',roc_auc_score(cor, our))
    
        
        
        
        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
            








