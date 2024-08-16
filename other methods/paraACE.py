import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import re
from sklearn.metrics import accuracy_score, roc_auc_score
import itertools, pickle

class Initial_FC_Net(nn.Module):
    def __init__(
        self,
        X_num_features,
        Z_num_features,
        hidden_units,
        task_type = "regression"
    ):
        super(Initial_FC_Net, self).__init__()

        self.task_type = task_type
        self.interaction_mlp = create_mlp([X_num_features + Z_num_features] + hidden_units + [1])

    def forward(self, x_z):
        output_X_Z = self.interaction_mlp(x_z)
        if self.task_type == "regression":
            return output_X_Z
        elif self.task_type == "classification":
            return torch.sigmoid(output_X_Z)
            
def Initial_FC_train(
    net,
    data_loaders,
    nepochs=100,
    verbose=False,
    early_stopping=True,
    patience=5,
    device=torch.device("cpu"),
):

    task_type = net.task_type
    optimizer= torch.optim.Adam(net.parameters(), betas=(0.9, 0.99))
    criterion = nn.MSELoss(reduction="mean") if task_type == "regression" else nn.BCELoss(reduction="mean")

    def evaluate_loss(net, data_loader, criterion, device):
        losses = []
        for X_inputs, Z_inputs, targets in data_loader:
            X_inputs = X_inputs.to(device)
            Z_inputs = Z_inputs.to(device)
            targets = targets.to(device)
            X_Z_inputs = torch.cat((X_inputs, Z_inputs), dim=1).to(device)
            loss = criterion(net(X_Z_inputs), targets).cpu().data
            losses.append(loss)
        return torch.stack(losses).mean()

    def evaluate_accu(net, data_loader, device):
        accus = []
        for X_inputs, Z_inputs, targets in data_loader:
            X_inputs = X_inputs.to(device)
            Z_inputs = Z_inputs.to(device)
            targets = targets.to(device)
            X_Z_inputs = torch.cat((X_inputs, Z_inputs), dim=1).to(device)
            outputs = net(X_Z_inputs)
            accu = accuracy_score(targets.squeeze(1).detach().numpy(), outputs.squeeze(1).detach().numpy().round())
            accus.append(accu)
        return np.mean(accus)

    def evaluate_auc(net, data_loader, device):
        aucs = []
        for X_inputs, Z_inputs, targets in data_loader:
            X_inputs = X_inputs.to(device)
            Z_inputs = Z_inputs.to(device)
            targets = targets.to(device)
            X_Z_inputs = torch.cat((X_inputs, Z_inputs), dim=1).to(device)
            outputs = net(X_Z_inputs)
            auc = roc_auc_score(targets.squeeze(1).detach().cpu().numpy(), outputs.squeeze(1).detach().numpy())
            aucs.append(auc)
        return np.mean(aucs)

    best_loss = float("inf")
    best_net = None

    if "val" not in data_loaders:
        early_stopping = False

    patience_counter = 0

    if verbose:
        print("starting to train")
        if early_stopping:
            print("early stopping enabled")

    for epoch in range(nepochs):
        running_loss = 0.0
        run_count = 0
        for i, data in enumerate(data_loaders["train"], 0):
            X_inputs, Z_inputs, targets = data
            X_inputs = X_inputs.to(device)
            Z_inputs = Z_inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            X_Z_inputs = torch.cat((X_inputs, Z_inputs), dim=1)
            outputs = net(X_Z_inputs)
            loss = criterion(outputs, targets).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            run_count += 1

        if epoch % 1 == 0:
            key = "val" if "val" in data_loaders else "train"
            val_loss = evaluate_loss(net, data_loaders[key], criterion, device)

            if epoch % 2 == 0:
                if verbose:
                    print(
                        "[epoch %d, total %d] train loss: %.4f, val loss: %.4f"
                        % (epoch + 1, nepochs, running_loss / run_count, val_loss)
                    )
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_net = copy.deepcopy(net) 
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        net = best_net
                        val_loss = best_loss
                        if verbose:
                            print("early stopping!")
                        break # The training process will stop early if the validation loss has not improved for a certain number of epochs (determined by patience).

            prev_loss = running_loss
            running_loss = 0.0

    if "test" in data_loaders:
        key = "test"
    elif "val" in data_loaders:
        key = "val"
    else:
        key = "train"

    if task_type == "regression":
        test_loss = evaluate_loss(net, data_loaders[key], criterion, device).item()
        output = (net, test_loss)
        if verbose:
            print("Finished Training. Test loss: ", test_loss)
    elif task_type == "classification":
        test_loss = evaluate_loss(net, data_loaders[key], criterion, device).item()
        test_accu = evaluate_accu(net, data_loaders[key], device)
        test_auc = evaluate_auc(net, data_loaders[key], device)
        output = (net, test_loss, test_accu, test_auc)
        if verbose:
            print("Finished Training. Test loss: %.4f, Test accuracy: %.4f, Test auc: %.4f" % (test_loss, test_accu, test_auc))

    return output


def detect_XZ_Hessian_UCB(FCnet,X_train, Z_train, interactions_count, device, verbose):
    np.random.seed(0)
    def one_hot(i,p):
        batch_size=1
        # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
        y = torch.LongTensor([[i]])
        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(batch_size, p)
        # print(y_onehot)
        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        return y_onehot

    def evaluate_2nd_derivative(net,i,k,N):
        delta=1.5#np.random.rand()+0.5
        j=np.random.randint(N)
        xeval = XZ_train[j,:].reshape(1,p) #pick one sample
        xeval = torch.tensor(xeval, dtype=torch.float32).to(device)

        ### two side
        f0 = net(xeval-one_hot(i,p)*delta-one_hot(k,p)*delta)
        fi =net(xeval+one_hot(i,p)*delta-one_hot(k,p)*delta)
        fik =net(xeval+one_hot(i,p)*delta+one_hot(k,p)*delta)
        fk=net(xeval+one_hot(k,p)*delta-one_hot(i,p)*delta)

        inter_strength=(fik-fi-fk+f0)/(4*delta**2)
        reward=inter_strength.detach().numpy()**2 #abs()
        return -float(reward)

    print("start dectecting")
    Xp = X_train.shape[1]
    Zp = Z_train.shape[1]
    XZ_train = np.hstack((X_train, Z_train))
    N=XZ_train.shape[0]
    p=XZ_train.shape[1]
    # Larms: List of all pairs of features (combinations of two features).
    Larms=[]
    for i in itertools.combinations(range(p),2):
        Larms.append(i)

    # Create the list with corresponding X and Z labels
    XZ_Larms = []
    for i, j in Larms:
        i_label = f"X{i+1}" if i < Xp else f"Z{i+1-Xp}"
        j_label = f"X{j+1}" if j < Xp else f"Z{j+1-Xp}"
        XZ_Larms.append((i_label, j_label))

    n=len(Larms)
    Delta = 1.0/n
    step_size=1
    num_arms=1
    lcb = np.zeros(n, dtype='float')       #At any point, stores the mu - lower_confidence_interval
    ucb = np.zeros(n, dtype='float')       #At any point, stores the mu + lower_confidence_interval
    T = step_size*np.ones(n, dtype='int')#At any point, stores number of times each arm is pulled

    n_init_try=3
    init_data=np.zeros((n_init_try,n))
    for j in range(n_init_try):
        for i in range(n):
            init_data[j,i]=evaluate_2nd_derivative(FCnet,Larms[i][0],Larms[i][1],N)

    maxrecord=np.max(init_data,axis=0)
    minrecord=np.min(init_data,axis=0)
    sigma=(maxrecord-minrecord)/2


    estimate=np.mean(init_data,axis=0)
    lcb = estimate - np.sqrt(sigma**2*np.log(1/Delta)/step_size)
    ucb = estimate + np.sqrt(sigma**2*np.log(1/Delta)/step_size)
    if verbose:
        print("Initialization done! initial try"+str(n_init_try)+'times')
    #print(estimate,sigma,np.sqrt(sigma**2*np.log(1/Delta)/step_size))


    def choose_arm():
        n=100 # MAXPULL: maximum number of times an arm is allowed to be pulled before special handling.
        low_lcb_arms = np.argpartition(lcb,num_arms)[:num_arms] # Selects num_arms arms that have the lowest lower confidence bounds (LCB).
        # Among the arms with the lowest LCB, identify those that have been pulled more than n times and whose UCB is not equal to their LCB.
        arms_pulled_morethan_n = low_lcb_arms[ np.where( (T[low_lcb_arms]>=n) & (ucb[low_lcb_arms] != lcb[low_lcb_arms]) ) ]

        # For these over-pulled arms, reset their UCB and LCB to the estimate and return None
        if arms_pulled_morethan_n.shape[0]>0:
            # Compute the distance of these arms accurately
            #print('more_than_n')
            #estimate[arms_pulled_morethan_n] = evaluate_2nd_derivative_brute(net,Larms[int(arms_pulled_morethan_n)][0],Larms[int(arms_pulled_morethan_n)][1])
            #T[arms_pulled_morethan_n]  += n
            ucb[arms_pulled_morethan_n] = estimate[arms_pulled_morethan_n]
            lcb[arms_pulled_morethan_n] = estimate[arms_pulled_morethan_n]
            return None

        if ucb.min() <  lcb[np.argpartition(lcb,1)[1]]: #Exit condition
            return None

        # Arms that are eligible for pulling (based on the number of pulls) are returned for the next evaluation.
        arms_to_pull          = low_lcb_arms[ np.where(T[low_lcb_arms]<n) ]
        return arms_to_pull


    # updates the estimates and confidence bounds for a given arm based on the latest evaluation.
    def pull_arm(arms,N):
        Tmean = evaluate_2nd_derivative(FCnet,Larms[int(arms)][0],Larms[int(arms)][1],N)
        estimate[arms]   = (estimate[arms]*T[arms] + Tmean*step_size)/( T[arms] + step_size + 0.0 )
        T[arms]          = T[arms]+step_size
        lcb[arms]        = estimate[arms] - np.sqrt(sigma[arms]**2*np.log(1/Delta)/(T[arms]+0.0))
        ucb[arms]        = estimate[arms] + np.sqrt(sigma[arms]**2*np.log(1/Delta)/(T[arms]+0.0))
        maxrecord[arms]   = max(maxrecord[arms],estimate[arms])
        minrecord[arms]   = min(minrecord[arms],estimate[arms])
        sigma[arms]       = (maxrecord[arms]-minrecord[arms])/2



    chosen_arms=[] # List to store the indices of the chosen feature pairs (arms).
    record_chosen_arms=[] # List to store detailed records of the chosen arms for resetting later.
    k,K=1,interactions_count #set pick interactions
    cnt=0

    while k <=K :
        arms_to_pull=choose_arm()
        while arms_to_pull==None:

            chosen_arm_pos=np.argmin(ucb)
            chosen_arms.append(chosen_arm_pos) # add the chosen arm to K best arms
            # record the(position, ucb, mean, lcb,T)
            record_chosen_arms.append((chosen_arm_pos,
                                       ucb[chosen_arm_pos],
                                       estimate[chosen_arm_pos],
                                      lcb[chosen_arm_pos],
                                       T[chosen_arm_pos]
                                      ))
            if verbose:
                print('chosen arm:',chosen_arm_pos,'strength:',-estimate[chosen_arm_pos], 'iteration:',cnt)

            # set the ucb mean lcb to be a large number, so this arm won't be pulled
            ucb[chosen_arm_pos],estimate[chosen_arm_pos],lcb[chosen_arm_pos]=0.5,0.5,0.5
            arms_to_pull=choose_arm()

            k=k+1
        #print(arms_to_pull)
        pull_arm(arms_to_pull,N)
        cnt=cnt+1

    #reset values
    for i in record_chosen_arms:
        ucb[i[0]]=i[1]
        estimate[i[0]]=i[2]
        lcb[i[0]]=i[3]

    interaction_strength=[]
    for i in range(len(chosen_arms)):
        interaction_strength.append((XZ_Larms[chosen_arms[i]], np.abs(record_chosen_arms[i][2]))) #selected

    return interaction_strength, record_chosen_arms
