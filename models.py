# Building model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep, global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.optim.lr_scheduler import MultiStepLR



class GCNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro= 1024, output_dim=128, dropout=0.2):
        super(GCNN, self).__init__()

        print('GCNN Loaded')

        # for protein 1
        self.n_output = n_output
        self.pro1_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro1_fc1 = nn.Linear(num_features_pro, output_dim)

        # for protein 2
        self.pro2_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro2_fc1 = nn.Linear(num_features_pro, output_dim)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256 ,64)
        self.out = nn.Linear(64, self.n_output)

    def forward(self, pro1_data, pro2_data):

        #get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch


        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        
	# global pooling
        x = gep(x, pro1_batch)   

        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)



        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(xt)

	# global pooling
        xt = gep(xt, pro2_batch)  

        # flatten
        xt = self.relu(self.pro2_fc1(xt))
        xt = self.dropout(xt)


	# Concatenation  
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out
        

net = GCNN()
print(net)

"""# GAT"""

class AttGNN(nn.Module):
    def __init__(self, n_output=1, num_features_pro= 1024, output_dim=128, dropout=0.2, heads = 1 ):
        super(AttGNN, self).__init__()

        print('AttGNN Loaded')

        self.hidden = 8
        self.heads = 1
        
        # for protein 1
        self.pro1_conv1 = GATConv(num_features_pro, self.hidden* 16, heads=self.heads, dropout=0.2)
        self.pro1_fc1 = nn.Linear(128, output_dim)


        # for protein 2
        self.pro2_conv1 = GATConv(num_features_pro, self.hidden*16, heads=self.heads, dropout=0.2)
        self.pro2_fc1 = nn.Linear(128, output_dim)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, n_output)
        


    def forward(self, pro1_data, pro2_data):

        # get graph input for protein 1 
        pro1_x, pro1_edge_index, pro1_batch = pro1_data.x, pro1_data.edge_index, pro1_data.batch
        # get graph input for protein 2
        pro2_x, pro2_edge_index, pro2_batch = pro2_data.x, pro2_data.edge_index, pro2_data.batch
         
        
        x = self.pro1_conv1(pro1_x, pro1_edge_index)
        x = self.relu(x)
        
	# global pooling
        x = gep(x, pro1_batch)  
       
        # flatten
        x = self.relu(self.pro1_fc1(x))
        x = self.dropout(x)



        xt = self.pro2_conv1(pro2_x, pro2_edge_index)
        xt = self.relu(self.pro2_fc1(xt))
	
	# global pooling
        xt = gep(xt, pro2_batch)  

        # flatten
        xt = self.relu(xt)
        xt = self.dropout(xt)

	
	# Concatenation
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out

net_GAT = AttGNN()
print(net_GAT)

"""# Residual Graph Isomorphism Network"""

class ResGIN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=1024, output_dim=128, num_layers=4, dropout=0.2):
        super(ResGIN, self).__init__()
        
        print('ResGIN Loaded')
        
        # Embedding layer for protein 1
        self.pro1_embedding = nn.Linear(num_features_pro, output_dim)
        
        # GIN layers with residual connections for protein 1
        self.pro1_gin_layers = nn.ModuleList()
        for i in range(num_layers):
            nn_layer1 = nn.Sequential(
                nn.Linear(output_dim, output_dim*2),
                nn.BatchNorm1d(output_dim*2),
                nn.ReLU(),
                nn.Linear(output_dim*2, output_dim)
            )
            self.pro1_gin_layers.append(GINConv(nn_layer1, eps=0.1, train_eps=True))
        
        # Embedding layer for protein 2
        self.pro2_embedding = nn.Linear(num_features_pro, output_dim)
        
        # GIN layers with residual connections for protein 2
        self.pro2_gin_layers = nn.ModuleList()
        for i in range(num_layers):
            nn_layer2 = nn.Sequential(
                nn.Linear(output_dim, output_dim*2),
                nn.BatchNorm1d(output_dim*2),
                nn.ReLU(),
                nn.Linear(output_dim*2, output_dim)
            )
            self.pro2_gin_layers.append(GINConv(nn_layer2, eps=0.1, train_eps=True))
        
        # Activation, dropout, and normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
        # Combined layers
        self.fc1 = nn.Linear(2 * output_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, n_output)
        
    def forward(self, pro1_data, pro2_data):
        # Process protein 1
        x1 = self._process_protein(pro1_data, self.pro1_embedding, self.pro1_gin_layers)
        
        # Process protein 2
        x2 = self._process_protein(pro2_data, self.pro2_embedding, self.pro2_gin_layers)
        
        # Concatenate protein representations
        xc = torch.cat([x1, x2], dim=1)
        
        # Final classification layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        
        return out
    
    def _process_protein(self, protein_data, embedding_layer, gin_layers):
        x, edge_index, batch = protein_data.x, protein_data.edge_index, protein_data.batch
        
        # Initial embedding
        x = embedding_layer(x)
        
        # Apply GIN layers with residual connections
        for gin in gin_layers:
            x_new = gin(x, edge_index)
            x = x + x_new  # Residual connection
            x = self.relu(x)
            x = self.dropout(x)
        
        # Global pooling to get graph-level representation
        x = gep(x, batch)
        
        return x

net_ResGIN = ResGIN()
print(net_ResGIN)

