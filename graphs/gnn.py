import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm

import rdkit
import rdkit.Chem as Chem

class GCN(torch.nn.Module):
    '''Implementation of a graph convolutional neural network.'''
    def __init__(self, in_feats: int = 130, n_hidden: int = 100, num_conv_layers: int = 2, lr: float = 1e-3,
                 epochs: int = 50, n_out: int = 1, n_layers: int = 2, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):
        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay
        self.atom_embedding = torch.nn.Linear(in_feats, n_hidden)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GCNConv(n_hidden, n_hidden))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))
        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))
        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.elu(self.atom_embedding(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)
        x = self.out(x)
        return x
    
    def embedding(self, x, edge_index, batch):
        x = F.elu(self.atom_embedding(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        return x
    
    def penultimate(self, x, edge_index, batch):
        x = F.elu(self.atom_embedding(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)
        return x

class GNN:
    '''
        Wrapper around a PyTorch-Geometric GCN implentation suitable
        for active learning. All methods require dataloaders for training, 
        validation, and test sets.
    '''
    def __init__(self, n_models=4, uncertainty=False, embedding='embedding', in_feats=100):
        self.name = 'gnn'
        self.model = None
        self.n_models = n_models
        self.uncertainty = uncertainty
        self.embedding = embedding
        self.in_feats = in_feats
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _train_gcn(self, model, train_loader, val_loader):
        '''Executes training loop on GCN models.'''
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
        criterion = nn.MSELoss()
        train_losses = []
        val_losses = []
        print(f'Training on device: {self.device}.')
        for epoch in range(1000):
            model.train()
            train_loss = 0.0
            for batch_idx, datum in enumerate(train_loader):
                x = datum.x.to(self.device)
                edge_idx = datum.edge_index.to(self.device)
                batch = datum.batch.to(self.device)
                y = datum.y.to(self.device)
                optimizer.zero_grad()
                y_hat = model(x=x, edge_index=edge_idx, batch=batch)
                loss = criterion(y.view(-1), y_hat.view(-1))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = 0.0
            for batch_idx, datum in enumerate(val_loader):
                x = datum.x.to(self.device)
                edge_idx = datum.edge_index.to(self.device)
                batch = datum.batch.to(self.device)
                y = datum.y.to(self.device)
                y_hat = model(x=x, edge_index=edge_idx, batch=batch)
                loss = criterion(y.view(-1), y_hat.view(-1))
                val_loss += loss.item()
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # print(f'Epoch: {epoch + 1} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            scheduler.step(val_loss)
            if optimizer.param_groups[0]['lr'] < 1e-5:
                break
    
    def train(self, train_loader, val_loader, tune=False):
        '''
            Trains a single GCN model or an ensemble of GCN models, if specified.
            If this is the first instance of model training or `tune` is set to
            True, then the models are trained from scratch. Otherwise, they are
            fine-tuned from the previous iteration.
        '''
        if self.model is None or tune:
            self.model = GCN(in_feats=self.in_feats)
            self.models = [GCN(in_feats=self.in_feats) for _ in range(self.n_models - 1)]
        self._train_gcn(self.model, train_loader, val_loader)
        if self.uncertainty:
            for m in self.models:
                self._train_gcn(m, train_loader, val_loader)

    def predict(self, test_loader):
        '''
            Make predictions on graphs in test_loader. Only uses the single model,
            rather than the ensemble, for predictions.
        '''
        y_pred = []
        self.model.eval()
        for batch_idx, datum in enumerate(test_loader):
            x = datum.x.to(self.device)
            edge_idx = datum.edge_index.to(self.device)
            batch = datum.batch.to(self.device)
            y_hat = self.model(x=x, edge_index=edge_idx, batch=batch)
            y_pred.append(y_hat.detach().cpu().numpy().reshape(-1))
        y_pred = np.hstack((y_pred))
        return y_pred

    def predict_single_graph(self, graph):
        '''
            Helper method for making a prediction on a single graph. This is
            used for the hallucination-based batch selection.
        '''
        x = graph.x.to(self.device)
        edge_index = graph.edge_index.to(self.device)
        batch = torch.zeros(x.size(0), dtype=torch.int64, device=self.device)
        y_hat = self.model(x=x, edge_index=edge_index, batch=batch)
        return y_hat.detach().cpu().numpy().reshape(-1).item()
    
    def get_uncertainties(self, test_loader):
        '''Measures uncertainty using an ensemble of GCNs.'''
        
        y_pred_ensemble = []

        # Get predictions from first model.
        y_pred = []
        self.model.eval()
        for batch_idx, datum in enumerate(test_loader):
            x = datum.x.to(self.device)
            edge_idx = datum.edge_index.to(self.device)
            batch = datum.batch.to(self.device)
            y_hat = self.model(x=x, edge_index=edge_idx, batch=batch)
            y_pred.append(y_hat.detach().cpu().numpy().reshape(-1))
        y_pred = np.hstack((y_pred))
        y_pred_ensemble.append(y_pred)

        # Get predictions from ensemble of models.
        for m in self.models:
            y_pred = []
            m.eval()
            for batch_idx, datum in enumerate(test_loader):
                x = datum.x.to(self.device)
                edge_idx = datum.edge_index.to(self.device)
                batch = datum.batch.to(self.device)
                y_hat = m(x=x, edge_index=edge_idx, batch=batch)
                y_pred.append(y_hat.detach().cpu().numpy().reshape(-1))
            y_pred = np.hstack((y_pred))
            y_pred_ensemble.append(y_pred)

        # Compute uncertainty.
        y_pred_ensemble = np.stack(y_pred_ensemble, axis=0)
        y_std = np.std(y_pred_ensemble, axis=0)

        return y_std
    
    def get_embeddings(self, test_loader):
        '''
            Generates the appropriate latent embeddings for test graphs
            from the single GCN model. Embeddings can come from the 
            graph embedding (i.e., `embedding`) or from the penultimate
            layer of the neural network (i.e., `penultimate`).
        '''
        embeddings = []
        self.model.eval()
        for batch_idx, datum in enumerate(test_loader):
            x = datum.x.to(self.device)
            edge_idx = datum.edge_index.to(self.device)
            batch = datum.batch.to(self.device)
            if self.embedding == 'embedding':
                embedding = self.model.embedding(x=x, edge_index=edge_idx, batch=batch)
            else:
                embedding = self.model.penultimate(x=x, edge_index=edge_idx, batch=batch)
            embeddings.append(embedding.detach().cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings
    
def build_gnn(model_name, in_feats):
    if model_name == 'gnn-embedding':
        return GNN(n_models=1, uncertainty=False, embedding='embedding', in_feats=in_feats)
    elif model_name == 'gnn-penultimate':
        return GNN(n_models=1, uncertainty=False, embedding='penultimate', in_feats=in_feats)
    elif model_name in ['gnn-ensemble', 'gnn-hallucinate', 'gnn-top']:
        return GNN(n_models=5, uncertainty=True, embedding='penultimate', in_feats=in_feats)
    elif model_name in ['gnn-pareto']:
        return GNN(n_models=5, uncertainty=True, embedding='embedding', in_feats=in_feats)
    else:
        raise Exception('Not a valid GNN model name.')