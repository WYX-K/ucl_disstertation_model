import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt

from lstm_model import MyDataset


class MyModel:
    """Initialize the model with extractor, classifier, and parameters for training.

    Parameters
    ----------
    extractor: The feature extractor model
    classifier: The classifier model
    params: The parameters for training
    """

    def __init__(self, extractor=None, params=None, classifier=None, n_class=3):

        self.extractor = extractor
        self.classifier = classifier
        self.params = params
        self.n_class = n_class
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, X_train, y_train, show_loss=False):
        """Train the model and return the trained extractor and classifier.

        Parameters
        ----------
        X_train: The input data for training
        y_train: The target data for training
        show_loss: Whether to show the loss curve. Default is False.

        Returns
        -------
        extractor: The trained feature extractor model
        classifier: The trained classifier model
        """
        # Split the data into training and testing sets
        train_data_loader = self.__preprocess_data(
            X_train, y_train)
        # Train the feature extractor
        extractor = self.__train_extractor(train_data_loader, show_loss)
        if self.classifier is None:
            return extractor
        classifier = self.__train_classifier(extractor, train_data_loader)
        return extractor, classifier

    def evaluate(self, extractor, X_test, y_test, classifier=None):
        """Evaluate the model and return the predicted probabilities.

        Parameters
        ----------
        extractor: The feature extractor model
        classifier: The classifier model
        X_test: The input data for testing
        y_test: The target data for testing

        Returns
        -------
        predict_proba: The predicted probabilities of the positive class
        labels: The target labels
        """
        test_data_loader = self.__preprocess_data(
            X_test, y_test)
        # Use the trained classifier to predict the probabilities of the positive class
        # predict_proba = classifier.predict_proba(
        #     extracted_features_test)
        predict_proba_with_labels = self.__evaluate_model(
            extractor, classifier, test_data_loader)
        labels = predict_proba_with_labels[:, -1].reshape(-1, 1)
        predict_proba = predict_proba_with_labels[:, :-1]
        return predict_proba, labels

    def predict(self, extractor, classifier, X):
        """Predict the labels of the input data.

        Parameters
        ----------
        extractor: The feature extractor model
        classifier: The classifier model
        X: The input data for prediction

        Returns
        -------
        labels: The predicted labels
        """
        data_loader = self.__preprocess_data(X, None)
        predict_proba_with_labels = self.__evaluate_model(
            extractor, classifier, data_loader)
        labels = predict_proba_with_labels[:, -1].reshape(-1, 1)
        return labels

    def save_model(self, extractor, classifier, extractor_name, classifier_name):
        """Save the extractor and classifier models.

        Parameters
        ----------
        extractor: The feature extractor model
        classifier: The classifier model
        extractor_name: The name of the extractor model
        classifier_name: The name of the classifier model
        """
        model_save_dir = f'./models/{extractor_name}_{classifier_name}'
        if not os.path.exists(model_save_dir):
            # Create the directory if it does not exist
            os.makedirs(model_save_dir)
        extractor_model_path = os.path.join(
            model_save_dir, f'{extractor_name}.pth')
        classifier_model_path = os.path.join(
            model_save_dir, f'{classifier_name}.dta')
        # Save the extractor model
        torch.save(extractor.state_dict(), extractor_model_path)
        # Save the classifier model
        joblib.dump(classifier, classifier_model_path)

    def __preprocess_data(self, X, y):
        """Preprocess the data by padding the region and time sequences to the same length.

        Parameters
        ----------
        X: The input data
        y: The target data (optional)

        Returns
        -------
        data_loader: The DataLoader object for the preprocessed data
        """
        def collate_fn(dataset):
            region_sequences, time_sequences, targets, other_features, original_length = [], [], [], [], []
            for data in dataset:
                region_sequence, time_sequence, other_variable, y = data
                region_sequences.append(torch.tensor(
                    region_sequence, dtype=torch.int))
                time_sequences.append(torch.tensor(
                    time_sequence, dtype=torch.float))
                targets.append(torch.tensor(y, dtype=torch.long))
                other_features.append(other_variable)
                zero_index = np.where(region_sequence == 0)[0]
                if zero_index.size == 0:
                    original_length.append(region_sequence.size)
                else:
                    original_length.append(zero_index[0])
            region_sequences = torch.stack(region_sequences)
            time_sequences = torch.stack(time_sequences)
            targets = torch.stack(targets)
            other_features = np.asarray(other_features)
            original_length = torch.tensor(original_length)
            return region_sequences, time_sequences, targets, other_features, original_length
            # return X, y, original_length
        dataset = MyDataset(X, y)
        data_loader = DataLoader(
            dataset=dataset, batch_size=self.params['batch_size'], collate_fn=collate_fn)
        return data_loader

    def __train_extractor(self, train_data_loader, show_loss):
        """Train the extractor model and return the trained model.

        Parameters
        ----------
        train_data_loader: The DataLoader object for the training data
        show_loss: Whether to show the loss curve

        Returns
        -------
        model: The trained extractor model
        """
        device = self.device
        model = self.extractor.to(device)
        if self.n_class == 1:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters())
        model.train()
        if show_loss:
            loss_list = []
            for epoch in range(self.params['num_epochs']):
                loss_list_step = []
                for region_batch, time_batch, labels_batch, _, original_length in train_data_loader:
                    region_batch, time_batch, labels_batch = region_batch.to(
                        device), time_batch.to(device), labels_batch.to(device)
                    outputs = model(region_batch, time_batch,
                                    original_length)  # Forward pass
                    optimizer.zero_grad()
                    # Compute loss
                    if self.n_class == 1:
                        loss = criterion(outputs.squeeze(),
                                         labels_batch.squeeze().float())
                    else:
                        loss = criterion(outputs, labels_batch.squeeze())
                    loss.backward()  # Backward pass
                    loss_list_step.append(loss.item())
                    optimizer.step()  # Update the weights
                loss_list.append(np.mean(loss_list_step))
            self.__draw_loss(loss_list)
        else:
            for epoch in range(self.params['num_epochs']):
                for region_batch, time_batch, labels_batch, _, original_length in train_data_loader:
                    region_batch, time_batch, labels_batch = region_batch.to(
                        device), time_batch.to(device), labels_batch.to(device)
                    outputs = model(region_batch, time_batch,
                                    original_length)  # Forward pass
                    optimizer.zero_grad()
                    # Compute loss
                    if self.n_class == 1:
                        loss = criterion(outputs.squeeze(),
                                         labels_batch.squeeze().float())
                    else:
                        loss = criterion(outputs, labels_batch.squeeze())
                    loss.backward()  # Backward pass
                    optimizer.step()  # Update the weights
        return model

    def __train_classifier(self, extractor, data_loader):
        """Extract features from the data using the extractor model.

        Parameters
        ----------
        extractor: The feature extractor model
        data_loader: The DataLoader object for the data

        Returns
        -------
        classifier: The trained classifier model
        """
        device = self.device
        model = extractor.to(device)
        model.eval()
        with torch.no_grad():
            for region_batch, time_batch, labels_batch, other_features, original_length in data_loader:
                region_batch, time_batch = region_batch.to(
                    device), time_batch.to(device)
                region_embed = model.region_embedding(
                    region_batch)  # Embed the region sequences
                # Embed the time sequences
                time_embed = model.time_embedding(
                    region_batch) * time_batch.unsqueeze(-1)
                # Combine the embeddings
                combined_embed = torch.cat([region_embed, time_embed], dim=-1)
                combined_embed_packed = pack_padded_sequence(
                    combined_embed, original_length, batch_first=True, enforce_sorted=False)
                # Pass the combined embeddings through the LSTM
                _, (hidden, _) = model.lstm(combined_embed_packed)
                # Get the features from the LSTM
                extracted_features = hidden[-1].cpu().numpy()
                extracted_features = np.concatenate(
                    [other_features, extracted_features], axis=1)
                # Combine the other variables with the extracted features
                self.classifier.fit(extracted_features, labels_batch)
        return self.classifier

    def __evaluate_model(self, extractor, classifier, data_loader):
        """Evaluate the model and return the predicted probabilities with labels.

        Parameters
        ----------
        extractor: The feature extractor model
        classifier: The classifier model
        data_loader: The DataLoader object for the data

        Returns
        -------
        results: The predicted probabilities with labels
        """
        device = self.device
        model = extractor.to(device)
        model.eval()

        def get_proba_and_label(region_batch, time_batch, labels_batch, other_features, original_length):
            region_batch, time_batch = region_batch.to(
                device), time_batch.to(device)
            region_embed = model.region_embedding(
                region_batch)  # Embed the region sequences
            time_embed = model.time_embedding(
                # Embed the time sequences
                region_batch) * time_batch.unsqueeze(-1)
            # Combine the embeddings
            combined_embed = torch.cat([region_embed, time_embed], dim=-1)
            combined_embed_packed = pack_padded_sequence(
                combined_embed, original_length, batch_first=True, enforce_sorted=False)
            # Pass the combined embeddings through the LSTM
            _, (hidden, _) = model.lstm(combined_embed_packed)
            if classifier == None:
                if self.n_class == 1:
                    outputs = model.binary_classifier(hidden[-1])
                    proba = outputs.cpu().numpy()
                else:
                    outputs = model.multi_classifier(hidden[-1])
                    proba = outputs.cpu().numpy()
            else:
                extracted_features = hidden[-1].cpu().numpy()
                # Combine the other variables with the extracted features
                extracted_features = np.concatenate(
                    [other_features, extracted_features], axis=1)
                # Get the prediction probabilities
                proba = classifier.predict_proba(extracted_features)
            labels = labels_batch.numpy().reshape(-1, 1)
            return np.concatenate([proba, labels], axis=1)

        with torch.no_grad():
            results = map(lambda data: get_proba_and_label(*data), data_loader)
            results = np.vstack(list(results))
        return results

    def __draw_loss(self, loss_list_step):
        """Draw the loss curve for the model.

        Parameters
        ----------
        loss_list_step: The list of loss values at each step
        """
        plt.plot(np.linspace(
            1, self.params['num_epochs'], self.params['num_epochs']), loss_list_step)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss in Different Epochs')
        plt.xticks(np.linspace(1, self.params['num_epochs'], 10).astype(int))
        plt.yticks(np.linspace(0, 1, 5))
        plt.show()
