import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt


class MyModel:
    """Initialize the model with extractor, classifier, and parameters for training.

    Parameters
    ----------
    extractor: The feature extractor model
    classifier: The classifier model
    params: The parameters for training
    """

    def __init__(self, extractor, classifier, params):

        self.extractor = extractor
        self.classifier = classifier
        self.params = params
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
        padded_region_sequences_train, padded_time_sequences_train, other_variables_train, labels_train = self.__preprocess_data(
            X_train, y_train)
        # Train the feature extractor
        extractor = self.__train_extractor(
            padded_region_sequences_train, padded_time_sequences_train, labels_train, show_loss)
        # Extract features from the training data using the trained extractor
        extracted_features = self.__extract_features(
            extractor, padded_region_sequences_train, padded_time_sequences_train, other_variables_train)
        # Train the classifier using the extracted features
        classifier = self.__train_classifier(
            extracted_features, labels_train)
        return extractor, classifier

    def evaluate(self, extractor, classifier, X_test):
        """Evaluate the model and return the predicted probabilities.

        Parameters
        ----------
        extractor: The feature extractor model
        classifier: The classifier model
        X_test: The input data for testing

        Returns
        -------
        predict_proba: The predicted probabilities of the positive class
        """
        padded_region_sequences_test, padded_time_sequences_test, other_variables_test = self.__preprocess_data(
            X_test)
        # Extract features from the testing data using the trained extractor
        extracted_features_test = self.__extract_features(
            extractor, padded_region_sequences_test, padded_time_sequences_test, other_variables_test)
        # Use the trained classifier to predict the probabilities of the positive class
        predict_proba = classifier.predict_proba(
            extracted_features_test)
        return predict_proba

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

    def __preprocess_data(self, X, y=None):
        """Preprocess the data by padding the region and time sequences to the same length.

        Parameters
        ----------
        X: The input data
        y: The target data (optional)

        Returns
        -------
        padded_region_sequences: The padded region sequences
        padded_time_sequences: The padded time sequences
        other_variables: Other variables in the data
        labels (if y is not None): The target labels
        """
        # Pad the region and time sequences to the same length
        region_sequences = X[:, 2::2]
        time_sequences = X[:, 3::2]
        other_variable = X[:, :2]
        padded_region_sequences = pad_sequence([torch.IntTensor(
            seq) for seq in region_sequences], batch_first=True, padding_value=0)
        padded_time_sequences = pad_sequence([torch.FloatTensor(
            seq) for seq in time_sequences], batch_first=True, padding_value=0)
        other_variables = torch.FloatTensor(other_variable)
        if y is None:
            return padded_region_sequences, padded_time_sequences, other_variables
        else:
            labels = torch.FloatTensor(y)
            return padded_region_sequences, padded_time_sequences, other_variables, labels

    def __train_extractor(self, padded_region_sequences_train, padded_time_sequences_train, labels_train, show_loss):
        """Train the extractor model and return the trained model.

        Parameters
        ----------
        padded_region_sequences_train: The padded region sequences for training
        padded_time_sequences_train: The padded time sequences for training
        labels_train: The target labels for training

        Returns
        -------
        model: The trained extractor model
        """
        device = self.device
        model = self.extractor.to(device)
        region_batch = padded_region_sequences_train.to(device)
        time_batch = padded_time_sequences_train.to(device)
        labels_batch = labels_train.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters())
        model.train()
        if show_loss:
            loss_list_step = []
            for epoch in range(self.params['num_epochs']):
                outputs = model(region_batch, time_batch)  # Forward pass
                optimizer.zero_grad()
                # Compute loss
                loss = criterion(outputs, labels_batch.long())
                loss.backward()  # Backward pass
                loss_list_step.append(loss.item())
                optimizer.step()  # Update the weights
            self.__draw_loss(loss_list_step)
        else:
            for epoch in range(self.params['num_epochs']):
                outputs = model(region_batch, time_batch)
                loss = criterion(outputs, labels_batch.long())
                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights
        return model

    def __extract_features(self, extractor, region_sequences, time_sequences, other_variable):
        """Extract features from the data using the extractor model.

        Parameters
        ----------
        extractor: The feature extractor model
        region_sequences: The region sequences data
        time_sequences: The time sequences data
        other_variable: Other variables data

        Returns
        -------
        extracted_features: The extracted features from the data
        """
        device = self.device
        model = extractor.to(device)
        model.eval()
        with torch.no_grad():
            region_sequences, time_sequences = region_sequences.to(
                device), time_sequences.to(device)
            region_embed = model.region_embedding(
                region_sequences)  # Embed the region sequences
            # Embed the time sequences
            time_embed = model.time_embedding(
                region_sequences) * time_sequences.unsqueeze(-1)
            # Combine the embeddings
            combined_embed = torch.cat([region_embed, time_embed], dim=-1)
            # Pass the combined embeddings through the LSTM
            _, (hidden, _) = model.lstm(combined_embed)
            extracted_features = hidden[-1].cpu().numpy()  # Get the features from the LSTM
            # Combine the other variables with the extracted features
            extracted_features = np.concatenate(
                [other_variable, extracted_features], axis=1)
        return extracted_features

    def __train_classifier(self, extracted_features, labels_train):
        """Train the classifier model with the extracted features and return the trained model.

        Parameters
        ----------
        extracted_features: The extracted features from the data
        labels_train: The target labels for training

        Returns
        -------
        classifier: The trained classifier model
        """
        classifier = self.classifier.fit(
            extracted_features, labels_train)  # Train the classifier
        return classifier

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
