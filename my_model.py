import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


class MyModel:
    """Initialize the model with extractor, classifier, sequences, other variables, labels, parameters, and device.

    Parameters
    ----------
    extractor: The feature extractor model
    classifier: The classifier model
    region_sequences: The region sequences data
    time_sequences: The time sequences data
    other_variable: Other variables data
    labels: The labels for the data
    params: The parameters for training
    device: The device to run the model on
    """

    def __init__(self, extractor, classifier, region_sequences, time_sequences, other_variable, labels, params, device):
        """Train the model and return the trained extractor and classifier.

        Returns
        -------
        extractor: The trained feature extractor model
        classifier: The trained classifier model
        """
        self.extractor = extractor
        self.classifier = classifier
        self.region_sequences = region_sequences
        self.time_sequences = time_sequences
        self.other_variable = other_variable
        self.labels = labels
        self.params = params
        self.device = device

    def train(self):
        """Train the model and return the trained extractor and classifier.

        Returns
        -------
        extractor: The trained feature extractor model
        classifier: The trained classifier model
        """
        self.__split_data()  # Split the data into training and testing sets
        extractor = self.__train_extractor()  # Train the feature extractor
        # Extract features from the training data using the trained extractor
        extracted_features = self.__extract_features(
            extractor, self.region_sequences_train, self.time_sequences_train, self.other_variable_train)
        classifier = self.__train_classifier(
            extracted_features)  # Train the classifier using the extracted features
        return extractor, classifier

    def evaluate(self, extractor, classifier):
        """Evaluate the model and return the predicted probabilities and the test labels.

        Parameters
        ----------
        extractor: The feature extractor model
        classifier: The classifier model

        Returns
        -------
        predict_proba: The predicted probabilities of the positive class
        labels_test: The test labels
        """
        # Extract features from the testing data using the trained extractor
        extracted_features_test = self.__extract_features(
            extractor, self.region_sequences_test, self.time_sequences_test, self.other_variable_test)
        # Use the trained classifier to predict the probabilities of the positive class
        predict_proba = classifier.predict_proba(
            extracted_features_test)[:, 1]
        return predict_proba, self.labels_test

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

    def __split_data(self):
        """Split the data into training and testing sets."""
        # Pad the region and time sequences to the same length
        padded_region_sequences = pad_sequence([torch.IntTensor(
            seq) for seq in self.region_sequences], batch_first=True, padding_value=0)
        padded_time_sequences = pad_sequence([torch.FloatTensor(
            seq) for seq in self.time_sequences], batch_first=True, padding_value=0)
        labels = torch.FloatTensor(self.labels)
        other_variables = torch.tensor(self.other_variable)
        ratio = 0.8
        train_idx = int(len(padded_region_sequences) * ratio)
        test_idx = int(len(padded_region_sequences) * ratio)
        # Split the data into training and testing sets
        self.region_sequences_train, self.time_sequences_train, self.labels_train, self.other_variable_train = padded_region_sequences[
            :train_idx], padded_time_sequences[:train_idx]/padded_time_sequences[:train_idx].max(), labels[:train_idx], other_variables[:train_idx]
        self.region_sequences_test, self.time_sequences_test, self.labels_test, self.other_variable_test = padded_region_sequences[
            test_idx:], padded_time_sequences[test_idx:]/padded_time_sequences[:test_idx].max(), labels[test_idx:], other_variables[test_idx:]

    def __train_extractor(self):
        """Train the extractor model and return the trained model."""
        model = self.extractor
        device = self.device

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters())
        model.train()
        for epoch in range(self.params['num_epochs']):
            region_batch = self.region_sequences_train.to(device)
            time_batch = self.time_sequences_train.to(device)
            labels_batch = self.labels_train.to(device)
            optimizer.zero_grad()
            outputs = model(region_batch, time_batch)  # Forward pass
            # Compute loss
            loss = criterion(outputs, labels_batch.unsqueeze(-1))
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
        model = extractor
        device = self.device
        model.eval()
        with torch.no_grad():
            region_sequences, time_sequences = region_sequences.to(
                device), time_sequences.to(device)
            region_embed = model.region_embedding(
                region_sequences)  # Embed the region sequences
            time_embed = model.time_embedding(
                # Embed the time sequences
                region_sequences) * time_sequences.unsqueeze(-1)
            # Combine the embeddings
            combined_embed = torch.cat([region_embed, time_embed], dim=-1)
            # Pass the combined embeddings through the LSTM
            _, (hidden, _) = model.lstm(combined_embed)
            extracted_features = hidden.squeeze(
                0).cpu().numpy()  # Get the features from the LSTM
        extracted_features = np.concatenate(
            # Combine the other variables with the extracted features
            [other_variable, extracted_features], axis=1)
        return extracted_features

    def __train_classifier(self, extracted_features):
        """Train the classifier model with the extracted features and return the trained model.

        Parameters
        ----------
        extracted_features: The extracted features from the data

        Returns
        -------
        classifier: The trained classifier model
        """
        classifier = self.classifier.fit(
            extracted_features, self.labels_train)  # Train the classifier
        return classifier
