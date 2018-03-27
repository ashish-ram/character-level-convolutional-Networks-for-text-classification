import numpy as np


class Data(object):
    """
    Class to handle loading and processing of raw datasets.
    """

    def __init__(self,
                 training_data_x,
                 training_labels_y,
                 alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 input_size=1014, num_of_classes=12):
        """
        Initialization of a Data object.

        Args:
            data_source (str): Raw data file path
            alphabet (str): Alphabet of characters to index
            input_size (int): Size of input features
            num_of_classes (int): Number of classes in data
        """
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}  # Maps each character to an integer
        self.no_of_classes = num_of_classes
        for index, char in enumerate(self.alphabet):
            self.dict[char] = index + 1
        self.length = input_size
        self.training_data_x = training_data_x
        self.training_labels_y = training_labels_y

    def load_data(self):
        """
        Load raw data from the source file into data variable.

        Returns: None

        """
        data = []
        f_x = open(self.training_data_x, 'r')
        f_y = np.loadtxt(self.training_labels_y, dtype='int64')
        lines = f_x.readlines()
        for y, x in zip(f_y, lines):
            data.append((y, x))
        self.data = np.array(data)
        print("########## Data loaded ##########")

    def get_all_data(self):
        """
        Return all loaded data for training the network from data variable.

        Returns:
            (np.ndarray) Data transformed from raw to indexed form with associated one-hot label.

        """
        data_size = len(self.data)
        start_index = 0
        end_index = data_size
        batch_texts = self.data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.str_to_indexes(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def str_to_indexes(self, s):
        """
        Convert a string to character indexes based on character dictionary.

        Args:
            s (str): String to be converted to indexes

        Returns:
            s1hot (np.ndarray): Indexes of characters in s

        """
        s = s.lower()
        max_length = min(len(s), self.length)
        s1hot = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                s1hot[i - 1] = self.dict[c]
        return s1hot
