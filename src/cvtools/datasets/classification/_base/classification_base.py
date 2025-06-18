"""
Base class for classification datasets.
"""

# Author: Atif Khurshid
# Created: 2025-06-18
# Modified: None
# Version: 1.0
# Changelog:
#     - None


class _ClassificationBase:
    """
    Base class for classification datasets.
    This class provides a common interface for classification datasets.
    """

    def __init__(self):

        self.classes: list
        self.labels: list
        self.class2label: dict
        self.label2class: dict


    def __initialize__(self):
        """
        Initialize the dataset by setting up class names, labels, and mappings.
        This method should be called at the end of the subclass' constructor.
        """
        self.class2index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.index2class = {idx: cls for cls, idx in self.class2index.items()}


    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.labels)


    def __getitem__(self, index: int) -> tuple:
        """
        Get a sample from the dataset.
        
        Parameters
        ----------
        index : int
            Index of the sample to retrieve.
        
        Returns
        -------
        tuple
            A tuple containing the sample data and its label.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    

    @property
    def num_classes(self) -> int:
        """
        Return the number of classes in the dataset.
        
        Returns
        -------
        int
            The number of classes.
        """
        return len(self.classes)
    

    def class_name_to_index(self, x: str) -> int:
        """
        Convert a class name to its corresponding index.

        Parameters
        ----------
        x : str
            The class name to convert.

        Returns
        -------
        int
            The corresponding index for the class name.
        """
        return self.class2index.get(x, -1)


    def index_to_class_name(self, x: int) -> str:
        """
        Convert an index to its corresponding class name.

        Parameters
        ----------
        x : int
            The index to convert.

        Returns
        -------
        str
            The corresponding class name for the index.
        """
        return self.index2class.get(x, "Unknown")
