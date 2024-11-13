import numpy as np
from builtins import range
from builtins import zip
from scipy import sparse
from scipy.sparse import issparse, lil_matrix

from .partition import LabelSpacePartitioningClassifier


class MajorityVotingClassifier(LabelSpacePartitioningClassifier):
    """Majority Voting ensemble classifier

    Divides the label space using provided clusterer class, trains a provided base classifier
    type classifier for each subset and assign a label to an instance
    if more than half of all classifiers (majority) from clusters that contain the label
    assigned the label to the instance.

    Parameters
    ----------
    classifier : :class:`~sklearn.base.BaseEstimator`
        the base classifier that will be used in a class, will be
        automatically put under :code:`self.classifier`.
    clusterer : :class:`~skmultilearn.cluster.LabelSpaceClustererBase`
        object that partitions the output space, will be
        automatically put under :code:`self.clusterer`.
    require_dense : [bool, bool]
        whether the base classifier requires [input, output] matrices
        in dense representation, will be automatically
        put under :code:`self.require_dense`.


    Attributes
    ----------
    model_count_ : int
        number of trained models, in this classifier equal to the number of partitions
    partition_ : List[List[int]], shape=(`model_count_`,)
        list of lists of label indexes, used to index the output space matrix, set in :meth:`_generate_partition`
        via :meth:`fit`
    classifiers : List[:class:`~sklearn.base.BaseEstimator`], shape=(`model_count_`,)
        list of classifiers trained per partition, set in :meth:`fit`


    Examples
    --------
    Here's an example of building an overlapping ensemble of chains

    .. code :: python

        from skmultilearn.ensemble import MajorityVotingClassifier
        from skmultilearn.cluster import FixedLabelSpaceClusterer
        from skmultilearn.problem_transform import ClassifierChain
        from sklearn.naive_bayes import GaussianNB


        classifier = MajorityVotingClassifier(
            clusterer = FixedLabelSpaceClusterer(clusters = [[1,2,3], [0, 2, 5], [4, 5]]),
            classifier = ClassifierChain(classifier=GaussianNB())
        )
        classifier.fit(X_train,y_train)
        predictions = classifier.predict(X_test)

    More advanced examples can be found in `the label relations exploration guide <../labelrelations.ipynb>`_

    """

    def __init__(self, classifier=None, clusterer=None, require_dense=None):
        super(MajorityVotingClassifier, self).__init__(
            classifier=classifier, clusterer=clusterer, require_dense=require_dense
        )

    def predict(self, X):
        """Predict labels for X"""
        voters = np.zeros(self._label_count, dtype='int')
        predictions = []
        n_samples = X.shape[0]

        # Collect predictions
        for classifier in self.classifiers_:
            pred = classifier.predict(X)
            if issparse(pred):
                pred = pred.toarray()
            # Special handling for MLARAM predictions
            if len(pred.shape) == 1:
                pred = pred.reshape(-1, 1)
            elif pred.shape[1] != len(self.partition_[0]):
                pred = pred.reshape(n_samples, -1)
            predictions.append(pred)

        # Initialize votes matrix
        votes = lil_matrix((n_samples, self._label_count), dtype='int')

        # Accumulate votes
        for model in range(self.model_count_):
            prediction = predictions[model]
            for label_idx, partition_label in enumerate(self.partition_[model]):
                if label_idx >= prediction.shape[1]:
                    continue

                # Get the column of votes for this partition
                current_votes = votes[:, partition_label].toarray().flatten()
                new_votes = prediction[:, label_idx].flatten()

                # Update votes using direct assignment
                for i in range(n_samples):
                    votes[i, partition_label] = current_votes[i] + new_votes[i]

                voters[partition_label] += 1

        # Normalize votes
        nonzeros = votes.nonzero()
        for row, column in zip(nonzeros[0], nonzeros[1]):
            votes[row, column] = np.round(
                votes[row, column] / float(voters[column]))

        return self._ensure_output_format(votes, enforce_sparse=False)