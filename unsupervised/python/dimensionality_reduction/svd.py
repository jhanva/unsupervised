# External libraries
import numpy as np


class SVD:
    """Initialize the SVD object.

    Args:
        n_components: Number of components to retain.

    """

    def __init__(self, n_components: int = None):
        """Initialize a new SVD instance.

        Args:
            n_components (int, optional): Number of components to retain.

        """
        self.n_components = n_components
        self.u = None
        self.sigma = None
        self.v = None

    def fit(self, matrix: np.array) -> None:
        """Fit the SVD to the input matrix.

        Args:
            matrix (numpy.ndarray): The input matrix.

        """
        matrix = np.array(matrix)
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        self.u = u
        self.sigma = s
        self.v = v

    def fit_transform(self, matrix: np.array) -> np.array:
        """Fit the SVD and return the transformed data.

        Args:
            matrix: The input matrix.

        Returns:
            Transformed data.

        """
        self.fit(matrix)
        return self.transform()

    def transform(self) -> np.array:
        """Transform the data based on the specified number of components.

        Returns:
            Transformed data.

        """
        if self.n_components:
            result = (
                self.u[:, : self.n_components]
                @ np.diag(self.sigma[: self.n_components])
                @ self.v[: self.n_components, :]
            )

        else:
            result = self.u @ np.diag(self.sigma) @ self.v

        return result
