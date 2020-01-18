import numpy as np


class PCA:
    def __init__(self, square_data, k=10):  # k defaults to 10
        """
        Create a new PCA instance

        :param square_data: the input data
        :param k: number of principal components
        """
        self.square_data = square_data
        M, rows, columns = self.square_data.shape
        self.k = k
        self.num_examples = M
        self.image_vec = np.reshape(square_data, (M, rows * columns))
        self.mean_face = np.mean(self.image_vec, axis=0)
        self.std_face = np.std(self.image_vec, axis=0)
        self.components, self.singular_values = self.get_components()

    def get_components(self):
        """
        Calculate the principal components and singular values

        :return: principal components and singular values
        """
        Phi = (self.image_vec - self.mean_face) / self.std_face
        A = Phi
        C = np.matmul(A, A.T)
        C = np.divide(C, self.num_examples - 1)
        evals, Vi = np.linalg.eigh(C)
        z = list(zip(evals, Vi))
        z.sort(reverse=True)

        idx = np.argsort(evals)[::-1]
        evecs = Vi[:, idx]
        evecs = evecs[:, :self.k]
        pc = evecs

        # final components (num pixels by k matrix)
        components = np.matmul(A.T, pc)
        components = components / np.linalg.norm(components, axis=0)
        # get singluar values
        sorted_evals = np.array([x[0] for x in z])
        postive_evals = sorted_evals[:self.k]
        singular_values = np.sqrt(postive_evals.reshape(1, -1))
        assert np.allclose(np.linalg.norm(components, axis=0), 1)

        return components, singular_values

    def transform(self, images):
        """
        Transform an array of images

        :param images: untransformed images, array
        :return: PCA transformed images
        """
        return np.array([self.transform_single(i) for i in images])

    def transform_single(self, image):  # take an image, and pc's, and output compressed image
        """

        :param image: pca transform a single image
        :return: pca transformed image
        """
        image = image.reshape(1, -1)
        image = (image - self.mean_face) / self.std_face
        compressed_image_vectors = np.matmul(image, self.components) / self.singular_values
        return compressed_image_vectors.reshape(-1, )
