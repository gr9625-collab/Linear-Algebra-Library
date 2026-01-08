import numpy as np

class Matrix:
    def __init__(self, entries):
        self.entries = np.array(entries, dtype=float)

        if self.entries.ndim != 2:
            raise ValueError("Matrix must be 2-dimensional")
        
    @property
    def shape(self):
        return self.entries.shape
    
    def is_vec(self):
        m, n = self.shape
        return n == 1
    
    def vec_proj(self, other):
        if not (self.is_vec() and other.is_vec()):
            raise ValueError("Vectors must have exactly one column")
        top = sum(self.entries * other.entries)
        bottom = sum(other.entries * other.entries)

        return Matrix((top / bottom) * other.entries)
    
    def norm(self):
        if not self.is_vec():
            raise ValueError("Vectors must have exactly one column")
        A = self.entries
        return sum(A * A) ** 0.5
    
    def gram_schmidt(self):
        m, n = self.shape
        A = self.entries
        Q = np.zeros((m, n))

        for i in range(n):
            w = A[:, i]

            for j in range(i):
                qj = Q[:, j]
                w = w - Matrix(w).vec_proj(Matrix(qj))

            norm_w = Matrix(w).norm()
            if norm_w < 1e-12:
                raise ValueError("Columns are linearly dependent")

            Q[:, i] = w / norm_w

        return Matrix(Q)
    
    def QR_decomp(self):
        m, n = self.shape
        A = self.entries
        Q = self.gram_schmidt().entries
        R = Matrix(Q.T).__matmul__(Matrix(A))
        return Matrix(Q), R
            


    def __matmul__(self, other):
        m, n = self.shape
        a, b = other.shape

        if not isinstance(other, Matrix):
            raise TypeError("Can only multiply by another Matrix")

        if a != n:
            raise ValueError(f"Cannot multiply {m} x {n} matrix and {a} x {b} matrix")
        
        result = np.zeros((m, b))

        for i in range(0, m):
            for j in range(0, b):
                col_j = [other.entries[k][j] for k in range(0, a)]
                result[i][j] = sum(self.entries[i] * col_j)
        
        return Matrix(result)

    def __str__(self):
        rows = []
        for row in self.entries:
            rows.append(
                "[" + " ".join(f"{0 if abs(x) < 1e-12 else x:8.3g}" for x in row) + "]"
            )
        return "\n".join(rows)
    
    # A is an m by n matrix
    def row_echelon(self):
        A = self.entries.copy()
        m, n = A.shape

        pivot_row = 0
        pivot_col = 0

        while pivot_row < m and pivot_col < n:
            # Find pivot
            pivot_found = False
            for i in range(pivot_row, m):
                if abs(A[i, pivot_col]) > 1e-12:
                    A[[pivot_row, i]] = A[[i, pivot_row]]
                    pivot_found = True
                    break

            if not pivot_found:
                pivot_col += 1
                continue

            # Eliminate below
            for i in range(pivot_row + 1, m):
                factor = A[i, pivot_col] / A[pivot_row, pivot_col]
                A[i] -= factor * A[pivot_row]

            pivot_row += 1
            pivot_col += 1

        return Matrix(A)
    
    def rank(self):
        R = self.row_echelon()
        return sum(
            not np.all(np.abs(row) < 1e-12)
            for row in R.entries
        )
    
    def reduced_row_echelon(self):
        # Put the m by n matrix in row echelon form first
        A = self.row_echelon().entries.copy()
        m, n = self.shape

        # Identify the leading columns
        leading_columns = []
        current_height = 0
        for i in range(0, n):
            if current_height >= m:
                break
            if A[current_height, i] != 0:
                leading_columns.append(i)
                current_height += 1

        # For each leading column, normalise the pivot row
        # Then eliminate the entries below the pivot element
        for i, j in enumerate(leading_columns):
            A[i] = A[i] / A[i, j]
            for k in range(0, i):
                A[k] -= A[k,j] * A[i]

        return Matrix(A)
    
    # We can then check if a matrix is invertible or not
    def is_invertible(self):
        A = self.reduced_row_echelon()
        m, n = self.shape

        if m != n:
            raise ValueError("Matrix is not square")
        
        return np.allclose(A.entries, np.identity(m))

    def matrix_inverse(self):
        m, n = self.shape
        if m != n:
            raise ValueError("Matrix is not square")

        A = np.hstack((self.entries, np.identity(m)))
        A = Matrix(A).reduced_row_echelon().entries
        left = A[:, :m]
        right = A[:, m:]

        if not np.allclose(left, np.identity(m)):
            print("Matrix is not invertible")
            raise ValueError("Matrix is not invertible")
        
        return Matrix(right)
    
    def determinant(self):
        A = self.entries.copy()
        m, n = self.shape

        if m != n:
            raise ValueError("Matrix is not square")
        
        determinant = 1
        pivot_row = 0
        pivot_col = 0

        while pivot_row < m and pivot_col < n:
            # Find pivot
            pivot_found = False
            for i in range(pivot_row, m):
                if abs(A[i, pivot_col]) > 1e-12:
                    A[[pivot_row, i]] = A[[i, pivot_row]]
                    if i != pivot_row:
                        determinant *= -1
                    pivot_found = True
                    break

            if not pivot_found:
                pivot_col += 1
                continue

            # Eliminate below
            for i in range(pivot_row + 1, m):
                factor = A[i, pivot_col] / A[pivot_row, pivot_col]
                A[i] -= factor * A[pivot_row]

            pivot_row += 1
            pivot_col += 1

        for i in range(0, m):
            determinant *= A[i, i]
        
        return determinant
    
    # We are up to here, coding the QR algorithm to allow us to numerically find eigenvalues

    def eigenvalues(self):
        m, n = self.shape
        if m != n:
            raise ValueError("Matrix is not square")

        A = Matrix(self.entries.copy())
        tol = 1e-10
        max_iter = 1000
        it = 0
        
        while np.linalg.norm(np.tril(A.entries, -1)) > tol and it < max_iter:
            Q, R = A.QR_decomp()
            A = R.__matmul__(Q)
            it += 1

        return np.diag(A.entries)

    
A = Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(A, "\n")
print(A.determinant(), "\n")


# print(A.reduced_row_echelon(), "\n")
# print(A.is_invertible())

# I want to take random matrices as input, from numpy, and see if my code agrees with it.
# To see if my code works and also to see if I know how to test code