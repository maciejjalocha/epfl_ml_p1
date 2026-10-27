"""Minimal sparse matrix arithmetic using NumPy only."""
import numpy as np


class SparseMatrix:
    def __init__(self, rows, cols, data, shape):
        self.rows = np.asarray(rows, dtype=np.int32)
        self.cols = np.asarray(cols, dtype=np.int32)
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = tuple(shape)

    @classmethod
    def from_blocks(cls, blocks, width):
        rows, cols, data, offset = [], [], [], 0
        for block in blocks:
            r, c = np.nonzero(block)
            rows.append((r + offset).astype(np.int32))
            cols.append(c.astype(np.int32))
            data.append(block[r, c])
            offset += len(block)
        return cls(np.concatenate(rows), np.concatenate(cols), np.concatenate(data), (offset, width))

    @classmethod
    def from_dense(cls, x):
        if isinstance(x, cls):
            return x
        return cls.from_blocks((x[i:i+4096] for i in range(0, len(x), 4096)), x.shape[1])

    def __matmul__(self, weights):
        return np.bincount(self.rows, weights=self.data * weights[self.cols],
                           minlength=self.shape[0]).astype(np.float32)

    def transpose_dot(self, residual):
        return np.bincount(self.cols, weights=self.data * residual[self.rows],
                           minlength=self.shape[1]).astype(np.float32)

    def select(self, columns):
        mapping = np.full(self.shape[1], -1, dtype=np.int32)
        mapping[columns] = np.arange(len(columns))
        new_cols = mapping[self.cols]
        keep = new_cols >= 0
        return SparseMatrix(self.rows[keep], new_cols[keep], self.data[keep],
                            (self.shape[0], len(columns)))

    def __getitem__(self, item):
        # Only the notebook's small column slices are materialized, never all columns.
        rows, columns = item
        if rows != slice(None):
            raise ValueError('Only full-row column slices are supported')
        indices = np.arange(self.shape[1])[columns]
        selected = self.select(indices)
        out = np.zeros(selected.shape, dtype=np.float32)
        out[selected.rows, selected.cols] = selected.data
        return out
