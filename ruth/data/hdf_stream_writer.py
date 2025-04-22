import h5py
import numpy as np

class HDF5Writer:
    def __init__(self, filename, dtype=None):
        self.file = h5py.File(filename, 'a')

        # Default compound dtype
        self.dtype = dtype or np.dtype([
            ('timestamp', 'S32'),
            ('vehicle_id', 'i4'),
            ('segment_id', 'i4'),
            ('start_offset', 'f8'),
            ('speed', 'f8'),
            ('status', 'S16'),
            ('active', 'i1')
        ])

        if 'results' not in self.file:
            self.dataset = self.file.create_dataset(
                'results',
                shape=(0,),
                maxshape=(None,),
                dtype=self.dtype,
                chunks=True
            )
        else:
            self.dataset = self.file['results']
        self.index = self.dataset.shape[0]

    def append(self, record):
        self.dataset.resize((self.index + 1,))

        # Create a 1-element structured array with the correct dtype
        structured = np.zeros(1, dtype=self.dtype)

        # Assign values by field name
        for field in self.dtype.names:
            structured[0][field] = record[field]

        self.dataset[self.index] = structured[0]  # assign single record
        self.index += 1
        self.file.flush()




    def close(self):
        self.file.close()
