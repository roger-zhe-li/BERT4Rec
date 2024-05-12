import pandas as pd
import tensorflow as tf  # just for transforming tfrecord


class DataProc():
    def __init__(self):
        super(DataProc, self).__init__()
        self.data_path = './recommendation/dataset/data-3.tfrecord'
        self.meta_path = './recommendation/movie_title_by_index.json'

    def parse_df_element(self, element):
        parser = {
            "userIndex": tf.io.FixedLenFeature([], tf.int64),
            "movieIndices": tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64),
            "timestamps": tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64)
        }
        content = tf.io.parse_single_example(element, parser)
        return content['userIndex'], content['movieIndices'], content['timestamps']

    def data_proc(self):
        data_path = self.data_path
        meta_path = self.meta_path
        dataset = tf.data.TFRecordDataset([data_path])
        parsed_tf_records = dataset.map(self.parse_df_element)

        df = pd.DataFrame(parsed_tf_records.as_numpy_iterator(),
                          columns=['user', 'movies', 'timestamps'])

        df['movies'] = df['movies'].apply(lambda x: x.tolist())
        df['timestamps'] = df['timestamps'].apply(lambda x: x.tolist())
        df['seq_length'] = df['movies'].str.len()
        # drop all users with < 5 items in the sequence
        df = df[df['seq_length'] >= 5][['user', 'movies', 'timestamps']].reset_index(drop=True)

        df_movie = pd.read_json(meta_path, typ='series').to_frame()
        df_movie.columns = ['title']
        df_movie['movie_id'] = list(range(int(df_movie.count())))
        df_movie = df_movie[['movie_id', 'title']]
        n_item = df_movie.count()[0]

        return df, n_item