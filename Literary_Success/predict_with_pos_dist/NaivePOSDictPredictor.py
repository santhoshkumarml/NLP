__author__ = 'santhosh'

content = 'meta_dict='
from sklearn.cluster import KMeans
import numpy

CLASS = 'class'
TAGS = 'TAGS'
SUCCESS_PATTERN = 'SUCCESS'
FAILURE_PATTERN = 'FAILURE'

with open('../novel_meta.meta', 'r') as f:
    meta_dict = dict()
    content = content+f.readline()
    exec(content)

diff_genres = meta_dict.keys()
for genre in diff_genres:
    meta_dict_for_genre = meta_dict[genre]
    file_names = [file_name for file_name in meta_dict_for_genre]
    diff_pos = list(set([pos_tag for file_name in file_names for pos_tag in meta_dict_for_genre[file_name][TAGS]]))

    n_samples = len(file_names)
    n_features = len(diff_pos)
    data = numpy.zeros(shape=(n_samples,n_features))

    sample_idx = 0
    for file_name in file_names:
        feature_idx = 0
        for pos_tag in diff_pos:
            if pos_tag not in meta_dict_for_genre[file_name][TAGS]:
                meta_dict_for_genre[file_name][TAGS][pos_tag] = 0.0
            data[sample_idx][feature_idx] = meta_dict_for_genre[file_name][TAGS][pos_tag]
            feature_idx+=1
        sample_idx+=1

    kmeans = KMeans(n_clusters=2)
    predicted_values = kmeans.fit_predict(data)

    ONE = {SUCCESS_PATTERN:[], FAILURE_PATTERN:[]}
    ZERO = {SUCCESS_PATTERN:[], FAILURE_PATTERN:[]}
    for sample_idx in range(n_samples):
        file_name = file_names[sample_idx]
        if predicted_values[sample_idx]:
            if meta_dict_for_genre[file_name][CLASS] == SUCCESS_PATTERN:
                ONE[SUCCESS_PATTERN].append(file_name)
            else:
                ONE[FAILURE_PATTERN].append(file_name)
        else:
            if meta_dict_for_genre[file_name][CLASS] == SUCCESS_PATTERN:
                ZERO[SUCCESS_PATTERN].append(file_name)
            else:
                ZERO[FAILURE_PATTERN].append(file_name)
    print '-------------------------------------------------------------'
    print 'GENRE:', genre
    print len(ONE[SUCCESS_PATTERN]), len(ONE[FAILURE_PATTERN])
    print len(ZERO[SUCCESS_PATTERN]), len(ZERO[FAILURE_PATTERN])
    print '-------------------------------------------------------------'





