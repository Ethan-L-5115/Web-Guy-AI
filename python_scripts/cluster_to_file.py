import csv

def write_clusters_to_file(cluster_list, file_path):
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'centroid', 'filepaths'])
        for i, cluster in enumerate(cluster_list):
            centroid = cluster[0]
            filepaths = cluster[1:]
            writer.writerow([i+1, centroid, *filepaths])
